"""
Re-run the §14 three-dataset benchmark and write every artifact the figures depend on.

The original §14 run was never committed: `src/figures.py` reads
`utils/telecomts_model_comparison.csv` and `utils/rcaeval_model_comparison.csv` and neither
existed, nor did the prediction pickles, so §14.2's tables could not be regenerated and the
TelecomTS session reconstruction no longer matched what §14.1 records (99 sessions / 1,077,076
samples against 168 / 643,284). This script makes the whole section reproducible from the
repository, and it uses the *same* splits as §15 so detection, localization and calibration are
all reported on one split per leg rather than three.

    python -m src.run_section14 [scratch_dir]

Writes into utils/: the two model-comparison tables, the pre-onset runway, the horizon sweep, the
per-anomaly-type breakdown and the case-study arrays; plus tts_preds.pkl / rcae_preds.pkl into
the scratch directory for the curve and reliability figures.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import localization as L
from . import models
from . import telecomts as T
from .rcaeval import load_case, load_case_index
from .telecomts import load_sessions, preonset_runway

UTILS = Path("utils")
HORIZONS_SEC = (1, 2, 5, 10, 20, 30)


def _load_rcaeval_ob(cache_dir: str = ".cache/rcaeval") -> list:
    idx = load_case_index(cache_dir)
    sel = idx[(idx.suite == "RE1") & (idx.system_name == "Online Boutique")]
    cases = []
    for _, row in sel.iterrows():
        try:
            cases.append(load_case(row, cache_dir, idx))
        except ValueError:
            pass      # 2 of 125 carry an injection timestamp outside their metric window
    return cases


def horizon_sweep(sessions: list, stride: int = 10, seed: int = 0) -> pd.DataFrame:
    """AUPRC against early-warning horizon, with and without label contamination.

    `auprc_all` counts every window whose horizon overlaps an anomaly, including windows in which
    the anomaly is *already under way*: that measures detection. `auprc_clean` keeps only windows
    that end before onset, which is the anticipation the paper claims. The gap between the two is
    the reason §0.18.7 calls this the decisive experiment.
    """
    window = int(T.WINDOW_SEC * T.SAMPLE_HZ)
    notes = L.build_telecomts_leg(sessions, stride=stride, verbose=False).notes
    blocks, scaler = notes["blocks"], notes["scaler"]
    rows = []
    for h_sec in HORIZONS_SEC:
        horizon = int(h_sec * T.SAMPLE_HZ)
        parts = {}
        for name, sess in blocks.items():
            tabs, ys, clean = [], [], []
            for s in sess:
                arr = scaler.transform(s.values)
                X, y, idxs, _ = L._tab_windows(arr, window, horizon, s.labels, stride)
                if not len(X):
                    continue
                # A window is genuinely pre-onset when its own last sample is still normal.
                pre = s.labels[idxs] == 0
                tabs.append(X)
                ys.append(y)
                clean.append(pre)
            if tabs:
                parts[name] = (np.vstack(tabs), np.concatenate(ys), np.concatenate(clean))
        if "train" not in parts or "eval" not in parts:
            continue
        Xtr, ytr, _ = parts["train"]
        Xev, yev, pre = parts["eval"]
        if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            continue
        est = L._fit_xgb(Xtr, ytr, seed)
        p = est.predict_proba(Xev)[:, 1]
        from sklearn.metrics import average_precision_score
        auprc_all = average_precision_score(yev, p)
        m = pre | (yev == 0)          # drop windows already inside an anomaly
        auprc_clean = (average_precision_score(yev[m], p[m])
                       if len(np.unique(yev[m])) > 1 else np.nan)
        rows.append({"horizon_sec": h_sec, "auprc_all": round(float(auprc_all), 4),
                     "auprc_clean": round(float(auprc_clean), 4),
                     "n_eval": int(len(yev)), "pos_rate": round(float(yev.mean()), 4),
                     "n_eval_clean": int(m.sum()),
                     "pos_rate_clean": round(float(yev[m].mean()), 4)})
        print(f"    H={h_sec:>2}s  AUPRC all={auprc_all:.3f}  clean={auprc_clean:.3f}", flush=True)
    return pd.DataFrame(rows)


def per_anomaly_type(leg: L.Leg, stride: int = 10, seed: int = 0) -> pd.DataFrame:
    """XGBoost AUPRC for each anomaly type's windows against all negatives.

    Each row scores one fault mode's positives against the shared negative pool, so the AUPRCs are
    comparable to each other and to the pooled number, which a per-type re-split would not be.
    """
    from sklearn.metrics import average_precision_score
    window = int(T.WINDOW_SEC * T.SAMPLE_HZ)
    horizon = int(T.HORIZON_SEC * T.SAMPLE_HZ)
    est = L._fit_xgb(leg.train_tab, leg.train_y, seed)
    scaler = leg.notes["scaler"]

    tabs, ys, kinds = [], [], []
    for s in leg.notes["blocks"]["eval"]:
        X, y, idxs, _ = L._tab_windows(scaler.transform(s.values), window, horizon,
                                       s.labels, stride)
        if not len(X):
            continue
        kind = np.empty(len(idxs), dtype=object)
        kind[:] = ""
        for (a, b, _aff, atype) in s.segments:
            hit = (idxs + 1 <= b) & (idxs + horizon >= a)
            kind[hit] = str(atype)
        tabs.append(X)
        ys.append(y)
        kinds.append(kind)
    X = np.vstack(tabs)
    y = np.concatenate(ys)
    kind = np.concatenate(kinds)
    p = est.predict_proba(X)[:, 1]

    rows = []
    neg = y == 0
    for atype in sorted({k for k in kind if k}):
        sel = (y == 1) & (kind == atype)
        if not sel.any():
            continue
        m = sel | neg
        if len(np.unique(y[m])) < 2:
            continue
        rows.append({"anomaly_type": atype,
                     "auprc": round(float(average_precision_score(y[m], p[m])), 4),
                     "n_windows": int(sel.sum()),
                     "base_rate": round(float(y[m].mean()), 5)})
    return pd.DataFrame(rows).sort_values("auprc", ascending=False)


def case_study(sessions: list, out: Path) -> dict:
    """Save one incident's raw traces for the case-study figure: the segment with the most
    pre-onset runway among those whose affected-KPI set is small enough to read on a plot."""
    best = None
    for s in sessions:
        prev_end = -1
        for (a, b, affected, atype) in s.segments:
            runway = a - prev_end - 1
            prev_end = b
            if not affected or len(affected) > 6:
                continue
            if best is None or runway > best[0]:
                best = (runway, s, a, b, affected, atype)
    if best is None:                       # no small-set segment; fall back to the longest runway
        for s in sessions:
            for (a, b, affected, atype) in s.segments:
                if affected and (best is None or (b - a) > (best[3] - best[2])):
                    best = (0, s, a, b, affected, atype)
    _, s, a, b, affected, atype = best
    lo, hi = max(0, a - 600), min(s.n, b + 300)
    np.savez(out, values=s.values[lo:hi], labels=s.labels[lo:hi],
             channels=np.array(T.CHANNELS, dtype=object),
             affected=np.array(affected, dtype=int),
             seg_start=a - lo, seg_end=b - lo, atype=str(atype))
    return {"session": s.session_id, "type": str(atype), "segment": [int(a), int(b)],
            "affected": [T.CHANNELS[i] for i in affected]}


def main(scratch: str = ".") -> None:
    UTILS.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    summary = {}

    print("== TelecomTS ==", flush=True)
    sessions = load_sessions(cache_dir=".cache")
    tts = L.build_telecomts_leg(sessions, with_raw=True)
    preonset_runway(sessions).to_csv(UTILS / "telecomts_preonset_runway.csv", index=False)

    t0 = time.time()
    res_t = models.run_comparison(tts.train_tab, tts.train_raw, tts.train_y,
                                  tts.eval_tab, tts.eval_raw, tts.eval_y)
    res_t.table.to_csv(UTILS / "telecomts_model_comparison.csv", index=False)
    pickle.dump({"ev_y": tts.eval_y, "preds": res_t.predictions},
                open(scratch / "tts_preds.pkl", "wb"))
    print(f"  TelecomTS hierarchy: {time.time() - t0:.0f}s", flush=True)
    summary["telecomts"] = {"windows": {k: int(v) for k, v in
                                        zip(("train", "eval", "calib"),
                                            (len(tts.train_tab), len(tts.eval_tab),
                                             len(tts.calib_tab)))},
                            "pos_rate": {"train": round(float(tts.train_y.mean()), 4),
                                         "eval": round(float(tts.eval_y.mean()), 4),
                                         "calib": round(float(tts.calib_y.mean()), 4)},
                            "sessions": tts.notes["sessions"],
                            "best": res_t.table.iloc[0][["model", "auroc", "auprc", "ece"]].to_dict()}

    print("== RCAEval ==", flush=True)
    cases = _load_rcaeval_ob()
    rce = L.build_rcaeval_leg(cases, with_raw=True)
    t0 = time.time()
    res_r = models.run_comparison(rce.train_tab, rce.train_raw, rce.train_y,
                                  rce.eval_tab, rce.eval_raw, rce.eval_y)
    res_r.table.to_csv(UTILS / "rcaeval_model_comparison.csv", index=False)
    pickle.dump({"ev_y": rce.eval_y, "preds": res_r.predictions},
                open(scratch / "rcae_preds.pkl", "wb"))
    print(f"  RCAEval hierarchy: {time.time() - t0:.0f}s", flush=True)
    summary["rcaeval"] = {"windows": {"train": len(rce.train_tab), "eval": len(rce.eval_tab),
                                      "calib": len(rce.calib_tab)},
                          "pos_rate": {"train": round(float(rce.train_y.mean()), 4),
                                       "eval": round(float(rce.eval_y.mean()), 4),
                                       "calib": round(float(rce.calib_y.mean()), 4)},
                          "cases": rce.notes["cases"], "common_columns": rce.notes["common_columns"],
                          "best": res_r.table.iloc[0][["model", "auroc", "auprc", "ece"]].to_dict()}

    print("== horizon sweep ==", flush=True)
    horizon_sweep(sessions).to_csv(UTILS / "telecomts_horizon_sweep.csv", index=False)

    print("== per anomaly type ==", flush=True)
    per_anomaly_type(tts).to_csv(UTILS / "telecomts_per_anomaly_type.csv", index=False)

    print("== case study ==", flush=True)
    summary["case_study"] = case_study(sessions, UTILS / "telecomts_case_study.npz")

    json.dump(summary, open(UTILS / "section14_summary.json", "w"), indent=2, default=str)
    print("\nwrote:", flush=True)
    for f in sorted(UTILS.glob("*")):
        print("   ", f, flush=True)
    print("   ", scratch / "tts_preds.pkl")
    print("   ", scratch / "rcae_preds.pkl")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
