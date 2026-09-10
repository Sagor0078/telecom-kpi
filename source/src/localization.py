"""
Root-cause localization scored against ground truth on all three legs (notebook §15).

§14 scored detection and early warning only. §14.5 records the gap this module closes: the
localization ranking that anchors 1 and 3 is loaded by every leg's loader and was never scored,
so "RCA rigor" was claimed on the leg assigned to carry it without a number behind it. The
fixed-k metrics were already implemented in `src.rca_metrics`; what was missing was the per-leg
split construction, the attribution point, and the mapping from each dataset's own ground-truth
encoding into a common ranking space.

One method, three legs, per §0.18.2. Every leg fits the same XGBoost from `src.models`
(the best detector on all three legs, §14.2, which is what keeps SHAP's exact TreeExplainer path
available), aggregates |phi| over each channel's mean/std/last features exactly as §8 does on SMD,
ranks channels, and scores the ranking with `src.rca_metrics.score_ranking`.

Two asymmetries are deliberate and are the reason this is a module rather than a loop:

  * **Where the attribution is taken.** On an early-warning leg (TelecomTS, SMD) the model is
    explained at the last window before onset, because that is the window the warning fires on.
    On a detection leg (RCAEval) the pre-injection window contains no fault signal at all
    (§14.4), so explaining it would be explaining noise; the attribution is taken at the first
    window lying wholly after the injection, which is the window the detection fires on. You
    explain the decision the model actually makes on that leg.
  * **What the ground truth names.** SMD gives dimension indices and TelecomTS gives named
    channels, both of which map directly onto ranked features. RCAEval names a root-cause
    *service*, which owns several metric columns, so its ground-truth set is every common column
    belonging to that service. The chance floor is computed per event against that event's own
    |G_e| and dimension count, so the legs stay comparable despite the difference.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import rca_metrics

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.9, eval_metric="logloss", tree_method="hist", n_jobs=-1)


@dataclass
class Event:
    """One ground-truth incident, with the window its explanation is taken from."""

    event_id: str
    block: str                 # 'train' (in-sample, contrast only), 'eval' or 'calib'
    features: np.ndarray       # (3 * feat,) scaled mean/std/last vector for the chosen window
    gt: set                    # 1-indexed ground-truth channel set
    kind: str = ""             # anomaly type / fault type, for per-type breakdowns


@dataclass
class Leg:
    name: str
    feat: int
    channels: list
    events: list
    train_tab: np.ndarray
    train_y: np.ndarray
    calib_tab: np.ndarray = None      # for the conformal layer (§15.3)
    calib_y: np.ndarray = None
    eval_tab: np.ndarray = None
    eval_y: np.ndarray = None
    # (n, window, feat) sequences, populated only when with_raw=True. The tree/SHAP path never
    # needs them and they are the bulk of the memory, so they are off by default.
    train_raw: np.ndarray = None
    eval_raw: np.ndarray = None
    notes: dict = field(default_factory=dict)


# ------------------------------------------------------------------ shared helpers

def tab_at(arr: np.ndarray, t: int, window: int) -> np.ndarray:
    """The mean/std/last feature vector for the window ending at index `t`.

    `make_windows` builds these on a strided grid; an event's onset almost never lands on that
    grid, and re-running the grid at stride 1 to hit one row would cost the whole series. This
    computes the single row directly, which is what the models were trained on either way.
    """
    w = arr[t - window + 1: t + 1]
    return np.concatenate([w.mean(0), w.std(0), w[-1]])


def _fit_xgb(train_tab: np.ndarray, train_y: np.ndarray, seed: int):
    from xgboost import XGBClassifier
    train_y = np.asarray(train_y).astype(int)
    pos = int(train_y.sum())
    neg = len(train_y) - pos
    if pos == 0 or neg == 0:
        raise ValueError(f"training block is single-class (pos={pos}, neg={neg})")
    est = XGBClassifier(scale_pos_weight=neg / max(pos, 1), random_state=seed, **XGB_PARAMS)
    est.fit(train_tab, train_y)
    return est


def shap_ranking(explainer, features: np.ndarray, feat: int) -> tuple[list[int], np.ndarray]:
    """Rank channels by |phi| aggregated over their mean/std/last features (as §8 does on SMD)."""
    sv = explainer.shap_values(features.reshape(1, -1))
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv).reshape(-1)
    agg = np.abs(sv[:feat]) + np.abs(sv[feat:2 * feat]) + np.abs(sv[2 * feat:3 * feat])
    return (np.argsort(-agg) + 1).tolist(), agg


def score_leg(leg: Leg, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Fit, explain and score every event on one leg. One row per event."""
    import shap

    t0 = time.perf_counter()
    est = _fit_xgb(leg.train_tab, leg.train_y, seed)
    explainer = shap.TreeExplainer(est)
    if verbose:
        print(f"  {leg.name}: fitted XGBoost on {len(leg.train_tab)} windows "
              f"({time.perf_counter() - t0:.1f}s), scoring {len(leg.events)} events")

    rows = []
    for ev in leg.events:
        ranking, agg = shap_ranking(explainer, ev.features, leg.feat)
        scored = rca_metrics.score_ranking(ranking, ev.gt)
        chance = rca_metrics.random_ranking_expected_scores(leg.feat, ev.gt, seed=seed)
        k = len(ev.gt)
        rows.append({
            "leg": leg.name, "event": ev.event_id, "block": ev.block, "kind": ev.kind,
            "seed": seed, "n_dims": leg.feat, "n_gt": k,
            "top_k_pred": [leg.channels[i - 1] for i in ranking[:max(k, 3)]],
            "gt_channels": sorted(leg.channels[i - 1] for i in ev.gt),
            # k=|G_e| precision=recall, the view §8 reports on SMD, kept for comparability
            "precision_at_g": len(set(ranking[:k]) & ev.gt) / k,
            **{f"chance_{m}": v for m, v in chance.items()},
            **scored,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, k_report: int = 5) -> pd.DataFrame:
    """Out-of-sample (EVAL+CALIB) means per leg, against the matched chance floor."""
    oos = df[df.block != "train"]
    out = []
    for leg, g in oos.groupby("leg", sort=False):
        row = {"leg": leg, "events": len(g), "n_dims": int(g.n_dims.iloc[0]),
               "mean_|G|": round(g.n_gt.mean(), 2)}
        for m in ["mrr", f"hit@{k_report}", f"precision@{k_report}",
                  f"recall@{k_report}", f"ndcg@{k_report}"]:
            row[m] = round(g[m].mean(), 3)
            row[f"{m} (chance)"] = round(g[f"chance_{m}"].mean(), 3)
        row["precision@|G|"] = round(g.precision_at_g.mean(), 3)
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ per-leg construction

def _tab_windows(arr: np.ndarray, window: int, horizon: int | None = None,
                 labels: np.ndarray | None = None, stride: int = 1, with_raw: bool = False):
    """Window grid: (X_tab, y, idxs), plus the raw tensor only when asked for.

    `src.data.make_windows` and its two siblings also materialise the raw sequences, which the
    sequence models need and SHAP does not. On TelecomTS's 643k samples that tensor is the
    difference between a few hundred MB and several GB, so the tree path builds its own grid.
    """
    n = len(arr)
    last_t = n - (horizon if horizon else 0) - 1
    idxs = np.arange(window - 1, last_t + 1, stride)
    feat = arr.shape[1]
    if len(idxs) == 0:
        empty_raw = np.zeros((0, window, feat), np.float32) if with_raw else None
        return np.zeros((0, feat * 3)), np.zeros(0, int), np.zeros(0, int), empty_raw
    means, stds, lasts = (np.zeros((len(idxs), feat)) for _ in range(3))
    raw = np.zeros((len(idxs), window, feat), dtype=np.float32) if with_raw else None
    for j, t in enumerate(idxs):
        w = arr[t - window + 1: t + 1]
        means[j], stds[j], lasts[j] = w.mean(0), w.std(0), w[-1]
        if with_raw:
            raw[j] = w
    X = np.concatenate([means, stds, lasts], axis=1)
    if horizon and labels is not None:
        y = np.array([labels[t + 1: t + horizon + 1].max() for t in idxs])
    elif labels is not None:
        y = labels[idxs]
    else:
        y = np.zeros(len(idxs), dtype=int)
    return X, y.astype(int), idxs, raw


def _chronological_blocks(items: list, fractions=(0.6, 0.2)) -> dict:
    n = len(items)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return {"train": items[:a], "eval": items[a:b], "calib": items[b:]}


def build_telecomts_leg(sessions: list, stride: int = 10, stratify: bool = True,
                        with_raw: bool = False, verbose: bool = True) -> Leg:
    """TelecomTS: session-level 60/20/20, 3 s window, 5 s early-warning horizon (§14.1).

    `stratify=True` splits within each anomaly type instead of taking sessions in time order.
    Reconstructed session lengths are extremely skewed (median 576 samples, max 99,968), so a
    plain chronological 60/20/20 by session count puts 0.499 of EVAL's windows in the positive
    class against an overall rate of 0.045, and leaves whole anomaly types absent from a block.
    Sessions are independent single-scenario recordings, so a session-level split is leakage-safe
    whatever order it is taken in; this is the same argument §14.4 makes for RCAEval cases, and
    the same correction, applied to the unit this leg's ground truth is defined over.
    """
    from . import telecomts as T

    window = int(T.WINDOW_SEC * T.SAMPLE_HZ)
    horizon = int(T.HORIZON_SEC * T.SAMPLE_HZ)
    if stratify:
        blocks = {"train": [], "eval": [], "calib": []}
        keyed = {}
        for sess in sessions:
            keyed.setdefault(sess.segments[0][3] if sess.segments else "", []).append(sess)
        for key in sorted(keyed):
            for name, sel in _chronological_blocks(keyed[key]).items():
                blocks[name].extend(sel)
    else:
        blocks = _chronological_blocks(sessions)
    scaler = StandardScaler().fit(np.vstack([s.values for s in blocks["train"]]))

    grids, events = {}, []
    for name, sess in blocks.items():
        tabs, ys, raws = [], [], []
        for s in sess:
            arr = scaler.transform(s.values)
            X, y, _, raw = _tab_windows(arr, window, horizon, s.labels, stride, with_raw)
            if len(X):
                tabs.append(X)
                ys.append(y)
                if with_raw:
                    raws.append(raw)
            # The warning fires on the last window before onset, so that is the window explained.
            for (a, b, affected, atype) in s.segments:
                if not affected or a - 1 < window - 1:
                    continue
                events.append(Event(
                    event_id=f"{s.session_id}:{a}-{b}", block=name,
                    features=tab_at(arr, a - 1, window),
                    gt={i + 1 for i in affected}, kind=str(atype)))
        grids[name] = (np.vstack(tabs), np.concatenate(ys),
                       np.concatenate(raws) if raws else None) if tabs else (
            np.zeros((0, T.FEAT * 3)), np.zeros(0, int), None)

    leg = Leg(name="TelecomTS", feat=T.FEAT, channels=list(T.CHANNELS),
              events=events,
              train_tab=grids["train"][0], train_y=grids["train"][1],
              eval_tab=grids["eval"][0], eval_y=grids["eval"][1],
              calib_tab=grids["calib"][0], calib_y=grids["calib"][1],
              train_raw=grids["train"][2], eval_raw=grids["eval"][2],
              notes={"sessions": {k: len(v) for k, v in blocks.items()},
                     "window": window, "horizon": horizon, "stride": stride,
                     # The §14 helpers re-window these sessions at other horizons, so they
                     # need the same block assignment and the same fitted scaler.
                     "blocks": blocks, "scaler": scaler})
    if verbose:
        _report(leg, blocks)
    return leg


def build_rcaeval_leg(cases: list, stride: int = 10, with_raw: bool = False,
                      verbose: bool = True) -> Leg:
    """RCAEval: fault-stratified case split 60/20/20, 60 s window, detection target (§14.1/§14.4).

    Column space is the intersection across cases, because the metric set varies by system
    (49-376 columns) and a ranking is only comparable across cases over a shared space.
    """
    from . import rcaeval as R

    window = R.WINDOW_SEC * R.SAMPLE_HZ
    common = sorted(set.intersection(*[set(c.columns) for c in cases]))
    col_of = {c: i + 1 for i, c in enumerate(common)}

    # Cases are independent experiments, so the split is stratified rather than chronological
    # (the §14.4 method correction). Stratifying on fault type alone is not enough *for a
    # localization evaluation*: case ids sort by service ('re1ob_adservice_cpu_1'), so a
    # fault-stratified split taken in id order segregates root-cause services across blocks
    # exactly as the chronological split segregated fault types. The model then never trains on
    # the services it is scored on, and its ranking lands below the chance floor because it
    # confidently names services it has seen. Stratify on the (fault, service) cell, which is
    # the unit the ground truth is defined over, and split that cell's repetitions.
    blocks = {"train": [], "eval": [], "calib": []}
    cells = sorted({(c.fault, c.root_cause_service) for c in cases})
    for fault, service in cells:
        group = sorted([c for c in cases if c.fault == fault and c.root_cause_service == service],
                       key=lambda c: c.case)
        for name, sel in _chronological_blocks(group).items():
            blocks[name].extend(sel)

    scaler = StandardScaler().fit(np.vstack([c.select(common) for c in blocks["train"]]))

    grids, events, skipped = {}, [], []
    for name, cs in blocks.items():
        tabs, ys, raws = [], [], []
        for c in cs:
            arr = scaler.transform(c.select(common))
            X, y, _, raw = _tab_windows(arr, window, None, c.labels, stride, with_raw)
            if len(X):
                tabs.append(X)
                ys.append(y)
                if with_raw:
                    raws.append(raw)
            gt = {col_of[c.columns[i]] for i in c.root_cause_columns() if c.columns[i] in common}
            onset = int(np.argmax(c.labels))
            # First window lying wholly after the injection: the window detection fires on.
            t = onset + window - 1
            if not gt or t >= len(arr):
                skipped.append(c.case)
                continue
            events.append(Event(event_id=c.case, block=name, features=tab_at(arr, t, window),
                                gt=gt, kind=c.fault))
        grids[name] = (np.vstack(tabs), np.concatenate(ys),
                       np.concatenate(raws) if raws else None) if tabs else (
            np.zeros((0, len(common) * 3)), np.zeros(0, int), None)

    leg = Leg(name="RCAEval", feat=len(common), channels=common,
              events=events,
              train_tab=grids["train"][0], train_y=grids["train"][1],
              eval_tab=grids["eval"][0], eval_y=grids["eval"][1],
              calib_tab=grids["calib"][0], calib_y=grids["calib"][1],
              train_raw=grids["train"][2], eval_raw=grids["eval"][2],
              notes={"cases": {k: len(v) for k, v in blocks.items()},
                     "common_columns": len(common), "window": window, "stride": stride,
                     "skipped_no_gt_or_runway": skipped})
    if verbose:
        _report(leg, blocks)
        if skipped:
            print(f"    {len(skipped)} case(s) skipped (root-cause service absent from the "
                  f"common column set, or no full post-injection window)")
    return leg


def build_smd_leg(machine=None, verbose: bool = True) -> Leg:
    """SMD: the existing chronological split of §0.6, re-scored here with the fixed-k metrics.

    §8 reported only the k=|G_e| precision=recall view on this leg. Re-running it through the same
    code path as the other two legs is what makes the three numbers comparable.
    """
    from . import data as D

    m = D.load_machine() if machine is None else machine
    sp = D.build_split(m)
    tab_by_block = {"train": sp.sup_tab, "eval": sp.ev_tab, "calib": sp.cal_tab}

    events = []
    for (s, e, dims) in m.segments:
        if (s - 1) not in sp.block_index:
            continue
        name, row = sp.block_index[s - 1]
        events.append(Event(event_id=f"{s}-{e}", block=name,
                            features=tab_by_block[name][row], gt=set(dims), kind=""))

    leg = Leg(name="SMD", feat=D.FEAT, channels=[f"dim{i}" for i in range(1, D.FEAT + 1)],
              events=events,
              train_tab=sp.train_tab_full, train_y=sp.train_y_full,
              eval_tab=sp.ev_tab, eval_y=sp.ev_y,
              calib_tab=sp.cal_tab, calib_y=sp.cal_y,
              notes={"machine": m.machine_id})
    if verbose:
        _report(leg, None)
    return leg


def _report(leg: Leg, blocks) -> None:
    by_block = pd.Series([e.block for e in leg.events]).value_counts().to_dict() if leg.events else {}
    print(f"  {leg.name}: {leg.feat} channels | "
          f"windows train/eval/calib = {len(leg.train_tab)}/{len(leg.eval_tab)}/{len(leg.calib_tab)} | "
          f"positive rate {leg.train_y.mean():.3f}/{leg.eval_y.mean():.3f}/{leg.calib_y.mean():.3f}")
    print(f"    events by block: {by_block}")


# ------------------------------------------------------------------ conformal layer (§15.3)

def conformal_layer(leg: Leg, alpha: float = 0.10, seed: int = 0) -> dict:
    """Split-conformal LAC on one leg: coverage and abstention on EVAL, calibrated on CALIB.

    §14.5 records that CALIB blocks were built and held out for TelecomTS and RCAEval but that
    §7's wrapper had only ever been run on SMD. This is the same wrapper, unchanged: the
    nonconformity score is 1 - P(y | x), the threshold is the ceil((n+1)(1-alpha))/n empirical
    quantile of the calibration scores, and a two-element set is the system's explicit abstention.
    """
    est = _fit_xgb(leg.train_tab, leg.train_y, seed)
    p_cal = est.predict_proba(leg.calib_tab)
    p_ev = est.predict_proba(leg.eval_tab)
    y_cal = np.asarray(leg.calib_y).astype(int)
    y_ev = np.asarray(leg.eval_y).astype(int)

    scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = float(np.quantile(scores, level, method="higher"))

    keep = (1.0 - p_ev) <= q                     # (n_eval, 2) membership per class
    covered = keep[np.arange(len(y_ev)), y_ev]
    sizes = keep.sum(axis=1)
    return {
        "leg": leg.name, "seed": seed, "alpha": alpha, "target_coverage": 1 - alpha,
        "q_hat": round(q, 4), "n_calib": n, "n_eval": len(y_ev),
        "empirical_coverage": round(float(covered.mean()), 4),
        "abstain_rate": round(float((sizes == 2).mean()), 4),
        "singleton_rate": round(float((sizes == 1).mean()), 4),
        "empty_rate": round(float((sizes == 0).mean()), 4),
    }


# ------------------------------------------------------------------ multi-seed (§15.4)

def multi_seed(leg: Leg, seeds=(0, 1, 2, 3, 4), alpha: float = 0.10,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat localization and the conformal layer across seeds.

    §14.5's first bullet, and §VIII's "no statistical validation": every number in §14 came from
    a single seed. The split is fixed and only the model seed varies, so this measures the spread
    the estimator contributes, not sampling variability of the split itself, which would need
    repeated splits and is the larger job named in §15.5.
    """
    loc, conf = [], []
    for s in seeds:
        loc.append(score_leg(leg, seed=s, verbose=False).assign(seed=s))
        conf.append(conformal_layer(leg, alpha=alpha, seed=s))
        if verbose:
            print(f"    seed {s} done")
    return pd.concat(loc, ignore_index=True), pd.DataFrame(conf)


def seed_spread(loc: pd.DataFrame, metrics=("mrr", "precision@5", "recall@5", "ndcg@5")) -> pd.DataFrame:
    """Mean and standard deviation across seeds of each leg's out-of-sample metrics."""
    oos = loc[loc.block != "train"]
    rows = []
    for leg, g in oos.groupby("leg", sort=False):
        per_seed = g.groupby("seed")[list(metrics) + ["precision_at_g"]].mean()
        row = {"leg": leg, "seeds": len(per_seed), "events": len(g) // len(per_seed)}
        for m in list(metrics) + ["precision_at_g"]:
            row[m] = f"{per_seed[m].mean():.3f} ± {per_seed[m].std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
