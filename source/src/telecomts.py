"""
TelecomTS loading, session reconstruction, and the leakage-safe chronological split.

TelecomTS (Feng et al., arXiv:2510.06063; HF dataset `AliMaatouk/TelecomTS`) is the **primary
5G-domain benchmark** in this project's dataset strategy — see the main notebook's §0.18.2. It
ships as 128-sample windows at 10 Hz, but those windows are *sliding, not independent*: consecutive
windows advance by a 32-sample (3.2 s) stride and their overlapping regions are identical across
all 18 channels. This module exploits that to reconstruct the continuous underlying series, which
is what makes an early-warning target definable at all (§0.18.7).

Interface deliberately mirrors `src.data` (SMD) so the same modelling code runs on both without
per-dataset branching, per §0.18.2's "the method must be identical across datasets" rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HF_BASE = ("https://huggingface.co/datasets/AliMaatouk/TelecomTS/resolve/"
           "refs%2Fconvert%2Fparquet/default/train")
SHARDS = ("0000.parquet", "0001.parquet", "0002.parquet")

# 18 channels in dataset order. Two are categorical strings ('UDP', 'TCP', ...).
CHANNELS = [
    "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB", "UL_SNR",
    "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer", "PRBs_DL_Current", "PRBs_UL_Current",
    "PRB_Utilization_DL", "PRB_Utilization_UL", "UL_Protocol", "UL_NumberOfPackets",
    "DL_Protocol", "DL_NumberOfPackets",
]
CATEGORICAL = ("UL_Protocol", "DL_Protocol")
FEAT = len(CHANNELS)
SAMPLE_HZ = 10
WINDOW_SAMPLES = 128
SCENARIO_KEYS = ("zone", "application", "mobility", "congestion")

# Seconds-scale defaults. SMD's W=20 / H=30 are *minutes* on 1-sample-per-minute server telemetry;
# 5G radio degradation evolves in seconds, so the horizon is re-parameterised rather than copied
# (notebook §0.18.3 item 5 — a domain correction, not a concession).
WINDOW_SEC = 3.0
HORIZON_SEC = 5.0


@dataclass
class TelecomTSSession:
    """One contiguous, single-scenario recording reconstructed from overlapping windows."""

    session_id: str
    scenario: dict
    values: np.ndarray            # (N, 18) float; categorical channels label-encoded
    labels: np.ndarray            # (N,) int 0/1 per timestamp
    types: np.ndarray             # (N,) object; anomaly type name or '' per timestamp
    segments: list                # [(start, end, [affected channel idx], anomaly_type), ...]
    start_index: int              # absolute 10 Hz sample index of values[0]
    categories: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def minutes(self) -> float:
        return self.n / SAMPLE_HZ / 60


def _shard_path(shard: str, cache_dir: str | Path) -> Path:
    """Return a local path for `shard`, downloading it into `cache_dir` on first use."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"telecomts_{shard}"
    if not dest.exists():
        import urllib.request
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{HF_BASE}/{shard}", tmp)
        tmp.rename(dest)
    return dest


def _encode_categoricals(raw: dict, categories: dict) -> None:
    """Assign integer codes to categorical channel values, extending `categories` in place."""
    for ch in CATEGORICAL:
        mapping = categories.setdefault(ch, {})
        for v in raw[ch]:
            if v not in mapping:
                mapping[v] = len(mapping)


def load_sessions(shards=SHARDS, cache_dir: str | Path = ".cache",
                  local_paths: list[str | Path] | None = None,
                  min_samples: int = WINDOW_SAMPLES) -> list[TelecomTSSession]:
    """Reconstruct continuous single-scenario sessions from TelecomTS's overlapping windows.

    Windows are keyed by absolute 10 Hz sample index derived from `start_time`, so overlapping
    windows collapse onto the same timestamps. Overlaps are verified identical (see
    `verify_overlap_consistency`), so first-writer-wins is lossless rather than arbitrary.

    Raises on an unexpected channel set or window length rather than silently truncating, matching
    the `src.data` loader's fail-loud contract.
    """
    paths = [Path(p) for p in local_paths] if local_paths else [_shard_path(s, cache_dir) for s in shards]

    samples: dict[int, list] = {}
    anomalies: dict[int, tuple] = {}
    scen_of: dict[int, tuple] = {}
    categories: dict[str, dict] = {}

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["start_time", "KPIs", "anomalies", "labels"]):
            df = batch.to_pandas()
            for st, kpis, ano, lab in zip(df.start_time, df.KPIs, df.anomalies, df.labels):
                if set(kpis) != set(CHANNELS):
                    raise ValueError(f"{path.name}: unexpected channel set {sorted(set(kpis))}")
                n = len(kpis[CHANNELS[0]])
                if n != WINDOW_SAMPLES:
                    raise ValueError(f"{path.name}: window has {n} samples, expected {WINDOW_SAMPLES}")

                _encode_categoricals(kpis, categories)
                base = int(round(pd.Timestamp(st).timestamp() * SAMPLE_HZ))
                scen = tuple(str(lab[k]) for k in SCENARIO_KEYS)

                cols = []
                for ch in CHANNELS:
                    v = kpis[ch]
                    cols.append([categories[ch][x] for x in v] if ch in CATEGORICAL
                                else np.asarray(v, dtype=np.float64))
                block = np.asarray(cols, dtype=np.float64).T  # (128, 18)

                for i in range(n):
                    samples.setdefault(base + i, block[i])
                    scen_of.setdefault(base + i, scen)

                if ano.get("exists"):
                    d = ano.get("anomaly_duration") or {}
                    s0 = int(d.get("start", 0))
                    s1 = int(d.get("end", n - 1))
                    aff_raw = ano.get("affected_kpis")
                    aff_raw = [] if aff_raw is None else list(aff_raw)
                    affected = [CHANNELS.index(k) for k in aff_raw if k in CHANNELS]
                    for i in range(max(0, s0), min(n - 1, s1) + 1):
                        anomalies.setdefault(base + i, (str(ano.get("type", "")), tuple(affected)))

    if not samples:
        raise ValueError("no TelecomTS windows loaded")

    idx = np.fromiter(sorted(samples), dtype=np.int64)

    # A session breaks on a timestamp gap or a scenario change.
    breaks = np.nonzero(
        (np.diff(idx) != 1)
        | np.array([scen_of[a] != scen_of[b] for a, b in zip(idx[:-1], idx[1:])])
    )[0]
    bounds = np.concatenate([[0], breaks + 1, [len(idx)]])

    sessions = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        run = idx[a:b]
        if len(run) < min_samples:
            continue
        values = np.vstack([samples[i] for i in run])
        labels = np.zeros(len(run), dtype=int)
        types = np.empty(len(run), dtype=object)
        types[:] = ""
        for j, i in enumerate(run):
            if i in anomalies:
                labels[j] = 1
                types[j] = anomalies[i][0]

        segments = []
        if labels.any():
            edges = np.diff(np.concatenate([[0], labels, [0]]))
            for s, e in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1):
                affected = anomalies[run[s]][1]
                segments.append((int(s), int(e), sorted(affected), types[s]))

        scen = dict(zip(SCENARIO_KEYS, scen_of[run[0]]))
        sessions.append(TelecomTSSession(
            session_id=f"{'-'.join(scen.values())}@{run[0]}",
            scenario=scen, values=values, labels=labels, types=types,
            segments=segments, start_index=int(run[0]), categories=categories,
        ))

    sessions.sort(key=lambda s: s.start_index)
    return sessions


def verify_overlap_consistency(path: str | Path, n_pairs: int = 200) -> dict:
    """Check that overlapping windows agree on their shared timestamps.

    This is the assumption session reconstruction rests on; if it fails, first-writer-wins in
    `load_sessions` would silently pick one of two disagreeing readings.
    """
    pf = pq.ParquetFile(path)
    df = next(pf.iter_batches(batch_size=n_pairs + 1,
                              columns=["start_time", "KPIs"])).to_pandas()
    df["st"] = pd.to_datetime(df.start_time)
    df = df.sort_values("st").reset_index(drop=True)

    checked = mismatched = 0
    for (_, a), (_, b) in zip(df.iloc[:-1].iterrows(), df.iloc[1:].iterrows()):
        off = int(round((b.st - a.st).total_seconds() * SAMPLE_HZ))
        if not 0 < off < WINDOW_SAMPLES:
            continue
        checked += 1
        for ch in CHANNELS:
            if list(a.KPIs[ch])[off:] != list(b.KPIs[ch])[:WINDOW_SAMPLES - off]:
                mismatched += 1
                break
    return {"pairs_checked": checked, "pairs_mismatched": mismatched,
            "consistent": mismatched == 0}


def make_windows(arr: np.ndarray, horizon: int | None = None, label_arr: np.ndarray | None = None,
                 window: int = int(WINDOW_SEC * SAMPLE_HZ), feat: int = FEAT, stride: int = 1):
    """Early-warning windows, identical in shape/semantics to `src.data.make_windows`.

    y[t] = 1 iff an anomalous timestamp falls in (t, t+horizon], using only data up to t.

    `stride` subsamples window *origins* — at 10 Hz a stride of 1 yields ten near-identical windows
    per second, which inflates memory without adding information. It thins the window grid only;
    every window still sees all `window` consecutive samples, so no telemetry is discarded.
    """
    n = len(arr)
    last_t = n - (horizon if horizon else 0) - 1
    idxs = np.arange(window - 1, last_t + 1, stride)
    if len(idxs) == 0:
        empty = np.zeros((0, feat * 3))
        return empty, np.zeros((0, window, feat), np.float32), np.zeros(0, int), np.zeros(0, int)

    means = np.zeros((len(idxs), feat))
    stds = np.zeros((len(idxs), feat))
    lasts = np.zeros((len(idxs), feat))
    raw = np.zeros((len(idxs), window, feat), dtype=np.float32)
    for j, t in enumerate(idxs):
        w = arr[t - window + 1: t + 1]
        means[j], stds[j], lasts[j], raw[j] = w.mean(0), w.std(0), w[-1], w
    X_tab = np.concatenate([means, stds, lasts], axis=1)

    if horizon and label_arr is not None:
        y = np.array([label_arr[t + 1: t + horizon + 1].max() for t in idxs])
    else:
        y = np.zeros(len(idxs), dtype=int)
    return X_tab, raw, y, idxs


def preonset_runway(sessions: list[TelecomTSSession]) -> pd.DataFrame:
    """Per anomaly segment, how many normal samples precede onset inside its own session.

    This is the quantity §0.18.7 makes decisive: it upper-bounds the early-warning horizon that can
    honestly be evaluated. A segment starting at index 0 of its session has no pre-onset data.
    """
    rows = []
    for s in sessions:
        prev_end = -1
        for (a, b, affected, atype) in s.segments:
            rows.append({
                "session": s.session_id, "type": atype,
                "onset_index": a, "length_samples": b - a + 1,
                "runway_samples": a - prev_end - 1,
                "runway_sec": (a - prev_end - 1) / SAMPLE_HZ,
                "affected": [CHANNELS[i] for i in affected],
                **s.scenario,
            })
            prev_end = b
    return pd.DataFrame(rows)
