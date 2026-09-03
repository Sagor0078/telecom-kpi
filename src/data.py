"""
SMD (Server Machine Dataset) loading, windowing, and the leakage-safe TRAIN/EVAL/CALIB split.

Refactored out of Telecom_Trustworthy_AI_Framework.ipynb so it is not duplicated across the
notebooks/*.ipynb files added by the §0 Research Design Audit roadmap (notebooks/ 01, 02, 04, 05).
See that notebook's §0.6-§0.7 for the exact protocol this module implements.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
FEAT = 38
WINDOW = 20
HORIZON = 30
# Cut points inside the anomaly-free gaps before event 4 and before event 6 (see notebook §0.6).
CUT1, CUT2 = 18800, 23000


def _fetch(path: str) -> str:
    with urllib.request.urlopen(f"{BASE_URL}/{path}", timeout=30) as r:
        return r.read().decode()


@dataclass
class SMDMachine:
    machine_id: str
    train: np.ndarray  # (N, 38) guaranteed anomaly-free historical operation
    test: np.ndarray  # (N, 38) labeled monitoring period
    labels: np.ndarray  # (N,) int 0/1, test_label
    segments: list  # [(start, end, [1-indexed contributing dims]), ...] from interpretation_label


def load_machine(machine_id: str = "machine-1-1") -> SMDMachine:
    """Load one SMD machine's train/test/test_label/interpretation_label directly from GitHub.

    Raises on shape mismatch or unexpected feature count rather than silently truncating,
    per the §26/§27 data-loader requirements in the governing research prompt.
    """
    train = np.loadtxt(io.StringIO(_fetch(f"train/{machine_id}.txt")), delimiter=",")
    test = np.loadtxt(io.StringIO(_fetch(f"test/{machine_id}.txt")), delimiter=",")
    labels = np.loadtxt(io.StringIO(_fetch(f"test_label/{machine_id}.txt")), delimiter=",").astype(int)

    if train.shape[1] != FEAT or test.shape[1] != FEAT:
        raise ValueError(f"{machine_id}: expected {FEAT} KPI dims, got train={train.shape} test={test.shape}")
    if len(test) != len(labels):
        raise ValueError(f"{machine_id}: test length {len(test)} != labels length {len(labels)}")
    if np.isnan(train).any() or np.isnan(test).any():
        raise ValueError(f"{machine_id}: NaN values present in train/test — investigate before modeling")
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError(f"{machine_id}: Inf values present in train/test — investigate before modeling")

    segments = []
    for line in _fetch(f"interpretation_label/{machine_id}.txt").strip().splitlines():
        rng, dims = line.split(":")
        s, e = map(int, rng.split("-"))
        if not (0 <= s <= e < len(test)):
            raise ValueError(f"{machine_id}: interpretation_label segment {s}-{e} out of range for test length {len(test)}")
        segments.append((s, e, [int(d) for d in dims.split(",")]))

    return SMDMachine(machine_id=machine_id, train=train, test=test, labels=labels, segments=segments)


def make_windows(arr: np.ndarray, horizon: int | None = None, label_arr: np.ndarray | None = None,
                  window: int = WINDOW, feat: int = FEAT):
    """Build early-warning windows: tabular (mean/std/last) + raw (window x feat) + target + abs index.

    y[t] = 1 iff a labeled anomaly occurs anywhere in (t, t+horizon], using only data up to and
    including t (see notebook §0.3 for the formal task definition this implements).
    """
    n = len(arr)
    last_t = n - (horizon if horizon else 0) - 1
    idxs = np.arange(window - 1, last_t + 1)
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


@dataclass
class SplitData:
    scaler: StandardScaler
    sup_tab: np.ndarray; sup_raw: np.ndarray; sup_y: np.ndarray; sup_t: np.ndarray
    ev_tab: np.ndarray; ev_raw: np.ndarray; ev_y: np.ndarray; ev_t: np.ndarray
    cal_tab: np.ndarray; cal_raw: np.ndarray; cal_y: np.ndarray; cal_t: np.ndarray
    train_tab_full: np.ndarray; train_raw_full: np.ndarray; train_y_full: np.ndarray
    block_index: dict  # absolute test-timeline index -> (block name, row in that block's *_tab array)
    test_s: np.ndarray  # full scaled test array (for causal-analysis windows, etc.)


def build_split(m: SMDMachine, background_n: int = 6000, seed: int = 0) -> SplitData:
    """TRAIN(sup)+background / EVAL / CALIB leakage-safe chronological split (notebook §0.6)."""
    scaler = StandardScaler().fit(m.train)
    train_s, test_s = scaler.transform(m.train), scaler.transform(m.test)

    Xtr_tab, Xtr_raw, ytr, tr_idx = make_windows(train_s)

    sup_tab, sup_raw, sup_y, sup_t = make_windows(test_s[:CUT1], HORIZON, m.labels[:CUT1])
    ev_tab, ev_raw, ev_y, ev_t_local = make_windows(test_s[CUT1:CUT2], HORIZON, m.labels[CUT1:CUT2])
    cal_tab, cal_raw, cal_y, cal_t_local = make_windows(test_s[CUT2:], HORIZON, m.labels[CUT2:])
    ev_t, cal_t = ev_t_local + CUT1, cal_t_local + CUT2

    rng = np.random.default_rng(seed)
    bg_sel = rng.choice(len(tr_idx), size=min(background_n, len(tr_idx)), replace=False)
    train_tab_full = np.concatenate([sup_tab, Xtr_tab[bg_sel]], axis=0)
    train_raw_full = np.concatenate([sup_raw, Xtr_raw[bg_sel]], axis=0)
    train_y_full = np.concatenate([sup_y, ytr[bg_sel]], axis=0)

    block_index = {}
    for name, t_arr in [("train", sup_t), ("eval", ev_t), ("calib", cal_t)]:
        for row, t in enumerate(t_arr):
            block_index[int(t)] = (name, row)

    return SplitData(
        scaler=scaler,
        sup_tab=sup_tab, sup_raw=sup_raw, sup_y=sup_y, sup_t=sup_t,
        ev_tab=ev_tab, ev_raw=ev_raw, ev_y=ev_y, ev_t=ev_t,
        cal_tab=cal_tab, cal_raw=cal_raw, cal_y=cal_y, cal_t=cal_t,
        train_tab_full=train_tab_full, train_raw_full=train_raw_full, train_y_full=train_y_full,
        block_index=block_index, test_s=test_s,
    )
