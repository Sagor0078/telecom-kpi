"""
RCAEval loading, windowing, and the leakage-safe pre/post-injection split.

RCAEval (Pham et al., arXiv:2412.17015; HF dataset `phamquiluan/RCAEval`) is the
**service-management / RCA rigor benchmark** in this project's dataset strategy — see the main
notebook's §0.18.2. It contributes what neither TelecomTS nor SMD can: 735 failure cases across 11
fault types with an annotated root-cause *service* and *indicator*, produced by deliberate fault
injection with a recorded timestamp. Those injections are real interventions, so the
correlation-vs-causation component (§9) can be scored against known interventions rather than
against Granger temporal precedence alone.

Each case carries 360-2100 pre-injection timesteps at 1 Hz (6-35 min). That runway makes the data
look suited to minute-scale *early warning*, but it is not: the faults are externally scheduled
injections, so nothing in the telemetry precedes them and there is no precursor to learn. Measured
directly — a fault-stratified detection setup separates cleanly, while an early-warning setup at
H=300 s collapses to chance (AUROC 0.44-0.53 across all nine models). **Use this benchmark for
detection and RCA, not anticipation**; early warning is TelecomTS's job, at seconds scale (see
`src.telecomts.preonset_runway` and notebook §0.18.7).

Interface mirrors `src.data` (SMD) and `src.telecomts` so identical modelling code runs on all
three datasets, per §0.18.2.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HF_BASE = "https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/main"

# 1 Hz telemetry. SMD's H=30 is 30 minutes at 1 sample/min; the RCAEval equivalent in *samples*
# is 30*60. Defaults here are stated in seconds and converted, so horizons are comparable across
# datasets in wall-clock terms rather than in sample counts.
SAMPLE_HZ = 1
WINDOW_SEC = 60
HORIZON_SEC = 300


@dataclass
class RCAEvalCase:
    """One injected-failure case: telemetry plus the annotated root cause and injection time."""

    case: str
    suite: str
    system: str
    system_name: str
    root_cause_service: str
    fault: str
    fault_description: str
    inject_time: int
    metrics: pd.DataFrame          # 'time' + one column per (service, metric)
    values: np.ndarray             # (N, F) float, metric columns only
    columns: list                  # metric column names, aligned to `values`
    labels: np.ndarray             # (N,) int 0/1, 1 from inject_time onward

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def minutes(self) -> float:
        return self.n / SAMPLE_HZ / 60

    @property
    def preonset_samples(self) -> int:
        """Normal samples before injection — upper bound on an honest early-warning horizon."""
        return int((self.labels == 0).sum())

    def select(self, columns: list[str]) -> np.ndarray:
        """Cleaned float array for a column subset — the safe way to read this case's telemetry.

        `values` is non-finite-cleaned at load time but `metrics` is the raw frame, so reading
        columns straight off `metrics` reintroduces the NaNs (and silently poisons a StandardScaler
        fitted on them). Always go through here when selecting a subset such as the cross-case
        common column intersection.
        """
        arr = self.metrics[columns].to_numpy(dtype=np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def root_cause_columns(self) -> list[int]:
        """Indices of metric columns belonging to the annotated root-cause service.

        This is RCAEval's analogue of SMD's `interpretation_label` and TelecomTS's
        `affected_kpis`: the ground-truth target set a localization ranking is scored against.
        """
        prefix = f"{self.root_cause_service}_"
        return [i for i, c in enumerate(self.columns) if c.startswith(prefix)]


def _fetch(path: str, cache_dir: str | Path) -> Path:
    """Download `path` from the dataset repo into `cache_dir`, once."""
    cache_dir = Path(cache_dir)
    dest = cache_dir / path.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(f"{HF_BASE}/{path}", tmp)
        tmp.rename(dest)
    return dest


def load_case_index(cache_dir: str | Path = ".cache/rcaeval") -> pd.DataFrame:
    """The 735-row case index: suite, system, fault type, root-cause service, injection time."""
    return pd.read_parquet(_fetch("cases.parquet", cache_dir))


def load_case(case: str | pd.Series, cache_dir: str | Path = ".cache/rcaeval",
              index: pd.DataFrame | None = None) -> RCAEvalCase:
    """Load one case's metrics and attach its annotated root cause and injection boundary.

    Raises on a missing injection timestamp or an all-normal / all-faulty label vector rather than
    returning a case no split can be built from, matching the fail-loud contract in `src.data`.
    """
    if isinstance(case, str):
        index = load_case_index(cache_dir) if index is None else index
        rows = index[index["case"] == case]
        if rows.empty:
            raise ValueError(f"unknown RCAEval case {case!r}")
        row = rows.iloc[0]
    else:
        row = case

    df = pd.read_parquet(_fetch(f"{row['case']}/metrics.parquet", cache_dir))
    if "time" not in df.columns:
        raise ValueError(f"{row['case']}: metrics.parquet has no 'time' column")
    df = df.sort_values("time").reset_index(drop=True)

    inject = int(row["inject_time"])
    labels = (df["time"].to_numpy() >= inject).astype(int)
    if labels.all() or not labels.any():
        raise ValueError(f"{row['case']}: injection at {inject} lies outside the metric window "
                         f"[{df['time'].iloc[0]}, {df['time'].iloc[-1]}]")

    cols = [c for c in df.columns if c != "time"]
    values = df[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    return RCAEvalCase(
        case=str(row["case"]), suite=str(row["suite"]), system=str(row["system"]),
        system_name=str(row["system_name"]), root_cause_service=str(row["root_cause_service"]),
        fault=str(row["fault"]), fault_description=str(row.get("fault_description", "")),
        inject_time=inject, metrics=df, values=values, columns=cols, labels=labels,
    )


def load_suite(suite: str = "RE1", limit: int | None = None,
               cache_dir: str | Path = ".cache/rcaeval") -> list[RCAEvalCase]:
    """Load every case in one benchmark suite (RE1 metric-only, RE2/RE3 multi-source)."""
    index = load_case_index(cache_dir)
    sel = index[index["suite"] == suite]
    if sel.empty:
        raise ValueError(f"no cases for suite {suite!r}; available: {sorted(index['suite'].unique())}")
    if limit is not None:
        sel = sel.head(limit)
    return [load_case(row, cache_dir, index) for _, row in sel.iterrows()]


def make_windows(arr: np.ndarray, horizon: int | None = None, label_arr: np.ndarray | None = None,
                 window: int = WINDOW_SEC * SAMPLE_HZ, feat: int | None = None, stride: int = 1):
    """Windows over one case, matching `src.data.make_windows` in shape and semantics.

    With `horizon`, y[t] = 1 iff a faulty timestamp falls in (t, t+horizon] — the early-warning
    target. **On RCAEval that target is ill-posed**: faults are externally scheduled injections, so
    no precursor exists in the telemetry before the injection instant, and a horizon approaching a
    case's pre-injection runway drives the positive rate to ~1 (notebook §0.18.7). Pass
    `horizon=None` and take `label_arr[idxs]` for the *detection* target this benchmark supports.

    `feat` is inferred from `arr` because RCAEval's metric count varies by system (49-376 columns).
    `stride` subsamples window origins to keep sequence models tractable on long cases.
    """
    feat = arr.shape[1] if feat is None else feat
    n = len(arr)
    last_t = n - (horizon if horizon else 0) - 1
    idxs = np.arange(window - 1, last_t + 1, stride)
    if len(idxs) == 0:
        return (np.zeros((0, feat * 3)), np.zeros((0, window, feat), np.float32),
                np.zeros(0, int), np.zeros(0, int))

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


def preonset_runway(cases: list[RCAEvalCase]) -> pd.DataFrame:
    """Per case, the normal runway before injection — the RCAEval counterpart of §0.18.7's
    TelecomTS measurement, and the reason minute-scale horizons are defensible here."""
    return pd.DataFrame([{
        "case": c.case, "suite": c.suite, "system": c.system_name, "fault": c.fault,
        "root_cause_service": c.root_cause_service,
        "n_metrics": len(c.columns), "n_samples": c.n,
        "runway_samples": c.preonset_samples,
        "runway_sec": c.preonset_samples / SAMPLE_HZ,
        "runway_min": c.preonset_samples / SAMPLE_HZ / 60,
    } for c in cases])
