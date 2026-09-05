"""Minimal experiment registry (governing prompt §37-§39): every run appends one row to a CSV
rather than results being manually typed into a table (§55 requirement)."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "no-git-repo"
    except Exception:
        return "no-git-repo"


_CORE_FIELDS = ["timestamp", "git_commit", "experiment", "model", "machine", "seed"]


def append_row(csv_path: str | Path, row: dict) -> None:
    """Append one experiment-run row to a shared CSV log.

    Different experiments (baselines, RCA localization, causal analysis, ...) record different
    metrics. A naive DictWriter re-created per call -- as an earlier version of this function did
    -- silently misaligns columns once two calls use different key sets, because each call would
    fix its *own* fieldnames from that row alone while appending into a file whose header was fixed
    by an earlier, different row. Fixed here: a small set of core columns is always present in a
    fixed order, and everything else (which varies by experiment) is packed into one JSON column so
    the CSV's shape never depends on which experiment logged most recently.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(), **row}
    core = {k: row.get(k, "") for k in _CORE_FIELDS}
    extra = {k: v for k, v in row.items() if k not in _CORE_FIELDS}
    out_row = {**core, "metrics_json": json.dumps(extra, default=str)}

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CORE_FIELDS + ["metrics_json"])
        if write_header:
            w.writeheader()
        w.writerow(out_row)


def environment_info() -> dict:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for pkg in ["numpy", "pandas", "sklearn", "torch", "lightgbm", "shap", "statsmodels", "networkx"]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[f"{pkg}_version"] = "not installed"
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        info["cuda_available"] = False
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_kb"] = int(line.split()[1])
                    break
    except Exception:
        pass
    return info


def save_environment_info(path: str | Path = "results/environment.json") -> dict:
    info = environment_info()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info
