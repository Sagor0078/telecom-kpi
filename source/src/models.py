"""
Dataset-agnostic early-warning model hierarchy: naive, linear, bagged-tree, boosted-tree,
recurrent, attention, **deep ensembles**, and a cross-family soft-vote ensemble.

Deep ensembles (Lakshminarayanan et al., 2017 — M independently-seeded copies of one architecture,
averaged) are included for a reason beyond accuracy: they are the standard deep-learning approach
to uncertainty, so they are the natural comparator for §7's split-conformal layer under the
Enabling Technologies anchor (§0.18.1). Both are reported here with ECE alongside AUROC/AUPRC, so
"is the model right?" and "does it know when it isn't?" can be read from one table. Member spread
(`ensemble_std`) is retained as the ensemble's own uncertainty signal and as an input to the
disagreement-based trust score proposed in §0.13.

Hyperparameters are copied verbatim from the main notebook's §6 so that TelecomTS, RCAEval and SMD
results are directly comparable. This matters for the dataset strategy in §0.18.2: *"the method
must be identical across datasets wherever possible; changing the core algorithm per benchmark
destroys the generalization argument."* Dataset-specific preprocessing lives in the per-dataset
loaders (`src.data`, `src.telecomts`, `src.rcaeval`), never here.

The comparison is baseline work, not the contribution — §0.18.1 warns that "we compared nine models
and one won" reads as filler under the Enabling Technologies anchor. Its purpose is RQ1
(detection) and to fix a backbone for the RCA and conformal layers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

FAMILIES = {
    "Dummy prior": "Naive",
    "Logistic Regression": "Linear",
    "Random Forest": "Bagged tree",
    "Extra Trees": "Bagged tree",
    "XGBoost": "Boosted tree",
    "LightGBM": "Boosted tree",
    "LSTM": "Recurrent neural net",
    "GRU": "Recurrent neural net",
    "Transformer": "Attention",
    "MLP": "Feedforward neural net",
    "1D ResNet": "Convolutional neural net",
    "Deep ensemble (MLP)": "Deep ensemble",
    "Deep ensemble (1D ResNet)": "Deep ensemble",
    "Deep ensemble (LSTM)": "Deep ensemble",
    "Deep ensemble (GRU)": "Deep ensemble",
    "Deep ensemble (Transformer)": "Deep ensemble",
    "Deep ensemble (heterogeneous)": "Deep ensemble",
    "Soft-vote ensemble": "Ensemble",
}


@dataclass
class ComparisonResult:
    table: pd.DataFrame
    predictions: dict
    fit_seconds: dict
    # Per-member spread of each deep ensemble, aligned to the eval rows. This is the ensemble's own
    # uncertainty signal, and the input the §0.13 disagreement-based trust score needs.
    ensemble_std: dict = None


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width-bin ECE — the calibration measure §7 reports, computed for every model here so
    accuracy and calibration can be read off one table rather than two experiments."""
    y_true = np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def _tabular_models(spw: float, seed: int = 0) -> dict:
    import lightgbm as lgb
    from xgboost import XGBClassifier

    return {
        "Dummy prior": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=spw,
            verbose=-1, random_state=seed, n_jobs=-1),
    }


def _sequence_modules():
    import torch
    import torch.nn as nn

    class RecurrentClassifier(nn.Module):
        def __init__(self, feat, kind="LSTM", hidden=32):
            super().__init__()
            rnn_cls = nn.LSTM if kind == "LSTM" else nn.GRU
            self.rnn = rnn_cls(feat, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            _, state = self.rnn(x)
            h = state[0] if isinstance(state, tuple) else state
            return self.head(h[-1]).squeeze(-1)

    class TinyTransformer(nn.Module):
        def __init__(self, feat, window, d_model=32, nhead=4, layers=2):
            super().__init__()
            self.proj = nn.Linear(feat, d_model)
            self.pos = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64,
                                             batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            h = self.encoder(self.proj(x) + self.pos)
            return self.head(h.mean(dim=1)).squeeze(-1)

    class MLP(nn.Module):
        """Feedforward net over the flattened window — the plain-DL baseline the hierarchy lacked."""

        def __init__(self, feat, window, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(feat * window, hidden), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class ResNet1D(nn.Module):
        """Three residual Conv1d blocks — the 1D analogue of ResNet, and a standard strong
        time-series-classification baseline. The 2D ImageNet variants (ResNet-18/34, VGG-16,
        MobileNet-v3) do not apply: their pooling stacks collapse a 30-step window to nothing."""

        def __init__(self, feat, window, ch=32):
            super().__init__()
            self.inp = nn.Conv1d(feat, ch, 3, padding=1)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
                              nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch))
                for _ in range(3)])
            self.head = nn.Linear(ch, 1)

        def forward(self, x):
            h = self.inp(x.transpose(1, 2))          # (B, window, feat) -> (B, feat, window)
            for blk in self.blocks:
                h = torch.relu(h + blk(h))
            return self.head(h.mean(-1)).squeeze(-1)

    return torch, nn, RecurrentClassifier, TinyTransformer, MLP, ResNet1D


def _operating_point(y_true: np.ndarray, p: np.ndarray) -> dict:
    """F1-optimal threshold metrics. A degenerate scorer (single unique value) gets the trivial
    all-positive point rather than an exception, so the naive baseline still appears in the table."""
    uniq = np.unique(p)
    if len(uniq) < 2:
        thr = uniq[0] if len(uniq) else 0.5
        pred = np.ones_like(y_true)
        return {"threshold": float(thr), "f1": f1_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0)}

    grid = np.quantile(uniq, np.linspace(0.01, 0.99, 99))
    best = max(((f1_score(y_true, (p >= t).astype(int), zero_division=0), t) for t in grid),
               key=lambda x: x[0])
    pred = (p >= best[1]).astype(int)
    return {"threshold": float(best[1]), "f1": best[0],
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0)}


def run_comparison(train_tab, train_raw, train_y, eval_tab, eval_raw, eval_y,
                   epochs: int = 6, batch: int = 256, seed: int = 0,
                   include_sequence: bool = True, verbose: bool = True,
                   deep_ensemble_size: int = 5) -> ComparisonResult:
    """Fit the full hierarchy on one dataset's split and score it on the held-out block.

    `*_tab` are the rolling mean/std/last features; `*_raw` are the (n, window, feat) sequences.
    Returns per-model AUROC/AUPRC, F1-optimal operating point, fit/predict cost and parameter count.
    """
    train_y = np.asarray(train_y).astype(int)
    eval_y = np.asarray(eval_y).astype(int)
    pos = int(train_y.sum())
    neg = len(train_y) - pos
    if pos == 0 or neg == 0:
        raise ValueError(f"training block is single-class (pos={pos}, neg={neg}); "
                         "widen the horizon or pick a split containing an event")
    spw = neg / max(pos, 1)

    preds, fit_seconds, params, pred_seconds, ens_std = {}, {}, {}, {}, {}

    for name, est in _tabular_models(spw, seed).items():
        t0 = time.perf_counter()
        est.fit(train_tab, train_y)
        fit_seconds[name] = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds[name] = est.predict_proba(eval_tab)[:, 1]
        pred_seconds[name] = time.perf_counter() - t1
        params[name] = np.nan
        if verbose:
            print(f"  {name:20s} AUROC={roc_auc_score(eval_y, preds[name]):.3f} "
                  f"AUPRC={average_precision_score(eval_y, preds[name]):.3f} "
                  f"fit={fit_seconds[name]:.1f}s")

    if include_sequence:
        torch, nn, Recurrent, Transformer, MLP, ResNet1D = _sequence_modules()
        feat, window = train_raw.shape[2], train_raw.shape[1]
        Xtr = torch.tensor(train_raw, dtype=torch.float32)
        ytr = torch.tensor(train_y, dtype=torch.float32)
        Xev = torch.tensor(eval_raw, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([spw], dtype=torch.float32))

        def train_member(make_model, member_seed):
            """Fit one network from a given seed; returns its eval probabilities and costs."""
            torch.manual_seed(member_seed)
            model = make_model()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(member_seed)
            t0 = time.perf_counter()
            for _ in range(epochs):
                model.train()
                perm = torch.randperm(len(Xtr), generator=gen)
                for i in range(0, len(Xtr), batch):
                    idx = perm[i:i + batch]
                    opt.zero_grad()
                    lossf(model(Xtr[idx]), ytr[idx]).backward()
                    opt.step()
            fit_s = time.perf_counter() - t0
            model.eval()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = np.concatenate([torch.sigmoid(model(Xev[i:i + 1024])).cpu().numpy()
                                      for i in range(0, len(Xev), 1024)])
            return out, fit_s, time.perf_counter() - t1, sum(p.numel() for p in model.parameters())

        def report(name):
            if verbose:
                print(f"  {name:28s} AUROC={roc_auc_score(eval_y, preds[name]):.3f} "
                      f"AUPRC={average_precision_score(eval_y, preds[name]):.3f} "
                      f"ECE={expected_calibration_error(eval_y, preds[name]):.3f} "
                      f"fit={fit_seconds[name]:.1f}s")

        # A deep ensemble (Lakshminarayanan et al., 2017) is M independently-seeded copies of one
        # architecture, averaged. Member 0 doubles as the single-model result, so M members cost M
        # trainings rather than M+1. Its member spread is a distribution-free uncertainty signal,
        # and the natural comparator for the conformal layer of §7.
        M = max(1, int(deep_ensemble_size))
        architectures = {
            "LSTM": lambda: Recurrent(feat, kind="LSTM"),
            "GRU": lambda: Recurrent(feat, kind="GRU"),
            "Transformer": lambda: Transformer(feat, window),
            "MLP": lambda: MLP(feat, window),
            "1D ResNet": lambda: ResNet1D(feat, window),
        }

        for arch, make in architectures.items():
            members, fits, pres, npar = [], [], [], 0
            for m in range(M):
                out, fs, ps, npar = train_member(make, seed + 1000 * m)
                members.append(out)
                fits.append(fs)
                pres.append(ps)
                if m == 0:                                  # single-model baseline
                    preds[arch], fit_seconds[arch] = out, fs
                    pred_seconds[arch], params[arch] = ps, npar
                    report(arch)

            if M > 1:
                name = f"Deep ensemble ({arch})"
                stack = np.vstack(members)
                preds[name] = stack.mean(axis=0)
                ens_std[name] = stack.std(axis=0)
                fit_seconds[name] = float(np.sum(fits))
                pred_seconds[name] = float(np.sum(pres))
                params[name] = npar * M
                report(name)

        # Heterogeneous deep ensemble: average across architectures rather than across seeds, so
        # its diversity comes from inductive bias instead of initialisation.
        het = [f"Deep ensemble ({a})" if M > 1 else a for a in architectures]
        if all(h in preds for h in het):
            name = "Deep ensemble (heterogeneous)"
            stack = np.vstack([preds[h] for h in het])
            preds[name] = stack.mean(axis=0)
            ens_std[name] = stack.std(axis=0)
            fit_seconds[name] = float(sum(fit_seconds[h] for h in het))
            pred_seconds[name] = float(sum(pred_seconds[h] for h in het))
            params[name] = float(sum(params[h] for h in het))
            report(name)

    # Cross-family soft vote over rank-normalised scores, so models on different probability
    # scales contribute comparably. Excludes the degenerate naive prior.
    members = [n for n in preds if n != "Dummy prior"]
    if len(members) > 1:
        ranks = [pd.Series(preds[n]).rank(pct=True).to_numpy() for n in members]
        preds["Soft-vote ensemble"] = np.mean(ranks, axis=0)
        fit_seconds["Soft-vote ensemble"] = sum(fit_seconds[n] for n in members)
        pred_seconds["Soft-vote ensemble"] = sum(pred_seconds[n] for n in members)
        params["Soft-vote ensemble"] = np.nan

    rows = []
    for name, p in preds.items():
        op = _operating_point(eval_y, p)
        rows.append({
            "model": name, "family": FAMILIES.get(name, "?"),
            "auroc": roc_auc_score(eval_y, p) if len(np.unique(eval_y)) > 1 else np.nan,
            "auprc": average_precision_score(eval_y, p),
            "ece": expected_calibration_error(eval_y, p),
            **op,
            "fit_seconds": fit_seconds[name], "predict_seconds": pred_seconds[name],
            "n_params": params[name],
            "mean_member_std": float(np.mean(ens_std[name])) if name in ens_std else np.nan,
        })

    table = pd.DataFrame(rows).sort_values("auprc", ascending=False).reset_index(drop=True)
    return ComparisonResult(table=table, predictions=preds, fit_seconds=fit_seconds,
                            ensemble_std=ens_std)
