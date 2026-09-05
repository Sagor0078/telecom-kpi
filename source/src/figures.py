"""
Paper figures for the three-dataset evaluation (notebook §14).

Generates publication-ready PNGs into `paper/figures/` from the saved result artifacts, so figures
can be regenerated without re-running the benchmarks. Inputs are the CSVs in `utils/` plus the
prediction pickles produced by the benchmark runs.

Note on the soft-vote ensemble: it averages percentile *ranks*, not probabilities, so its scores are
uniform by construction and ECE against them is meaningless. It is excluded from the calibration
figure (fig4) and labelled as such, rather than plotted with a misleading value.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

OUT = Path("paper/figures")
DPI = 300

FAMILY_COLORS = {
    "Naive": "#9e9e9e", "Linear": "#8c6d31", "Bagged tree": "#2e7d32",
    "Boosted tree": "#1b5e20", "Feedforward neural net": "#ef6c00",
    "Convolutional neural net": "#c62828", "Recurrent neural net": "#1565c0",
    "Attention": "#6a1b9a", "Deep ensemble": "#00838f", "Ensemble": "#455a64",
}

# Window counts and positive rates per block, from the executed runs (§14.1).
SPLITS = {
    "TelecomTS\n(early warning, 5 s)": {"TRAIN": (48749, 0.0857), "EVAL": (3840, 0.1674),
                                        "CALIB": (10487, 0.0403)},
    "RCAEval RE1/OB\n(detection)": {"TRAIN": (29893, 0.5134), "EVAL": (9980, 0.4985),
                                    "CALIB": (10285, 0.5134)},
    "SMD machine-1-1\n(early warning, 30 min)": {"TRAIN": (24751, 0.066), "EVAL": (4151, 0.286),
                                                 "CALIB": (5430, 0.017)},
}


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def fig1_framework():
    """The three-leg evaluation architecture (§14)."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    ax.add_patch(plt.Rectangle((3.6, 5.1), 2.8, 0.65, fc="#eceff1", ec="#37474f", lw=1.4))
    ax.text(5, 5.42, "Proposed framework", ha="center", va="center", fontsize=11, weight="bold")

    legs = [
        (1.7, "TelecomTS", "5G RAN domain\nvalidity",
         "AD + telecom RCA\n+ early warning\n(seconds-scale)", "anchors 1, 2, 4, 5", "#c62828"),
        (5.0, "RCAEval", "RCA / service-mgmt\nrigor",
         "root-cause quality\n+ baseline comparison\n(detection only)", "anchors 1, 3", "#1565c0"),
        (8.3, "SMD", "out-of-domain\ngeneralization",
         "robustness across\nserver telemetry", "RQ4 evidence", "#2e7d32"),
    ]
    ax.plot([5, 5], [5.1, 4.75], color="#37474f", lw=1.4)
    ax.plot([1.7, 8.3], [4.75, 4.75], color="#37474f", lw=1.4)

    for x, name, role, output, anchor, color in legs:
        ax.annotate("", xy=(x, 4.15), xytext=(x, 4.75),
                    arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.4))
        ax.add_patch(plt.Rectangle((x - 1.35, 3.25), 2.7, 0.9, fc=color, ec="none", alpha=0.13))
        ax.text(x, 3.90, name, ha="center", fontsize=11, weight="bold", color=color)
        ax.text(x, 3.52, role, ha="center", fontsize=8.5, color="#37474f")

        ax.annotate("", xy=(x, 2.55), xytext=(x, 3.25),
                    arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.2))
        ax.add_patch(plt.Rectangle((x - 1.35, 1.6), 2.7, 0.95, fc="white", ec=color, lw=1.2))
        ax.text(x, 2.07, output, ha="center", va="center", fontsize=8.5, color="#263238")

        ax.annotate("", xy=(x, 0.95), xytext=(x, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.2))
        ax.text(x, 0.66, anchor, ha="center", fontsize=8.5, style="italic", color=color)

    ax.text(5, 0.06, "TNSM scope anchors: 1 Management Functions · 2 Reliability & QA · "
                     "3 Enabling Technologies · 4 Emerging Tech · 5 Applications",
            ha="center", fontsize=7.5, color="#607d8b")
    _save(fig, "fig1_framework.png")


def fig2_data_splits():
    """TRAIN / EVAL / CALIB block sizes and positive rates for all three datasets (§14.1)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {"TRAIN": "#1565c0", "EVAL": "#c62828", "CALIB": "#00838f"}

    for ax, (ds, blocks) in zip(axes, SPLITS.items()):
        names = list(blocks)
        counts = [blocks[b][0] for b in names]
        rates = [blocks[b][1] for b in names]
        bars = ax.bar(names, counts, color=[colors[b] for b in names], alpha=0.85, width=0.6)
        for bar, c, r in zip(bars, counts, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n{r:.1%} pos",
                    ha="center", va="bottom", fontsize=8.5)
        ax.set_title(ds, fontsize=10)
        ax.set_ylabel("windows")
        ax.set_ylim(0, max(counts) * 1.32)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Chronological leakage-safe splits — no VALIDATION block; CALIB is held out "
                 "for conformal calibration only (§0.6)", fontsize=9.5, y=1.04)
    fig.tight_layout()
    _save(fig, "fig2_data_splits.png")


def _load_tables():
    t = pd.read_csv("utils/telecomts_model_comparison.csv")
    r = pd.read_csv("utils/rcaeval_model_comparison.csv")
    return t, r


def fig3_model_comparison():
    """AUPRC across the full hierarchy, both datasets, coloured by family (§14.2)."""
    t, r = _load_tables()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, df, title, base in [
        (axes[0], t, "TelecomTS — early warning, 5 s", 0.167),
        (axes[1], r, "RCAEval RE1/OB — detection", 0.498),
    ]:
        d = df.sort_values("auprc")
        ax.barh(d.model, d.auprc, color=[FAMILY_COLORS.get(f, "#999") for f in d.family],
                alpha=0.9, height=0.7)
        ax.axvline(base, color="#c62828", ls="--", lw=1.2)
        ax.text(base, -0.9, f" base rate {base:.3f}", color="#c62828", fontsize=8, va="top")
        for i, v in enumerate(d.auprc):
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_xlabel("AUPRC (held-out EVAL)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, max(d.auprc) * 1.13)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.9) for c in FAMILY_COLORS.values()]
    fig.legend(handles, FAMILY_COLORS, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    _save(fig, "fig3_model_comparison.png")


def fig4_accuracy_vs_calibration():
    """The headline tension: tree accuracy vs. deep-ensemble calibration (§14.3)."""
    t, r = _load_tables()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    for ax, df, title in [(axes[0], t, "TelecomTS — early warning, 5 s"),
                          (axes[1], r, "RCAEval RE1/OB — detection")]:
        d = df[(df.model != "Soft-vote ensemble") & (df.model != "Dummy prior")]
        for _, row in d.iterrows():
            ax.scatter(row.ece, row.auprc, s=90, alpha=0.85,
                       color=FAMILY_COLORS.get(row.family, "#999"),
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(row.model, (row.ece, row.auprc), fontsize=6.8,
                        xytext=(4, 3), textcoords="offset points", color="#37474f")
        ax.set_xlabel("ECE  (lower = better calibrated) →")
        ax.set_ylabel("AUPRC  (higher = more accurate) →")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Accuracy and calibration pull apart: trees win AUPRC, deep ensembles win ECE.\n"
                 "Split-conformal (§7) resolves this by wrapping the accurate model in a "
                 "distribution-free guarantee.", fontsize=10, y=1.08)
    fig.tight_layout()
    _save(fig, "fig4_accuracy_vs_calibration.png")


def fig5_preonset_runway():
    """Pre-onset runway per TelecomTS anomaly segment — which horizons are evaluable (§14.4)."""
    rw = pd.read_csv("utils/telecomts_preonset_runway.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].hist(rw.runway_sec, bins=24, color="#1565c0", alpha=0.85, edgecolor="white")
    for h, c in [(1, "#2e7d32"), (5, "#ef6c00"), (10, "#c62828"), (30, "#6a1b9a")]:
        axes[0].axvline(h, color=c, ls="--", lw=1.4)
        axes[0].text(h, axes[0].get_ylim()[1] * 0.94, f" {h}s", color=c, fontsize=8.5)
    axes[0].set_xlabel("pre-onset runway (seconds)")
    axes[0].set_ylabel("anomaly segments")
    axes[0].set_title(f"TelecomTS: runway before onset (n={len(rw)} segments)", fontsize=10)
    axes[0].spines[["top", "right"]].set_visible(False)

    horizons = [1, 2, 5, 10, 20, 30]
    feasible = [(rw.runway_sec >= h).sum() for h in horizons]
    bars = axes[1].bar([str(h) for h in horizons], feasible,
                       color=["#2e7d32" if f > 0 else "#c62828" for f in feasible], alpha=0.85)
    for bar, f in zip(bars, feasible):
        axes[1].text(bar.get_x() + bar.get_width() / 2, f, f"{f}/{len(rw)}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_xlabel("early-warning horizon (seconds)")
    axes[1].set_ylabel("segments with sufficient runway")
    axes[1].set_title("Horizon feasibility: 30 s is unsupportable", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig5_preonset_runway.png")


def fig6_roc_pr(scratch: str):
    """ROC and precision-recall curves for the top models on each dataset (§14.2)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for col, (pkl, title, base) in enumerate([
        (f"{scratch}/tts_preds.pkl", "TelecomTS — early warning, 5 s", 0.167),
        (f"{scratch}/rcae_preds.pkl", "RCAEval RE1/OB — detection", 0.498),
    ]):
        d = pickle.load(open(pkl, "rb"))
        y, preds = d["ev_y"], d["preds"]
        show = ["XGBoost", "Random Forest", "Deep ensemble (1D ResNet)",
                "Deep ensemble (Transformer)", "MLP", "Logistic Regression"]
        show = [m for m in show if m in preds]

        for m in show:
            fpr, tpr, _ = roc_curve(y, preds[m])
            axes[0, col].plot(fpr, tpr, lw=1.5, label=m)
            pr, rc, _ = precision_recall_curve(y, preds[m])
            axes[1, col].plot(rc, pr, lw=1.5, label=m)

        axes[0, col].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        axes[0, col].set(xlabel="false positive rate", ylabel="true positive rate",
                         title=f"ROC — {title}")
        axes[1, col].axhline(base, color="#c62828", ls="--", lw=1.0)
        axes[1, col].set(xlabel="recall", ylabel="precision", title=f"PR — {title}")
        for ax in (axes[0, col], axes[1, col]):
            ax.legend(fontsize=7.5, loc="lower left" if ax is axes[0, col] else "upper right")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig6_roc_pr_curves.png")


def fig7_deep_ensemble():
    """Does deep ensembling help? Single model vs. M=5, per architecture (§14.3)."""
    t, r = _load_tables()
    pairs = [("MLP", "Deep ensemble (MLP)"), ("1D ResNet", "Deep ensemble (1D ResNet)"),
             ("LSTM", "Deep ensemble (LSTM)"), ("GRU", "Deep ensemble (GRU)"),
             ("Transformer", "Deep ensemble (Transformer)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for ax, df, metric, title, better in [
        (axes[0], t, "auprc", "TelecomTS — AUPRC: single vs. 5-member ensemble", "higher"),
        (axes[1], t, "ece", "TelecomTS — ECE: single vs. 5-member ensemble", "lower"),
    ]:
        g = df.set_index("model")[metric]
        x = np.arange(len(pairs))
        singles = [g[a] for a, _ in pairs]
        ens = [g[b] for _, b in pairs]
        ax.bar(x - 0.2, singles, 0.4, label="single model", color="#90a4ae")
        ax.bar(x + 0.2, ens, 0.4, label="deep ensemble (M=5)", color="#00838f")
        for i, (s, e) in enumerate(zip(singles, ens)):
            ax.text(i + 0.2, e, f"{e - s:+.3f}", ha="center", va="bottom", fontsize=8,
                    color="#2e7d32" if ((e > s) == (better == "higher")) else "#c62828")
        ax.set_xticks(x)
        ax.set_xticklabels([a for a, _ in pairs], fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig7_deep_ensemble.png")


def generate_all(scratch: str):
    print("generating paper figures ->", OUT)
    fig1_framework()
    fig2_data_splits()
    fig3_model_comparison()
    fig4_accuracy_vs_calibration()
    fig5_preonset_runway()
    fig6_roc_pr(scratch)
    fig7_deep_ensemble()
    fig8_reliability(scratch)
    fig9_horizon_sweep()
    fig10_per_anomaly_type()
    fig11_case_study()
    fig12_efficiency()
    print("done")




# --------------------------------------------------------------------------------------------
# Figures 8-12: added for the anchors that had no figure (calibration, RQ4 robustness,
# Applications & Case Studies), plus the contamination-corrected early-warning curve.
# --------------------------------------------------------------------------------------------

def fig8_reliability(scratch: str):
    """Reliability diagrams — what ECE summarises, for the §7 confidence-aware anchor."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    show = ["XGBoost", "Deep ensemble (Transformer)", "Deep ensemble (1D ResNet)",
            "Logistic Regression", "LightGBM"]

    for ax, (pkl, title) in zip(axes, [
        (f"{scratch}/tts_preds.pkl", "TelecomTS — early warning, 5 s"),
        (f"{scratch}/rcae_preds.pkl", "RCAEval RE1/OB — detection"),
    ]):
        d = pickle.load(open(pkl, "rb"))
        y, preds = d["ev_y"], d["preds"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for m in [m for m in show if m in preds]:
            p = preds[m]
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, 9)
            xs, ys = [], []
            for b in range(10):
                sel = idx == b
                if sel.sum() >= 20:
                    xs.append(p[sel].mean()); ys.append(y[sel].mean())
            ax.plot(xs, ys, "o-", lw=1.5, ms=4, label=m)
        ax.set(xlabel="mean predicted probability", ylabel="observed frequency", title=title)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Reliability diagrams — points below the diagonal are over-confident", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig8_reliability_diagrams.png")


def fig9_horizon_sweep():
    """Early warning vs. horizon, with and without label contamination.

    The upper curve counts windows in which the anomaly is *already visible* as positives; that
    measures detection, not anticipation. The lower curve scores only genuinely pre-onset windows.
    """
    d = pd.read_csv("utils/telecomts_horizon_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    axes[0].plot(d.horizon_sec, d.auprc_all, "o-", color="#c62828", lw=2,
                 label="all positives (contaminated — includes ongoing anomalies)")
    axes[0].plot(d.horizon_sec, d.auprc_clean, "s-", color="#1565c0", lw=2,
                 label="pre-onset only (true early warning)")
    axes[0].plot(d.horizon_sec, d.base_all, "--", color="#c62828", alpha=0.5, lw=1,
                 label="base rate (contaminated)")
    axes[0].plot(d.horizon_sec, d.base_clean, "--", color="#1565c0", alpha=0.5, lw=1,
                 label="base rate (pre-onset)")
    axes[0].set(xlabel="early-warning horizon (s)", ylabel="AUPRC",
                title="TelecomTS — the contamination gap")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)

    axes[1].plot(d.horizon_sec, d.lift_all, "o-", color="#c62828", lw=2, label="contaminated")
    axes[1].plot(d.horizon_sec, d.lift_clean, "s-", color="#1565c0", lw=2, label="pre-onset only")
    axes[1].axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    axes[1].text(1.2, 1.05, "no better than chance", fontsize=8)
    axes[1].set(xlabel="early-warning horizon (s)", ylabel="AUPRC / base rate  (lift)",
                title="Lift survives contamination correction")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Correcting for label contamination cuts absolute AUPRC ~5× at short horizons, "
                 "but the 4-6× lift over base rate is genuine.", fontsize=9.5, y=1.04)
    fig.tight_layout()
    _save(fig, "fig9_horizon_sweep.png")


def fig10_per_anomaly_type():
    """Per-anomaly-type performance — RQ4 robustness across 5G fault modes."""
    d = pd.read_csv("utils/telecomts_per_anomaly_type.csv").sort_values("auprc")
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.barh(d.anomaly_type, d.auprc, color="#1565c0", alpha=0.85, height=0.7)
    for bar, v, n in zip(bars, d.auprc, d.n_windows):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, f"{v:.3f}  (n={n})",
                va="center", fontsize=8)
    ax.set(xlabel="AUPRC (XGBoost, 5 s horizon, vs. all negatives)",
           title="TelecomTS — detection difficulty varies 8× across 5G fault modes")
    ax.set_xlim(0, d.auprc.max() * 1.28)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig10_per_anomaly_type.png")


def fig11_case_study():
    """One 5G incident end to end — anchor 5 (Applications & Case Studies)."""
    z = np.load("utils/telecomts_case_study.npz", allow_pickle=True)
    vals, labels = z["values"], z["labels"]
    chans = [str(c) for c in z["channels"]]
    affected = [chans[i] for i in z["affected"]]
    s0, s1 = int(z["seg_start"]), int(z["seg_end"])
    t = np.arange(len(vals)) / 10.0

    # Pick a mix so the affected/unaffected contrast is visible: 3 affected + 2 unaffected.
    unaffected = [c for c in chans if c not in affected]
    show = [c for c in ["RSRP", "UL_SNR", "DL_BLER"] if c in affected][:3] + unaffected[:2]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 1.55 * len(show)), sharex=True)
    for ax, ch in zip(axes, show):
        v = vals[:, chans.index(ch)]
        color = "#c62828" if ch in affected else "#546e7a"
        ax.plot(t, v, lw=0.9, color=color)
        ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color="#c62828", alpha=0.12)
        ax.axvline(t[s0], color="#c62828", ls="--", lw=1.2)
        tag = "  ← in affected_kpis" if ch in affected else ""
        ax.set_ylabel(ch, fontsize=8.5, color=color)
        ax.text(0.995, 0.86, tag, transform=ax.transAxes, ha="right", fontsize=7.5,
                color="#c62828")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time within session (s)")
    axes[0].set_title(f"5G incident case study — {z['atype']}\n"
                      f"shaded = labelled anomaly; red traces are the ground-truth affected KPIs "
                      f"({len(affected)} of {len(chans)})", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig11_case_study_timeline.png")


def fig12_efficiency():
    """Accuracy against training cost — RQ5 operationality."""
    t, r = _load_tables()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, df, title in [(axes[0], t, "TelecomTS — early warning, 5 s"),
                          (axes[1], r, "RCAEval RE1/OB — detection")]:
        d = df[df.model != "Dummy prior"]
        ax.scatter(d.fit_seconds.clip(lower=0.3), d.auprc, s=90, alpha=0.85,
                   color=[FAMILY_COLORS.get(f, "#999") for f in d.family],
                   edgecolor="white", linewidth=1.2, zorder=3)
        for _, row in d.iterrows():
            ax.annotate(row.model, (max(row.fit_seconds, 0.3), row.auprc), fontsize=6.8,
                        xytext=(4, 3), textcoords="offset points", color="#37474f")
        ax.set_xscale("log")
        ax.set(xlabel="training time (s, log scale) →", ylabel="AUPRC", title=title)
        ax.grid(alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RQ5 — the accurate models are also the cheap ones; attention and deep "
                 "ensembles cost 40-1000× for no gain", fontsize=10, y=1.03)
    fig.tight_layout()
    _save(fig, "fig12_efficiency.png")


if __name__ == "__main__":
    import sys
    generate_all(sys.argv[1] if len(sys.argv) > 1 else ".")
