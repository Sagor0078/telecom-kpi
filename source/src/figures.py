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

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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

# Window counts, positive rates and EVAL base rates come from the run that produced the tables,
# via utils/section14_summary.json (written by src/run_section14.py). They used to be transcribed
# into this file by hand, which is how the figures went on rendering a split whose own artifacts
# had gone missing: `_load_tables` raised on the absent CSVs, but every constant here still
# claimed the numbers were current. Read them, do not copy them.
SUMMARY_PATH = Path("utils/section14_summary.json")

# SMD is reported in §6 rather than by run_section14.py, so its split stays declared here.
SMD_SPLIT = {"TRAIN": (24751, 0.066), "EVAL": (4151, 0.286), "CALIB": (5430, 0.017)}
LEG_TITLES = {"telecomts": "TelecomTS\n(early warning, 5 s)",
              "rcaeval": "RCAEval RE1/OB\n(detection)"}


def _summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} is missing. Run `python -m src.run_section14` first; the figures are "
            "generated from that run's artifacts, not from constants stored in this file.")
    return json.loads(SUMMARY_PATH.read_text())


def _splits() -> dict:
    su = _summary()
    out = {}
    for leg, title in LEG_TITLES.items():
        w, pr = su[leg]["windows"], su[leg]["pos_rate"]
        out[title] = {b.upper(): (w[b], pr[b]) for b in ("train", "eval", "calib")}
    out["SMD machine-1-1\n(early warning, 30 min)"] = SMD_SPLIT
    return out


def _base_rate(leg: str) -> float:
    """EVAL positive rate for one leg — the reference line on the PR panels."""
    return _summary()[leg]["pos_rate"]["eval"]


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def fig1_framework():
    """The three-leg evaluation architecture (§14), as a full-width figure.

    Bullet content is deliberately narrower than each dataset's headline description, because the
    figure has to describe *this* evaluation rather than the datasets' capabilities: RE1 / Online
    Boutique ships no logs or traces (only RE2/RE3 do, and they are untouched), one SMD machine of
    28 is used, TelecomTS is testbed-derived rather than operator telemetry, and no external RCA
    baseline is compared anywhere. A figure claiming otherwise would contradict §VIII.
    """
    LEGS = [
        {"key": "tts", "name": "TelecomTS", "role": "5G RAN domain validity",
         "fill": "#fdecea", "edge": "#c62828",
         "facts": ["5G testbed telemetry (not operator)", "18 named PHY/MAC/network KPIs",
                   "109 segments, 11 anomaly types", "Reconstructed 10 Hz series"],
         "job": "Anomaly Detection\n+ Telecom RCA\n+ Early Warning",
         "does": ["Early warning at a 5 s horizon", "Localization vs. affected_kpis",
                  "Per-anomaly-type breakdown", "Pre-onset runway feasibility"],
         "anchor": "TNSM anchors: 1, 2, 4, 5"},
        {"key": "rce", "name": "RCAEval", "role": "RCA / service-management rigor",
         "fill": "#e8f0fe", "edge": "#1565c0",
         "facts": ["RE1 / Online Boutique: 123 cases", "24 common metric columns",
                   "5 fault types, root-cause service", "Injections are known interventions"],
         "job": "Root-Cause Quality\n+ Detection",
         "does": ["Localization vs. root-cause service", "Detection only (no anticipation)",
                  "Per-fault-type breakdown", "Causal filter on known onsets"],
         "anchor": "TNSM anchors: 1, 3"},
        {"key": "smd", "name": "SMD", "role": "Out-of-domain generalization",
         "fill": "#e8f5e9", "edge": "#2e7d32",
         "facts": ["machine-1-1 of 28, 38 dims", "Real-world server telemetry",
                   "8 segments, dimension-level truth", "Widely used AD benchmark"],
         "job": "Robustness Across\nServer Telemetry",
         "does": ["Out-of-domain generalization", "Conformal coverage + abstention",
                  "Explanation faithfulness audit", "Dimension-level localization"],
         "anchor": "RQ4 evidence"},
    ]
    INK = "#37474f"

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(-0.5, 100.5); ax.set_ylim(0, 100); ax.axis("off")

    def card(x0, x1, y0, y1, fill, edge, lw=1.3):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=fill, ec=edge, lw=lw, zorder=2))

    def bullets(items, xc, x_left, y_top, size, color=INK, step=4.0):
        for i, text in enumerate(items):
            y = y_top - i * step
            ax.text(x_left, y, "\u2022", ha="center", va="center", fontsize=size,
                    color=color, zorder=3)
            ax.text(x_left + 1.7, y, text, ha="left", va="center", fontsize=size,
                    color=color, zorder=3)

    def elbow(x_from, y_from, x_to, y_to, y_mid):
        """Right-angled connector: down, across, then into the target with an arrow head."""
        ax.plot([x_from, x_from], [y_from, y_mid], color=INK, lw=1.3, zorder=1)
        ax.plot([x_from, x_to], [y_mid, y_mid], color=INK, lw=1.3, zorder=1)
        ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, shrinkA=0, shrinkB=0))

    def down(x, y_from, y_to):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, shrinkA=0, shrinkB=0))

    # ---- top: the framework -------------------------------------------------
    card(17, 83, 90.0, 99.5, "#eceff1", INK, lw=1.5)
    ax.text(50, 96.4, "Proposed Framework", ha="center", va="center",
            fontsize=14, weight="bold", color="#263238", zorder=3)
    ax.text(50, 92.4, "(Anomaly Detection, Root-Cause Analysis and Early Warning "
                      "for Network Management)",
            ha="center", va="center", fontsize=8.2, color=INK, zorder=3)

    COL_W, GAP = 29.0, 4.0
    x0s = [2.5 + i * (COL_W + GAP) for i in range(3)]

    for leg, x0 in zip(LEGS, x0s):
        x1, xc = x0 + COL_W, x0 + COL_W / 2

        # ---- dataset card ---------------------------------------------------
        card(x0, x1, 62.5, 87.5, leg["fill"], leg["edge"], lw=1.2)
        ax.text(xc, 84.6, leg["name"], ha="center", va="center", fontsize=13.5,
                weight="bold", color=leg["edge"], zorder=3)
        ax.text(xc, 81.0, leg["role"], ha="center", va="center", fontsize=9.2,
                color=INK, zorder=3)
        ax.plot([x0 + 3, x1 - 3], [78.6, 78.6], color=leg["edge"], lw=0.8, alpha=0.55, zorder=3)
        bullets(leg["facts"], xc, x0 + 2.6, 75.6, 8.8)

        # ---- what the leg evaluates -----------------------------------------
        card(x0, x1, 30.5, 56.5, "white", leg["edge"], lw=1.6)
        ax.text(xc, 53.4, leg["job"], ha="center", va="top", fontsize=10.8,
                weight="bold", color="#263238", linespacing=1.4, zorder=3)
        # Titles run to two or three lines, so the bullet list starts below whichever it is
        # rather than at a fixed height the three-line title would collide with.
        bullets(leg["does"], xc, x0 + 2.6,
                53.4 - 3.5 * leg["job"].count("\n") - 5.3, 8.6, step=3.2)

        # ---- anchor tag ------------------------------------------------------
        card(x0 + 1.5, x1 - 1.5, 24.5, 29.5, leg["fill"], leg["fill"], lw=0)
        ax.text(xc, 27.0, leg["anchor"], ha="center", va="center", fontsize=9.0,
                style="italic", color=leg["edge"], zorder=3)

        down(xc, 62.5, 56.7)          # dataset -> job
        down(xc, 30.5, 29.7)          # job -> anchor tag
        elbow(50, 89.8, xc, 87.7, 88.7)   # framework -> dataset
        elbow(xc, 24.5, 50, 20.7, 22.0)   # anchor tag -> synthesis

    # ---- bottom: synthesis --------------------------------------------------
    card(28, 72, 11.5, 20.5, "#ede7f6", "#4527a0", lw=1.5)
    ax.text(50, 17.6, "Comprehensive Evaluation and Analysis", ha="center", va="center",
            fontsize=12.6, weight="bold", color="#263238", zorder=3)
    ax.text(50, 13.9, "(Network and Service Management Perspective)", ha="center", va="center",
            fontsize=8.6, color=INK, zorder=3)

    # ---- scope-anchor legend ------------------------------------------------
    ax.add_patch(FancyBboxPatch((1.5, 1.0), 97, 7.6,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec="#90a4ae", lw=1.0, ls="--", zorder=2))
    ax.text(50, 6.3, "TNSM scope anchors:    1) Management Functions    "
                     "2) Service Provisioning, Reliability & Quality Assurance",
            ha="center", va="center", fontsize=8.4, color=INK, zorder=3)
    ax.text(50, 3.2, "3) Enabling Technologies    4) Emerging Technologies & Standards    "
                     "5) Applications and Case Studies",
            ha="center", va="center", fontsize=8.4, color=INK, zorder=3)

    _save(fig, "fig1_framework.png")


def fig2_data_splits():
    """TRAIN / EVAL / CALIB block sizes and positive rates for all three datasets (§14.1)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {"TRAIN": "#1565c0", "EVAL": "#c62828", "CALIB": "#00838f"}

    for ax, (ds, blocks) in zip(axes, _splits().items()):
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
        (axes[0], t, "TelecomTS — early warning, 5 s", _base_rate("telecomts")),
        (axes[1], r, "RCAEval RE1/OB — detection", _base_rate("rcaeval")),
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
        (f"{scratch}/tts_preds.pkl", "TelecomTS — early warning, 5 s",
         _base_rate("telecomts")),
        (f"{scratch}/rcae_preds.pkl", "RCAEval RE1/OB — detection",
         _base_rate("rcaeval")),
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
