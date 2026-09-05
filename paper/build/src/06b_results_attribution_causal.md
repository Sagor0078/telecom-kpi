## 6.3 Component 2: Root-cause attribution

Because the best predictor is a tree model on every leg (Section 6.1), the
TreeExplainer attribution is exact rather than approximate. For every SMD
ground-truth segment we take the attribution at the last window before
onset, aggregate $|\phi_i|$ from (4) across each KPI dimension's
mean/std/last features, and rank dimensions. Table 10 reports precision@k = recall@k (k
set to the ground-truth set size) per segment; EVAL and CALIB segments are
genuinely out-of-sample, while TRAIN-block segments are in-sample and shown
only for contrast.

**Table 10. Component 2 results: SHAP-ranked root-cause attribution vs.
ground truth (SMD).**

| **Segment** | **Block** | **k** | **Precision@k = Recall@k** |
|---|---|---|---|
| 15849-16368 | train (in-sample) | 7 | 0.571 |
| 16963-17517 | train (in-sample) | 31 | 0.968 |
| 18071-18528 | train (in-sample) | 8 | 0.500 |
| 19367-20088 | eval (out-of-sample) | 14 | 0.571 |
| 20786-21195 | eval (out-of-sample) | 7 | 0.571 |
| 24679-24682 | calib (out-of-sample) | 4 | 0.500 |
| 26114-26116 | calib (out-of-sample) | 4 | 0.250 |
| 27554-27556 | calib (out-of-sample) | 4 | 0.750 |

The out-of-sample (EVAL + CALIB) mean precision/recall@k is 0.529, against a
mean chance level of approximately 0.174 for the corresponding k values,
well above chance but far from perfect, which we report as the honest
finding rather than only the best-case segment. Section 6.5 returns to the
three out-of-sample segments where localization is wrong, because *which*
events those are turns out to matter more than the mean.

Localization is scored on SMD only. Both TelecomTS (named `affected_kpis`)
and RCAEval (root-cause service and indicator) ship localization ground
truth and both loaders expose it, but the ranking is not yet scored on
either; that is the immediate next experiment (Section 9).

## 6.4 Component 3: Correlation vs. causal evidence

For the representative segment 20786-21195, Table 11 tests the top-5
SHAP-ranked KPI dimensions for Granger causality against a fixed target
dimension (KPI 14, present in every ground-truth set) over a local
pre/during-onset window, with Benjamini-Hochberg FDR control across the
tested dimensions.

**Table 11. Component 3 results: Granger-causality filtering of SHAP-ranked
candidates.**

| **KPI dim.** | **In ground truth** | **Pearson r** | **Granger p (min)** | **Verdict** |
|---|---|---|---|---|
| 7 | No | -0.075 | 0.199 | Correlational only |
| 9 | Yes | 0.951 | 0.000 | Causal evidence |
| 13 | Yes | 0.586 | 0.025 | Causal evidence |
| 30 | No | 0.046 | 0.450 | Correlational only |
| 33 | No | 0.026 | 0.791 | Correlational only |

The two dimensions that are both highly correlated and in the ground-truth
root-cause set (9, 13) pass the Granger test; the three that are only SHAP-
flagged noise (weak correlation, absent from ground truth) fail it, the
intended separation, though demonstrated here on one incident rather than
across a statistically powered sample (Section 8).

Note the relationship to Section 6.5: this filter is the one signal in the
framework that was *not* included in the trust-score test, and is therefore
the most promising candidate for the localization-quality signal that the
tested signals failed to provide.
