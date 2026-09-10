# 6. Results

## 6.1 Component 1: Early warning across three domains

Table 4 reports the full model hierarchy on the TelecomTS EVAL block at a
5 s horizon. This is the discriminative leg: a 0.160 AUPRC spread
separates the best model from the worst non-trivial one.

**Table 4. TelecomTS early warning, 5 s ahead (33 EVAL sessions, 26
segments, base rate 0.167).**

| **Model** | **Family** | **AUROC** | **AUPRC** | **ECE** | **F1** | **Fit (s)** |
|---|---|---|---|---|---|---|
| **XGBoost** | Boosted tree | **0.920** | **0.794** | 0.120 | 0.750 | 1.4 |
| Random Forest | Bagged tree | 0.885 | 0.762 | 0.120 | 0.801 | 4.4 |
| Extra Trees | Bagged tree | 0.880 | 0.759 | 0.119 | 0.757 | 1.1 |
| 1D ResNet (M=5) | Deep ensemble | 0.900 | 0.743 | 0.079 | 0.746 | 102.1 |
| Soft vote | Ensemble | 0.872 | 0.741 | 0.333 | 0.758 | 1227.7 |
| Heterogeneous (M=5) | Deep ensemble | 0.887 | 0.735 | 0.080 | 0.724 | 554.5 |
| MLP (M=5) | Deep ensemble | 0.846 | 0.706 | 0.090 | 0.730 | 8.8 |
| Transformer (M=5) | Deep ensemble | 0.836 | 0.704 | **0.063** | 0.741 | 301.5 |
| Transformer | Attention | 0.845 | 0.702 | **0.063** | 0.751 | 60.1 |
| LightGBM | Boosted tree | 0.859 | 0.698 | 0.121 | 0.678 | 2.3 |
| MLP | Feedforward | 0.848 | 0.680 | 0.102 | 0.714 | 1.5 |
| GRU (M=5) | Deep ensemble | 0.853 | 0.678 | 0.080 | 0.673 | 76.4 |
| LSTM | Recurrent | 0.822 | 0.667 | 0.103 | 0.664 | 12.6 |
| Logistic Reg. | Linear | 0.826 | 0.663 | 0.054 | 0.672 | 0.6 |
| GRU | Recurrent | 0.856 | 0.651 | 0.087 | 0.645 | 13.7 |
| LSTM (M=5) | Deep ensemble | 0.821 | 0.646 | 0.101 | 0.660 | 65.8 |
| 1D ResNet | Convolutional | 0.813 | 0.634 | 0.080 | 0.637 | 21.1 |
| Dummy prior | Naive | 0.500 | 0.167 | 0.082 | 0.287 | n/a |

On RCAEval the same hierarchy is applied to a detection target. The leg is
saturated: every non-naive model reaches at least 0.990 AUROC, with XGBoost
at 0.996 AUROC / 0.997 AUPRC / 0.012 ECE and even Logistic Regression at
0.993 / 0.994. Post-injection CPU, memory, and latency shifts in a
microservice mesh are large and immediate, so this leg discriminates poorly
between methods. We therefore use RCAEval for its assigned job (RCA quality
and baseline comparison) rather than as a detection benchmark, and do not
report its full table here.

On SMD, the out-of-domain leg at a 30-minute horizon, Extra Trees reaches
0.953 AUROC / 0.905 AUPRC and the Transformer 0.924 / 0.886 against a
0.286 base rate.

**Table 5. Cross-dataset synthesis, held-out EVAL block per leg.**

| **Leg** | **Task** | **Best model** | **AUROC** | **AUPRC** | **Base rate** | **AUPRC lift** |
|---|---|---|---|---|---|---|
| TelecomTS | early warning, 5 s | XGBoost | 0.920 | 0.794 | 0.167 | **4.7x** |
| RCAEval | detection | XGBoost | 0.996 | 0.997 | 0.498 | 2.0x |
| SMD | early warning, 30 min | Extra Trees | 0.953 | 0.905 | 0.286 | 3.2x |

![](figures/fig3_model_comparison.png){width="7.0in" height="3.5in"}

*Figure 3. AUPRC across the full model hierarchy on both primary legs,
colored by model family.*

![](figures/fig6_roc_pr_curves.png){width="3.4in" height="2.8in"}

*Figure 6. ROC and precision-recall curves for the leading models.*

**Finding 1: tree ensembles win accuracy on every leg, and attention does
not pay for itself.** The ordering is stable across three sampling rates (10
Hz / 1 Hz / one per minute), three channel counts (18 / 24 / 38), and three
domains.

**Table 6. Best tree model vs. Transformer, per leg.**

| **Leg** | **Best tree AUPRC** | **Transformer AUPRC** | **Delta** | **Fit-time ratio** |
|---|---|---|---|---|
| TelecomTS | 0.794 | 0.702 | **+0.092** | 43x |
| RCAEval | 0.997 | 0.994 | +0.003 | 82x |
| SMD | 0.905 | 0.886 | +0.019 | ~4x |

This is a negative result for the attention mechanism and we report it as
one. Attention costs 4-82x the training time for no accuracy gain on any
leg. It also removes a risk flagged when the comparison was designed:
because the best detector is a tree model on every leg, the SHAP
TreeExplainer path of Component 2 stays exact throughout, and attribution
numbers remain comparable across datasets instead of switching explainer
family mid-study.

**Horizon feasibility is bounded by the data, not the model.** Table 7
sweeps the TelecomTS horizon. The headline column (AUPRC over all EVAL
windows) is optimistic, because windows already inside an anomaly are
trivially separable; the *clean* column restricts scoring to genuine
pre-onset windows and is the honest measure of anticipation.

**Table 7. TelecomTS horizon sweep. "Clean" excludes windows already
inside an anomaly segment.**

| **Horizon (s)** | **AUPRC (all)** | **Lift (all)** | **AUPRC (clean)** | **AUROC (clean)** | **Clean positives** |
|---|---|---|---|---|---|
| 1 | 0.823 | 6.0x | 0.047 | 0.605 | 26 |
| 2 | 0.802 | 5.6x | 0.079 | 0.568 | 52 |
| 5 | 0.794 | 4.7x | 0.162 | 0.696 | 127 |
| 10 | 0.760 | 3.8x | 0.271 | 0.742 | 228 |
| 20 | 0.766 | 4.0x | 0.335 | 0.753 | 245 |
| 30 | 0.730 | 4.3x | 0.316 | 0.768 | 225 |

The two columns tell opposite stories, and the disagreement is the finding.
Measured over all windows, shorter horizons look better; measured over
genuine pre-onset windows only, they look far worse (AUPRC 0.047 at 1 s
against a 0.008 base rate, AUROC 0.605, barely above chance). Anticipation
improves monotonically with horizon up to 20 s because longer horizons admit
more pre-onset positives to learn from. We report the clean column as the
operative one and treat the 5 s headline of Table 4 as an upper bound rather
than a capability claim.

![](figures/fig9_horizon_sweep.png){width="7.0in" height="2.7in"}

*Figure 9. Horizon sweep on TelecomTS, all-window and clean pre-onset
scoring shown together.*

The ceiling is set by how much pre-onset telemetry each incident actually
provides. Across the 109 TelecomTS anomaly segments the median pre-onset
runway is 11.1 s and the maximum is 21.3 s: 94% of events afford at least 1
s of warning, 84% at least 5 s, 71% at least 10 s, and **none** affords 30
s. A 30 s horizon is therefore not a modelling failure but an ill-posed
request on this data. The seven Jamming events (the one anomaly type in
TelecomTS that is a real over-the-air phenomenon rather than an injected
one) have exactly zero runway: their onset coincides with the start of the
session, so no precursor exists to detect.

![](figures/fig5_preonset_runway.png){width="7.0in" height="2.4in"}

*Figure 5. Pre-onset runway per anomaly type and the resulting horizon
feasibility.*

**Difficulty varies sharply by anomaly type**, which a single aggregate
number conceals. Table 8 breaks the TelecomTS result down by type.

**Table 8. TelecomTS early warning by anomaly type (5 s horizon).**

| **Anomaly type** | **Windows** | **AUROC** | **AUPRC** |
|---|---|---|---|
| Antenna Failure | 102 | 0.935 | 0.823 |
| Resource Allocation Bugs | 53 | 0.906 | 0.663 |
| Faulty RF Filters (Temporal) | 141 | 0.933 | 0.575 |
| Co-Channel Interference (Severe) | 91 | 0.957 | 0.514 |
| Doppler Shift (Severe) | 62 | 0.923 | 0.304 |
| Faulty Handover Algorithm (Too Frequent) | 74 | 0.882 | 0.231 |
| High Network Congestion (Sudden Spike) | 16 | 0.971 | 0.125 |
| Buffer Overflow (Gradual Buildup) | 48 | 0.877 | 0.116 |
| Co-Channel Interference (Mild) | 56 | 0.878 | 0.101 |

AUPRC spans 0.101 to 0.823, an eightfold range, while AUROC stays in a
narrow 0.877-0.971 band. An operator reading only AUROC would conclude the
system performs uniformly well across fault types; the precision-recall view
shows it does not. Severe, physically abrupt faults (antenna failure) are
caught reliably; mild interference and gradual buffer buildup are close to
unusable at this horizon.

![](figures/fig10_per_anomaly_type.png){width="7.0in" height="3.6in"}

*Figure 10. Per-anomaly-type AUROC and AUPRC on TelecomTS, showing that an
aggregate score hides an eightfold spread in precision-recall performance.*

**Correction: early warning is ill-posed on RCAEval.** Our dataset strategy
originally assigned early-warning experiments to TelecomTS *and* RCAEval.
That did not survive contact with the data, and we record the correction
rather than dropping it silently. RCAEval's faults are externally scheduled
injections: the system is normal up to the injection instant, so no
precursor exists at any horizon. A first attempt at a 300 s early-warning
target produced AUROC 0.436 to 0.534 across all models, no better than
chance. Two causes were separable. The intrinsic one is that anticipating a
scheduled intervention from pre-intervention telemetry is not a learnable
problem. The second was our own method error: splitting cases
chronologically by injection time perfectly segregated fault types across
blocks, and cases are independent experiments, so the split must be
stratified by fault type. It now is. The consequence is that **the
early-warning anchor of this paper rests on TelecomTS alone**, a real
concentration of risk that we state in Section 8 rather than smooth over.

## 6.2 Accuracy and calibration pull in opposite directions

**Finding 2.** On TelecomTS the tree models average ECE 0.120 while the deep
ensembles average 0.082, a 1.5x calibration gap running *against* the
accuracy gap. The best-calibrated non-trivial models are the Transformer and
Deep ensemble (Transformer) at ECE 0.063; the most accurate, XGBoost, sits
at 0.120.

![](figures/fig4_accuracy_vs_calibration.png){width="7.0in" height="3.1in"}

*Figure 4. AUPRC against ECE on TelecomTS. The most accurate models are
not the best calibrated. The soft-vote ensemble is excluded: it averages
percentile ranks rather than probabilities, so its scores are uniform by
construction and ECE against them is not meaningful.*

This is the empirical case for the conformal layer of Component 5, and it
is stronger than the design argument that preceded it, because **the
trade-off is avoidable**. Conformal prediction wraps the highest-accuracy
model in a distribution-free coverage guarantee, delivering tree accuracy
without requiring tree calibration and without the deep ensemble's
102-554 s training cost. Under a network-management framing this is the
contribution: conformal prediction as a management-plane primitive,
supported by a measured tension rather than an assertion.

**Finding 3: deep ensembling helps unevenly, and mainly where single-model
variance is high.**

**Table 9. Single model vs. 5-member deep ensemble on TelecomTS.**

| **Architecture** | **AUPRC, single $\rightarrow$ M=5** | **ECE, single $\rightarrow$ M=5** |
|---|---|---|
| MLP | 0.680 $\rightarrow$ 0.706 (+0.026) | 0.102 $\rightarrow$ 0.090 (-0.011) |
| 1D ResNet | 0.634 $\rightarrow$ 0.743 (**+0.109**) | 0.080 $\rightarrow$ 0.079 (-0.002) |
| LSTM | 0.667 $\rightarrow$ 0.646 (**-0.022**) | 0.103 $\rightarrow$ 0.101 (-0.003) |
| GRU | 0.651 $\rightarrow$ 0.678 (+0.026) | 0.087 $\rightarrow$ 0.080 (-0.006) |
| Transformer | 0.702 $\rightarrow$ 0.704 (+0.003) | 0.063 $\rightarrow$ 0.063 (-0.000) |

The 1D ResNet gains most because its single-model variance is highest; the
LSTM *loses* accuracy. ECE improvements are consistent in sign but small
(0.000 to 0.011), materially weaker than the deep-ensemble literature
suggests, and reported as measured. Two practical notes: the MLP matches
LSTM and GRU accuracy at roughly a tenth of the cost (1.5 s vs. 12.6 and
13.7 s), and the 5-member 1D ResNet ensemble is the best
accuracy-calibration compromise on TelecomTS at AUPRC 0.743 and ECE 0.079.

![](figures/fig7_deep_ensemble.png){width="7.0in" height="2.5in"}

*Figure 7. Single model against 5-member deep ensemble, AUPRC and ECE.*

**Finding 4: the soft-vote ensemble does not earn its keep.** On TelecomTS
it costs 1,228 s (the sum of every member's training time) for AUPRC 0.741,
below XGBoost alone at 1.4 s. On RCAEval it is strictly dominated. On an
operational cost basis this is a recommend-against.

![](figures/fig12_efficiency.png){width="7.0in" height="2.8in"}

*Figure 12. Accuracy against training cost. Tree models occupy the
favorable corner on every leg.*
