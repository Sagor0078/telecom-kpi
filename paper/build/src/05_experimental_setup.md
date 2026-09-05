# 5. Experimental Setup

## 5.1 Three-dataset evaluation architecture

A framework that claims 5G RAN relevance cannot be validated on server
telemetry alone. TrustNet-RCA is therefore evaluated on three public
datasets, each assigned a distinct and explicitly bounded job, with the
*identical* method run on all three so that the legs are comparable
rather than merely adjacent:

- **TelecomTS** \[22\] is the **primary 5G-domain benchmark**. It is
  derived from a 5G testbed and exposes named PHY/MAC/network-layer KPIs,
  which is what makes a radio-domain claim possible at all.
- **RCAEval** \[5\] (RE1 / Online Boutique) carries **service-management
  and RCA rigor**: annotated microservice failures with reproducible
  baselines.
- **SMD** \[7\] (machine-1-1) is retained strictly as an
  **out-of-domain generalization** check. Its 38 dimensions are generic
  server metrics, not radio KPIs, and no result on SMD is offered here as
  evidence about RAN behavior.

**Table 2. Per-leg data preparation.**

| | TelecomTS | RCAEval | SMD |
|---|---|---|---|
| Role | 5G RAN domain validity | RCA / service-mgmt rigor | out-of-domain generalization |
| Unit | 168 reconstructed sessions | 123 injected-failure cases | 1 machine (machine-1-1) |
| Sampling | 10 Hz | 1 Hz | 1 / minute |
| Channels | 18 named PHY/MAC/network KPIs | 24 common metric columns | 38 anonymized dims |
| Volume | 643,284 samples (~1,072 min) | 123 cases, 6-35 min pre-injection | 28,479 samples |
| Events | 109 anomaly segments, 11 types | 123 injections, 5 fault types | 8 segments |
| Localization ground truth | `affected_kpis` (named) | root-cause service + indicator | `interpretation_label` (dim indices) |
| Split | session-level chronological 60/20/20 | fault-stratified case split 60/20/20 | chronological TRAIN/EVAL/CALIB |
| Window / target | 3 s window, 5 s horizon | 60 s window, detection | 20 min window, 30 min horizon |

![](figures/fig2_data_splits.png){width="7.0in" height="2.2in"}

*Figure 2. TRAIN / EVAL / CALIB window counts and positive rates across
all three datasets. This is not a conventional train/validation/test
split: there is no separate validation block because hyperparameters were
fixed by hand rather than tuned, EVAL plays the role of the final test
set, and CALIB is reserved exclusively for the conformal quantile and the
reliability diagram.*

## 5.2 TelecomTS session reconstruction

TelecomTS ships as 128-sample windows at 10 Hz, but those windows are
sliding rather than independent: consecutive windows advance by a
32-sample (3.2 s) stride, so their overlapping regions are identical
across all 18 channels. Treating them as independent samples would leak
the evaluation block into the training block through the overlap. We
therefore reconstruct the continuous underlying series before splitting,
which is also what makes an early-warning target definable at all. The
overlap identity was verified rather than assumed: 285 window pairs were
checked on the anomaly shard with zero mismatches, and one shard rebuilds
to 99,680 timestamps (166.6 continuous minutes) with 7 holes totalling
28.8 s (0.288% of the timeline).

Two of the 18 channels (`UL_Protocol`, `DL_Protocol`) are categorical strings
and are label-encoded; the remaining 16 are numeric.

## 5.3 Target construction and features

We construct an early-warning target rather than a post-hoc detection
target, (2): $y_t = 1$ if a labeled anomaly occurs anywhere in $(t,\,t+H]$,
using only data up to and including $t$. Tabular features are the rolling mean,
standard deviation, and last value per KPI dimension over the window; the
sequence models additionally consume the raw window.

The horizon is re-parameterized per domain rather than copied across
domains. SMD's W = 20 min / H = 30 min are minutes on one-sample-per-minute
server telemetry; 5G radio degradation evolves on a seconds scale, so
TelecomTS uses W = 3 s / H = 5 s. Section 6.1 reports the horizon sweep
that justifies this choice empirically rather than by assertion.

## 5.4 Data splits

Each leg is split so that no window in the evaluation block shares source
samples with the fitting block.

*TelecomTS* is split at session level, chronologically, 60/20/20; the
held-out EVAL block contains 33 sessions and 26 anomaly segments at a
0.167 positive base rate.

*RCAEval* is split by case, **stratified by fault type**. An initial
chronological split by injection time was found to segregate fault types
perfectly — TRAIN received all CPU and memory cases, EVAL all packet-loss,
CALIB all disk — which is a confounded split rather than a hard one.
Section 6.1 reports what that error produced and why the corrected split
still does not rescue early warning on this leg. After stratification the
per-block positive rates are 0.513 / 0.499 / 0.513, and EVAL holds 25
cases spanning all five fault types at a 0.498 base rate.

*SMD*'s eight ground-truth segments are not spread evenly across the
28,479-step timeline: segments 1-5 fall in \[15849, 21195\], segments 6-8
in \[24679, 27556\], and \[0, 15849) is anomaly-free. A naive 50/20/30
chronological split would leave the earliest block with zero positive
examples. We instead cut at two points inside anomaly-free gaps so that
each of three disjoint chronological blocks contains at least one full
ground-truth segment: TRAIN (segments 1-3), EVAL (segments 4-5; held out
for every reported detection metric), and CALIB (segments 6-8; used only
to fit the conformal threshold and reliability diagram). The original
anomaly-free train file is added as extra normal background for model
fitting only.

**Table 3. SMD split summary.**

| **Block** | **Role** | **Windows** | **Positive rate** |
|---|---|---|---|
| TRAIN (+ background) | Model fitting | 24,751 | 6.6% |
| EVAL | All reported detection metrics | 4,151 | 28.6% |
| CALIB | Conformal threshold + ECE only | 5,430 | 1.7% |

## 5.5 Models and hyperparameters

The model hierarchy is identical across legs: 11 base models spanning 8
families, plus 6 ensembles (5 deep ensembles at M = 5 members and one
cross-family soft vote). Base models are a dummy prior, Logistic
Regression, Random Forest, Extra Trees, XGBoost, LightGBM, an MLP, a 1D
ResNet, LSTM, GRU, and a Transformer encoder.

LightGBM: 300 trees, max depth 5, learning rate 0.05, and
`scale_pos_weight` set to the train-set negative/positive ratio. Transformer: input projection to
$d_{\text{model}} = 32$, learned positional embedding, 2 encoder layers (4 attention
heads, feed-forward dimension 64, dropout 0.1), mean-pooling over time, a
2-layer classification head; trained with Adam (lr = 1e-3), weighted binary
cross-entropy, batch size 256, 6 epochs. Granger causality: SSR F-test,
lags 1-5, statsmodels implementation, with Benjamini-Hochberg FDR control
across the tested dimensions. Split-conformal: least-ambiguous-set
classifier (LAC) \[4\], target coverage 90% ($\alpha = 0.10$), calibrated on
the CALIB block. Personalized PageRank: networkx implementation on a
reversed, hand-specified illustrative topology graph (Section 6.8), not fit
to SMD, which has no topology.

Hyperparameters were fixed by hand from the SMD leg and reused verbatim on
TelecomTS and RCAEval. This is deliberate — it keeps the three legs
comparable — but it means no leg is tuned, and the reported numbers are
therefore untuned-baseline numbers rather than best-achievable ones.

## 5.6 Metrics

Prediction: AUROC, AUPRC, and Expected Calibration Error (ECE, 10
equal-width bins) on the held-out EVAL block, with AUPRC read against each
leg's base rate since the base rates differ by a factor of three across
legs. Attribution: precision@k and recall@k of the top-k SHAP-ranked KPI
dimensions against the ground-truth label set, compared to the chance level
$k/d$. Causal filtering: Granger SSR F-test p-value (minimum across lags $1$ to $5$,
BH-FDR corrected) and Pearson correlation, reported side by side.
Calibration: empirical coverage of the conformal prediction set against the
90% target. Explanation quality: deletion-test faithfulness margin and
Spearman rank stability under input perturbation, both defined in Section
6.5. Early-warning usefulness: lead time, the gap between the first
sustained (>= 2 consecutive steps) crossing of a 0.5 risk threshold and the
official onset of each ground-truth segment.
