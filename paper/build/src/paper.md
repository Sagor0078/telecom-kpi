**Toward Trustworthy Root-Cause Analysis in Telecom Networks:**

**A Layered Attribution-Causal-Conformal Framework and Proof-of-Concept
Evaluation on Public KPI Telemetry**

*Author Name(s) to be added*

*Affiliation to be added*

**ABSTRACT**

Telecom operators increasingly deploy machine learning to predict
service degradation, but an accurate prediction is not the same as a
trustworthy one: an engineer also needs to know which KPI is
responsible, whether that KPI is merely correlated with the fault or
plausibly causes it, why the system believes so, and how confident to
be. Prior work on root-cause analysis (RCA) for multivariate telemetry
typically evaluates a single mechanism \-- feature attribution, causal
discovery, or uncertainty quantification \-- in isolation; no study we
identified jointly integrates and evaluates all three against
ground-truth root-cause labels. We propose a five-component framework
that chains (1) a probabilistic early-warning predictor, (2) exact
feature attribution, (3) a causal-precedence filter, (4) an
evidence-grounded natural-language explainer, and (5) distribution-free
uncertainty calibration, and we report a reproducible proof-of-concept
evaluation on the public Server Machine Dataset (SMD), selected
specifically because it provides ground-truth root-cause labels that
most public network-telemetry datasets lack. On a held-out evaluation
block, a Transformer encoder reaches 0.934 AUROC / 0.884 AUPRC for
early-warning prediction versus 0.898 / 0.825 for a gradient-boosted
baseline; out-of-sample root-cause attribution achieves 0.529 mean
precision/recall@k against ground truth, well above the \~0.174 chance
level; a Granger-causality filter correctly separates
ground-truth-consistent attribution candidates from spurious ones in the
incident examined; split-conformal calibration attains its 90% target
coverage (0.955 empirical) while routing 20% of cases to an explicit
human-escalation flag; and early-warning lead time is shown to be highly
heterogeneous across incidents (44 minutes to a complete miss), a
finding we argue is itself a trustworthiness result that point-accuracy
metrics would hide. This paper is explicitly scoped as a single-dataset,
baseline-free proof of concept intended to establish feasibility and
measurement protocol; we define the multi-dataset, multi-baseline,
statistically validated study required to generalize these findings as
immediate future work.

**Keywords:**

trustworthy AI; root-cause analysis; AIOps; conformal prediction; causal
inference; explainable AI; telecom network management; time-series
anomaly prediction

# 1. Introduction

Modern telecom networks generate continuous telemetry \-- latency,
packet loss, throughput, jitter, signal strength, handover-failure rate,
CPU/memory utilization, alarms, and configuration-change events \-- and
operators increasingly rely on machine learning to detect degradation
before it escalates into a service disruption. The dominant research
emphasis, however, has been point predictive accuracy: does the model
flag the right minute? This framing is insufficient for an operational
setting in which an automated flag triggers costly human or automated
remediation. An engineer facing a flagged degradation needs four
additional answers that accuracy alone does not provide: which KPI is
implicated, whether that KPI\'s relationship to the fault is causal or
merely coincidental, why the system believes so in terms the engineer
can audit, and how much to trust the flag at all.

Feature-attribution methods such as SHAP \[1\] answer the first of these
questions but are, by construction, correlational \-- a KPI can score
highly because it moves together with the fault without driving it.
Causal-discovery methods such as Granger causality \[2\] and
constraint-based causal graphs \[3\] address this but are rarely paired
with a calibrated confidence signal, so a causal claim is reported with
the same false certainty as a correlational one. Conformal prediction
\[4\] supplies exactly that calibrated confidence but is, to our
knowledge, not evaluated jointly with attribution and causal filtering.
Existing RCA benchmarks \[5\], \[6\] score attribution methods against
each other but do not ablate the value that a causal filter or a
calibration layer adds on top.

This paper makes three contributions. First, we propose a five-component
framework \-- predict, attribute, causally filter, calibrate, explain
\-- that makes each of these questions an explicit, separately-evaluable
pipeline stage rather than an implicit property of one end-to-end model.
Second, we implement the framework using standard, off-the-shelf methods
at each stage (a Transformer/LightGBM predictor, SHAP attribution, a
Granger-causality filter, split-conformal calibration, and a
personalized-PageRank graph view) and report a reproducible
proof-of-concept evaluation on the public Server Machine Dataset (SMD)
\[7\], chosen specifically because it is one of the few public
multivariate-telemetry datasets with ground-truth root-cause labels,
allowing the attribution and causal-filtering stages to be scored
against a known answer rather than only inspected qualitatively. Third,
we report results honestly at the scope we can currently support \-- one
dataset, one machine, no external baselines, no multi-seed statistical
testing \-- and define the concrete experimental programme (Section 9)
required to elevate these findings from a proof of concept to
generalizable evidence.

The remainder of this paper is organized as follows. Section 2 reviews
related work across classical ML, deep learning, transformer,
graph-neural-network, causal, and LLM-based approaches to RCA, and
situates the identified gap. Section 3 formalizes the prediction,
attribution, causal-filtering, and calibration problems. Section 4
describes the proposed framework and its five components. Section 5
details the experimental setup. Section 6 reports results for each
component. Section 7 discusses the findings, Section 8 states
limitations and threats to validity, Section 9 outlines practical
implications and future work, and Section 10 concludes.

# 2. Related Work

Classical machine learning. Tree ensembles (isolation forests, random
forests, gradient-boosted trees) remain a strong, auditable baseline for
KPI anomaly detection; a 2024 benchmark of ML methods on network anomaly
detection confirms they remain competitive with deep alternatives on
tabular telemetry \[8\].

Deep learning and Transformers. Recurrent and variational architectures
dominate multivariate KPI anomaly detection: OmniAnomaly \[7\] models
KPI vectors with a stochastic RNN and is also the source of the SMD
dataset used in this study; LSTM-NDT \[9\] forecasts KPI values and
flags deviation with a non-parametric dynamic threshold. Self-attention
architectures improve on recurrence for long-range temporal dependence
\[10\]; TranAD \[11\] uses an adversarially trained transformer encoder
for multivariate anomaly detection. Closest to our domain, \[12\] fuses
a transformer temporal encoder with a graph neural network over 5G RAN
topology for anomaly root-causing \-- effectively Components 1 and 2 of
our framework, evaluated on real (non-public) RAN data but without a
causal-versus-correlational distinction or calibrated uncertainty.

Graph neural networks. Once more than one KPI stream and a known
topology are available, root-cause evidence can propagate along
dependency edges instead of being scored per KPI in isolation. MicroRCA
\[13\] builds an attributed service graph and ranks root causes with
personalized PageRank; GDN \[14\] learns the dependency graph itself and
scores per-sensor deviation with attention.

Causal reasoning. GNN and attribution scores remain correlational: they
indicate that a KPI co-moved with the outcome, not that changing it
would change the outcome. Constraint-based causal discovery (the PC/FCI
family \[3\]) and neural Granger causal discovery \[15\] provide tools
to test temporal precedence and conditional independence; a recent
comparative study \[16\] benchmarks PC, FCI, Granger, fGES, and LiNGAM
as causal-graph builders for RCA.

RCA benchmarking. RCAEval \[5\] standardizes precision@k, MRR, and MAP@k
evaluation across 735 failure cases spanning eleven fault types in three
microservice systems; LEMMA-RCA \[6\] extends multi-domain, multi-modal
RCA evaluation to IT and operational-technology systems. Neither
benchmark ablates the joint contribution of causal filtering and
calibrated uncertainty on top of attribution \-- the gap this paper\'s
proof of concept targets.

LLM-based approaches. Recent AIOps systems use large language models
strictly as narrators over verified, tool-collected evidence rather than
as free-form diagnosers: RCACopilot-style systems \[17\] aggregate
multi-modal diagnostic evidence into incident narratives, and LLM-agent
RCA work \[18\] shows that grounding the LLM in retrieved telemetry is
what prevents hallucinated causes. We reuse this narrate-don\'t-diagnose
pattern for Component 4.

Uncertainty quantification. Conformal prediction \[4\] gives
distribution-free, finite-sample coverage guarantees without assuming a
noise model for KPI residuals, and is increasingly positioned as a
prerequisite for trustworthy AI deployment \[19\], including in
5G-specific deep models \[20\]. It is used here as the calibration layer
for Component 5.

**Table 1. Related-work comparison matrix.**

  ----------------------------------------------------------------------------------------------------
  **Ref.**    **Problem**         **Method**        **Dataset**     **Metrics**   **Gap relevant to
                                                                                  this work**
  ----------- ------------------- ----------------- --------------- ------------- --------------------
  \[7\]       MTS anomaly         Stochastic        SMD             F1            No RCA, no causal
              detection           RNN+VAE           (introduced)                  filter, no
                                                                                  calibration

  \[5\]       RCA benchmarking    Multi-method eval 3 microservice  PR@K, MRR,    No causal/conformal
                                  harness           systems         MAP@K         ablation; no telecom
                                                                                  domain

  \[6\]       Multi-domain RCA    Multi-modal       IT + OT systems \-            No telecom RAN; no
              dataset                                                             causal+calibration
                                                                                  study

  \[15\]      Causal-graph RCA    Neural Granger    Microservices   PR@K          No calibrated
                                  discovery                                       confidence layer

  \[12\]      RAN anomaly RCA     GNN + Transformer Real 5G RAN     \-            No
                                                    (private)                     causal/correlation
                                                                                  split; no
                                                                                  calibration

  \[4\]       Distribution-free   Split-conformal   General ML      Coverage      Not applied to RCA
              UQ                                                                  specifically
  ----------------------------------------------------------------------------------------------------

# 3. Problem Formulation

Let X_t in R\^d denote the KPI vector observed at minute t (d = 38 in
this study; in a real deployment d indexes named telecom KPIs such as
latency, packet loss, jitter, or handover-failure rate). A window of the
last W minutes is written X\_(t-W+1:t).

**Early-warning prediction (Component 1).**

> y_hat_t = f_theta( X\_(t-W+1:t) ) in \[0, 1\]
>
> y_t = 1 iff an anomaly occurs at some tau in (t, t+H\] (H = prediction
> horizon)
>
> L(theta) = - sum_t w\_{y_t} \[ y_t log(y_hat_t) + (1 - y_t) log(1 -
> y_hat_t) \]

f_theta ranges over a gradient-boosted tree ensemble and a Transformer
encoder in this study (Section 4.1); w\_{y_t} is a class-imbalance
weight.

**Attribution (Component 2).**

> phi_i(x) = sum\_{S subseteq N\\{i}} \[\|S\|!(\|N\|-\|S\|-1)!/\|N\|!\]
> \* \[ f(S u {i}) - f(S) \]

phi_i(x) is the Shapley value of feature i for input x; per-KPI
importance aggregates the Shapley values of that KPI\'s derived tabular
features (Section 5.2).

**Causal filter (Component 3).**

For candidate KPI i and target KPI j, i is retained as causal evidence
for j if, at some lag l in {1,\...,L}, the null hypothesis that i\'s
past does not improve the prediction of j\'s future (beyond j\'s own
past) is rejected by an SSR F-test at p \< 0.05:

> H0: Y_j,t \~ Y_j,(t-1..t-l) vs. H1: Y_j,t \~ Y_j,(t-1..t-l) +
> X_i,(t-1..t-l)

**Calibration (Component 5).**

Given a calibration set with known labels, the nonconformity score for
point i is s_i = 1 - P(true class \| x_i). The (1-alpha)-quantile of
{s_i}, denoted q_hat, defines a prediction set for a new point x:

> C(x) = { c : 1 - P(c \| x) \<= q_hat } with marginal guarantee P( y in
> C(x) ) \>= 1 - alpha

# 4. Proposed Framework

The framework decomposes the problem in Section 1 into five components,
each with its own input, output, and validation criterion, chained as
shown in Figure 1. This decomposition is deliberate: it lets each stage
be evaluated \-- and, in Section 6, scored against ground truth \--
independently, rather than treating trustworthiness as an emergent,
unmeasured property of one opaque end-to-end model.

![](./media/image1.png){width="5.8in" height="3.9955555555555557in"}

*Figure 1. Proposed five-component framework. Solid arrows show data
flow; Components 2 (graph view) and 5 are illustrative/parallel branches
feeding the final explanation stage.*

## 4.1 Component 1 \-- Predict degradation before disruption

Input: a W-minute KPI window per network element. Method: a Transformer
encoder (self-attention over the window) as the primary model,
benchmarked against a gradient-boosted tree baseline (LightGBM) on
rolling statistical features \-- retained as a cheap, auditable fallback
and as the model exactly explained in Component 2. Output: P(degradation
in the next H minutes), a continuously updated early-warning score
rather than a post-hoc flag.

## 4.2 Component 2 \-- Identify probable root causes

Input: the flagged window, the full KPI vector, and (where available)
the NE/site topology graph. Method: exact SHAP attribution on the
tree-ensemble predictor ranks which KPI dimensions drove the flagged
prediction; a graph view \-- personalized PageRank over the topology,
seeded at high-deviation nodes \-- ranks which network element is the
propagation source rather than merely a symptom, in the spirit of
MicroRCA \[13\]. Output: a ranked (KPI or NE, contribution score) list,
explicitly still correlational at this stage.

## 4.3 Component 3 \-- Distinguish correlation from causal evidence

Input: Component 2\'s ranked candidates and their raw time series.
Method: a Granger-causality test per candidate (Section 3), plus, where
a configuration-change timestamp coincides with a candidate\'s onset, a
before/after comparison treated as a quasi-experiment. Output: each
candidate labeled correlational only or causal evidence, with its test
statistic attached; Granger causality is necessary-but-not-sufficient
evidence for true causality, and this label is reported as a confidence
tier, not a proof.

## 4.4 Component 4 \-- Explain in engineer-understandable terms

Input: the outputs of Components 1, 2, 3, and 5. Method: a template
grounded strictly in the evidence already computed \-- not a
free-generation LLM call \-- following the narrate-don\'t-diagnose
pattern from RCACopilot- and RCAgent-style systems \[17\], \[18\]; an
LLM may phrase the narrative fluently but may not assert any fact absent
from the evidence. Output: a short report stating what is predicted, how
confident, which KPI/NE is implicated, whether that evidence is causal
or correlational, and what changed immediately before onset.

## 4.5 Component 5 \-- Quantify uncertainty and confidence

Input: Component 1\'s raw probability and a held-out calibration set.
Method: split-conformal prediction (Section 3) wraps the probability
into a set with a distribution-free, finite-sample coverage guarantee,
independent of whether the underlying model is well-calibrated on its
own; reliability diagrams / Expected Calibration Error (ECE) are
reported as a diagnostic. Output: a calibrated confidence level plus an
explicit abstain-and-escalate flag for low-confidence (ambiguous)
predictions.

# 5. Experimental Setup

## 5.1 Dataset

We use machine-1-1 from the NetManAIOps Server Machine Dataset (SMD)
\[7\], obtained directly from the dataset\'s public repository. SMD
provides 28,479 one-minute samples across 38 KPI dimensions (CPU,
memory, network, and load metrics) for a train split (guaranteed
anomaly-free historical operation) and a test split (labeled monitoring
period), together with interpretation_label annotations giving, for each
of eight ground-truth anomaly segments, the exact root-cause KPI
dimensions. We use SMD as a public, license-open proxy for telecom
NE/CPU/memory/traffic KPI monitoring: it is one of the few public
multivariate-telemetry datasets with root-cause ground truth, which is
precisely what is required to score, rather than only inspect, the
attribution and causal-filtering stages. Real RAN-level KPI datasets
(latency, jitter, handover-failure rate) with root-cause labels are
operator- and vendor-confidential and are not publicly available; this
substitution is a limitation, addressed in Section 8.

## 5.2 Target construction and features

We construct an early-warning target rather than a post-hoc detection
target: y_t = 1 if a labeled anomaly occurs anywhere in (t, t+H\], using
only data up to and including t (W = 20 minutes, H = 30 minutes).
Tabular features are the rolling mean, standard deviation, and last
value per KPI dimension over the window (114 features); the Transformer
additionally consumes the raw W x 38 window.

## 5.3 Data split

The eight ground-truth segments are not spread evenly across the
28,479-step test timeline: segments 1-5 fall in \[15849, 21195\],
segments 6-8 in \[24679, 27556\], and \[0, 15849) is anomaly-free. A
naive 50/20/30 chronological split would leave the earliest block with
zero positive examples. We instead cut at two points inside anomaly-free
gaps so that each of three disjoint, non-overlapping chronological
blocks contains at least one full ground-truth segment: TRAIN (earliest;
segments 1-3; fits the models), EVAL (middle; segments 4-5; held out for
every reported detection metric), and CALIB (latest; segments 6-8; used
only to fit the conformal threshold and reliability diagram, never for
model fitting or selection). The original train file (guaranteed
anomaly-free, chronologically earlier still) is added as extra normal
background for model fitting only.

**Table 2. Data split summary.**

  -----------------------------------------------------------------------
  **Block**         **Role**          **Windows**       **Positive rate**
  ----------------- ----------------- ----------------- -----------------
  TRAIN (+          Model fitting     24,751            6.6%
  background)                                           

  EVAL              All reported      4,151             28.6%
                    detection metrics                   

  CALIB             Conformal         5,430             1.7%
                    threshold + ECE                     
                    only                                
  -----------------------------------------------------------------------

## 5.4 Models and hyperparameters

LightGBM: 300 trees, max depth 5, learning rate 0.05, scale_pos_weight
set to the train-set negative/positive ratio. Transformer: input
projection to d_model = 32, learned positional embedding, 2 encoder
layers (4 attention heads, feed-forward dimension 64, dropout 0.1),
mean-pooling over time, a 2-layer classification head; trained with Adam
(lr = 1e-3), weighted binary cross-entropy, batch size 256, 6 epochs.
Granger causality: SSR F-test, lags 1-5, statsmodels implementation.
Split-conformal: least-ambiguous-set classifier (LAC) \[4\], target
coverage 90% (alpha = 0.10), calibrated on the CALIB block. Personalized
PageRank: networkx implementation on a reversed, hand-specified
illustrative topology graph (Section 6.6), not fit to SMD (SMD has no
topology).

## 5.5 Metrics

Prediction: AUROC, AUPRC on the held-out EVAL block. Attribution:
precision@k and recall@k of the top-k SHAP-ranked KPI dimensions against
the ground-truth interpretation_label set, compared to the chance level
k/38. Causal filtering: Granger SSR F-test p-value (minimum across lags
1-5) and Pearson correlation, reported side by side. Calibration:
empirical coverage of the conformal prediction set against the 90%
target, and Expected Calibration Error (ECE, 10 equal-width bins) of the
raw probability. Early-warning usefulness: lead time, the gap in minutes
between the first sustained (\>= 2 consecutive minutes) crossing of a
0.5 risk threshold and the official onset of each ground-truth segment.

# 6. Results

## 6.1 Component 1 \-- Degradation prediction

Table 3 and Figure 2 compare the two predictors on the held-out EVAL
block. The Transformer outperforms the LightGBM baseline on both
metrics, consistent with the attention-over-recurrence/trees rationale
in Section 2, while the baseline is retained because it is the model
exactly explained by SHAP in Section 6.2.

**Table 3. Component 1 results on the held-out EVAL block.**

  -----------------------------------------------------------------------
  **Model**               **AUROC**               **AUPRC**
  ----------------------- ----------------------- -----------------------
  LightGBM (baseline)     0.898                   0.825

  Transformer (primary)   0.934                   0.884
  -----------------------------------------------------------------------

![](./media/image2.png){width="4.3in" height="3.4224486001749783in"}

*Figure 2. ROC curves for both predictors on the held-out EVAL block.*

## 6.2 Component 2 \-- Root-cause attribution

For every ground-truth segment, we take the LightGBM TreeExplainer
attribution at the last window before onset, aggregate \|SHAP\| across
each KPI dimension\'s mean/std/last features, and rank dimensions. Table
4 reports precision@k = recall@k (k set to the ground-truth set size)
per segment; segments in the EVAL and CALIB blocks are genuinely
out-of-sample for the model, while TRAIN-block segments are in-sample
and shown only for contrast.

**Table 4. Component 2 results: SHAP-ranked root-cause attribution vs.
ground truth.**

  -----------------------------------------------------------------------
  **Segment**       **Block**         **k**             **Precision@k =
                                                        Recall@k**
  ----------------- ----------------- ----------------- -----------------
  15849-16368       train (in-sample) 7                 0.571

  16963-17517       train (in-sample) 31                0.968

  18071-18528       train (in-sample) 8                 0.500

  19367-20088       eval              14                0.571
                    (out-of-sample)                     

  20786-21195       eval              7                 0.571
                    (out-of-sample)                     

  24679-24682       calib             4                 0.500
                    (out-of-sample)                     

  26114-26116       calib             4                 0.250
                    (out-of-sample)                     

  27554-27556       calib             4                 0.750
                    (out-of-sample)                     
  -----------------------------------------------------------------------

The out-of-sample (EVAL + CALIB) mean precision/recall@k is 0.529,
against a mean chance level of approximately 0.174 for the corresponding
k values \-- well above chance, but far from perfect, which we report as
the honest finding rather than only the best-case segment.

## 6.3 Component 3 \-- Correlation vs. causal evidence

For the representative segment 20786-21195, Table 5 tests the top-5
SHAP-ranked KPI dimensions for Granger causality against a fixed target
dimension (KPI 14, present in every ground-truth set) over a local
pre/during-onset window.

**Table 5. Component 3 results: Granger-causality filtering of
SHAP-ranked candidates.**

  ---------------------------------------------------------------------------
  **KPI dim.**   **In ground    **Pearson r**  **Granger p    **Verdict**
                 truth**                       (min)**        
  -------------- -------------- -------------- -------------- ---------------
  7              No             -0.075         0.199          Correlational
                                                              only

  9              Yes            0.951          0.000          Causal evidence

  13             Yes            0.586          0.025          Causal evidence

  30             No             0.046          0.450          Correlational
                                                              only

  33             No             0.026          0.791          Correlational
                                                              only
  ---------------------------------------------------------------------------

The two dimensions that are both highly correlated and in the
ground-truth root-cause set (9, 13) pass the Granger test; the three
that are only SHAP-flagged noise (weak correlation, absent from ground
truth) fail it \-- the intended separation, though demonstrated here on
one incident rather than across a statistically powered sample (Section
8).

## 6.4 Component 5 \-- Uncertainty quantification

Split-conformal calibration on the CALIB block yields q_hat = 0.985 and
achieves 0.955 empirical coverage on EVAL against the 90% target \-- a
valid, if conservative, guarantee. 20.0% of EVAL windows are routed to
the ambiguous ({normal, degrading}) set, i.e. an explicit
escalate-to-human signal, and 0.0% to the empty set. The raw
probability\'s ECE on EVAL is 0.118: Figure 3 shows the model is
overconfident in its top bin (mean predicted probability 0.986,
empirical positive rate 0.752), exactly the failure mode conformal
wrapping is designed to survive without requiring the underlying model
to be well-calibrated.

![](./media/image3.png){width="4.3in" height="3.4224486001749783in"}

*Figure 3. Reliability diagram of the raw Transformer probability on
EVAL (ECE = 0.118).*

## 6.5 Early-warning lead time

Table 6 reports, per ground-truth segment, the gap between the first
sustained risk-score crossing of 0.5 and official onset. Out-of-sample
(EVAL + CALIB), the mean lead time is 12.5 minutes, with 4 of 5
incidents caught and one missed entirely.

**Table 6. Component 1 out-of-sample early-warning lead time.**

  -----------------------------------------------------------------------
  **Segment**             **Block**               **Lead time (min)**
  ----------------------- ----------------------- -----------------------
  19367-20088             eval                    44

  20786-21195             eval                    missed

  24679-24682             calib                   2

  26114-26116             calib                   2

  27554-27556             calib                   2
  -----------------------------------------------------------------------

![](./media/image4.png){width="5.6in" height="3.1652176290463694in"}

*Figure 4. Risk score vs. ground truth for segment 19367-20088, the
incident with the longest measured lead time.*

Lead time is highly heterogeneous rather than uniform: one incident is
flagged 44 minutes early with a clean ramp-up, three tiny CALIB-block
incidents are caught only \~2 minutes early (essentially at onset), and
one incident\'s risk score jumps from \~0.001 to \~0.98 in a single step
with no gradual precursor within the model\'s 20-minute lookback and is
missed by the lead-time rule entirely. We report this variance
explicitly rather than only the mean, since averaging it away would
misrepresent the system\'s actual, uneven early-warning capability \--
itself a trustworthiness finding, not only a performance number.

## 6.6 Component 2, graph view \-- illustrative topology-based ranking

SMD provides no network-element topology, so this experiment is
explicitly illustrative: a synthetic Core -\> Aggregation -\>
BaseStation -\> Cell graph with an injected root cause at Aggregation
and severity decaying (with noise) along descendants. Personalized
PageRank on the reversed graph, seeded by observed node severity, ranks
Aggregation first (score 0.319), correctly recovering the injected root
cause and demonstrating the topology-propagation mechanism that a
learned graph neural network (e.g. GDN \[14\], or \[12\] for telecom RAN
specifically) would generalize with data-driven edge weights.

# 7. Discussion

Taken together, the results in Section 6 show that each layer of the
proposed framework produces measurable, distinguishable evidence rather
than a single opaque score: the predictor\'s discrimination (Section
6.1) is separable from the attribution stage\'s correctness against
ground truth (6.2), which is separable again from the causal filter\'s
ability to prune spurious candidates (6.3), from the calibration
layer\'s coverage guarantee (6.4), and from the system\'s actual, uneven
early-warning behavior (6.5). This separability is the practical value
of decomposing trustworthiness into components: a 0.934 AUROC prediction
score alone would not have revealed that root-cause precision plateaus
at 0.529, that one incident type has no learnable precursor at all, or
that 20% of predictions should be escalated rather than acted on
automatically. We view the honest reporting of imperfect and
heterogeneous results \-- rather than only the best-case segment or a
pooled average \-- as itself part of the trustworthiness contribution,
consistent with the framework\'s design goal of surfacing uncertainty
rather than hiding it.

At the same time, these results are a single-dataset demonstration that
a joint pipeline is measurable and instrumentable, not evidence that it
outperforms simpler alternatives. Section 8 states this limitation
directly, and Section 9 defines the multi-dataset, multi-baseline,
statistically validated study required to make a generalizable claim.

# 8. Limitations and Threats to Validity

Single dataset, single machine. All results are for one machine
(machine-1-1) of one public dataset. Generalization across machines,
datasets, and fault types is untested here.

Domain proxy, not telecom data. SMD\'s 38 dimensions are generic server
metrics (CPU, memory, network, load), not named telecom KPIs (latency,
jitter, handover-failure rate). The methods transfer directly; the
specific numerical results reported here do not.

No external baselines. We compare LightGBM against a Transformer
internally, but do not compare the full pipeline against established RCA
methods such as MicroRCA \[13\], CausalRCA, RCD, or epsilon-Diagnosis
\-- required before any comparative claim can be made.

No statistical validation. Results are reported from a single run/seed
on a single chronological split; no significance testing, confidence
intervals, or effect sizes are computed. With only eight ground-truth
segments in this machine\'s timeline, statistical power for
segment-level claims is inherently limited.

Granger causality is not proof of causation. The causal-filter verdicts
in Section 6.3 reflect temporal-precedence evidence from an
observational test, not a true intervention; SMD provides no
configuration-change log to serve as a real quasi-experiment.

Illustrative, not fitted, topology experiment. The Section 6.6
graph-propagation result uses a hand-specified synthetic topology and
injected severities; it demonstrates the mechanism, not a validated
capability on real network topology.

No production or robustness evaluation. No deployed system exists to
report latency/throughput SLAs against; no adversarial-input,
alarm-storm, or missing-telemetry robustness tests were run.

# 9. Practical Implications and Future Work

For practitioners, three design choices in this framework are directly
actionable regardless of the specific models used: (i) treat prediction,
attribution, causal filtering, and calibration as separately monitored
pipeline stages so a regression in one is not masked by another; (ii)
report lead time and root-cause accuracy per incident rather than only
pooled, since \-- as Section 6.5 shows \-- the pooled average can
misrepresent genuinely bimodal behavior; and (iii) use a conformal (or
similarly calibrated) abstention signal to route low-confidence cases to
a human rather than resolving them silently.

The immediate next step for this research programme, detailed further in
an accompanying feasibility analysis, is a multi-dataset ablation study:
re-running this pipeline across SMD (all 28 machines), PSM, SWaT,
RCAEval \[5\], and LEMMA-RCA \[6\]; comparing against MicroRCA,
CausalRCA, RCD, and epsilon-Diagnosis as external baselines; and
testing, with paired significance tests across multiple seeds, whether
the causal-filtering and conformal-calibration layers measurably improve
root-cause precision and selective-prediction risk over attribution
alone \-- the joint ablation this paper\'s related-work review found no
prior study to have reported. A second line of future work is a
controlled human-subjects study of whether exposing the
causal-versus-correlational distinction and the abstention flag changes
engineer diagnostic accuracy and calibrated trust, a gap independently
identified by a systematic review of XAI human-evaluation methodology
\[21\]. A third, higher-risk line, contingent on data access, is
validation on real RAN-level KPI telemetry against the published
GNN+Transformer RAN RCA system \[12\].

# 10. Conclusion

We proposed a five-component framework \-- predict, attribute, causally
filter, calibrate, explain \-- for trustworthy root-cause analysis of
telecom KPI degradation, and reported a reproducible proof-of-concept
evaluation on the public SMD dataset, chosen specifically for its
ground-truth root-cause labels. A Transformer predictor reached 0.934
AUROC against a 0.898 LightGBM baseline; out-of-sample root-cause
attribution reached 0.529 mean precision/recall@k against a \~0.174
chance level; Granger-causality filtering correctly separated
ground-truth-consistent from spurious SHAP candidates in the incident
examined; split-conformal calibration achieved its 90% coverage target
while flagging 20% of cases for human escalation; and early-warning lead
time was shown to be highly heterogeneous, from 44 minutes to a complete
miss, across incidents. These results establish that the proposed
decomposition is measurable and instrumentable against ground truth on a
public benchmark. They do not establish that it outperforms simpler
alternatives, generalizes beyond one machine, or holds on real telecom
KPI data \-- claims that require the multi-dataset, multi-baseline,
statistically validated study defined in Section 9 and that we intend as
the next stage of this work.

# References

\[1\] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting
model predictions,\" in Proc. NeurIPS, 2017.

\[2\] C. W. J. Granger, \"Investigating causal relations by econometric
models and cross-spectral methods,\" Econometrica, vol. 37, no. 3, pp.
424-438, 1969.

\[3\] P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction,
and Search, 2nd ed. Cambridge, MA, USA: MIT Press, 2000.

\[4\] A. N. Angelopoulos and S. Bates, \"A gentle introduction to
conformal prediction and distribution-free uncertainty quantification,\"
arXiv:2107.07511, 2021.

\[5\] L. Pham, H. Zhang, H. Ha, F. Salim, and X. Zhang, \"RCAEval: A
benchmark for root cause analysis of microservice systems with telemetry
data,\" arXiv:2412.17015, 2024.

\[6\] L. Zheng, Z. Chen, D. Wang, C. Deng, R. Matsuoka, and H. Chen,
\"LEMMA-RCA: A large multi-modal multi-domain dataset for root cause
analysis,\" arXiv:2406.05375, 2024.

\[7\] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, \"Robust
anomaly detection for multivariate time series via stochastic recurrent
neural network,\" in Proc. ACM SIGKDD, 2019.

\[8\] P. Schummer, A. del Rio Ponce, J. Serrano Romero, D. Jimenez
Bermejo, G. Sanchez, and A. Llorente, \"Machine learning-based network
anomaly detection: Design, implementation, and evaluation,\" AI (MDPI),
vol. 5, no. 4, art. 143, 2024.

\[9\] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T.
Soderstrom, \"Detecting spacecraft anomalies using LSTMs and
nonparametric dynamic thresholding,\" in Proc. ACM SIGKDD, 2018.

\[10\] Q. Wen et al., \"Transformers in time series: A survey,\"
arXiv:2202.07125, 2022.

\[11\] S. Tuli, G. Casale, and N. R. Jennings, \"TranAD: Deep
transformer networks for anomaly detection in multivariate time series
data,\" Proc. VLDB Endowment, vol. 15, no. 6, 2022.

\[12\] A. Hasan, C. Boeira, K. Papry, Y. Ju, Z. Zhu, and I. Haque,
\"Root cause analysis of anomalies in 5G RAN using graph neural network
and transformer,\" arXiv:2406.15638, 2024.

\[13\] L. Wu, J. Tordsson, E. Elmroth, and O. Kao, \"MicroRCA: Root
cause localization of performance issues in microservices,\" in Proc.
IEEE/IFIP NOMS, 2020.

\[14\] A. Deng and B. Hooi, \"Graph neural network-based anomaly
detection in multivariate time series,\" in Proc. AAAI, 2021.

\[15\] C.-M. Lin, C. Chang, W.-Y. Wang, K.-D. Wang, and W.-C. Peng,
\"Root cause analysis in microservice using neural Granger causal
discovery,\" in Proc. AAAI, 2024. arXiv:2402.01140.

\[16\] L. Pham, H. Ha, and H. Zhang, \"Root cause analysis for
microservice system based on causal inference: How far are we?,\"
arXiv:2408.13729, 2024.

\[17\] A. Shan, J. Kaur, R. Singh, T. Banka, R. Yavatkar, and T.
Sridhar, \"RCA Copilot: Transforming network data into actionable
insights via large language models,\" arXiv:2507.03224, 2025.

\[18\] D. Roy et al., \"Exploring LLM-based agents for root cause
analysis,\" arXiv:2403.04123, 2024.

\[19\] A. Bellotti and X. Zhao, \"Conformal prediction and trustworthy
AI,\" arXiv:2508.06885, 2025.

\[20\] F. O. Catak, U. Cali, M. Kuzlu, and S. Sarp, \"Uncertainty aware
deep learning model for secure and trustworthy channel estimation in 5G
networks,\" arXiv:2305.02741, 2023.

\[21\] J. Kim, H. Maathuis, and D. Sent, \"Human-centered evaluation of
explainable AI applications: A systematic review,\" Frontiers in
Artificial Intelligence, 2024.
