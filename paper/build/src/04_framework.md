# 4. Proposed Framework

TrustNet-RCA decomposes the problem in Section 1 into five components, each
with its own input, output, and validation criterion, chained as shown in
Figure 1. This decomposition is deliberate: it lets each stage be evaluated
(and, in Section 6, scored against ground truth) independently, rather than
treating trustworthiness as an emergent, unmeasured property of one opaque
end-to-end model.

![](figures/fig1_framework.png){width="7.0in" height="4.1in"}

*Figure 1. The TrustNet-RCA five-component framework and its three-leg evaluation architecture. Solid arrows show data
flow; Components 2 (graph view) and 5 are illustrative/parallel branches
feeding the final explanation stage.*

## 4.1 Component 1: Predict degradation before disruption

Input: a W-minute KPI window per network element. Method: a Transformer
encoder (self-attention over the window) as the primary model, benchmarked
against a gradient-boosted tree baseline (LightGBM) on rolling statistical
features, retained as a cheap, auditable fallback and as the model exactly
explained in Component 2. Output: P(degradation in the next H minutes), a
continuously updated early-warning score rather than a post-hoc flag.

## 4.2 Component 2: Identify probable root causes

Input: the flagged window, the full KPI vector, and (where available) the
NE/site topology graph. Method: exact SHAP attribution on the tree-ensemble
predictor ranks which KPI dimensions drove the flagged prediction; a graph
view (personalized PageRank over the topology, seeded at high-deviation
nodes) ranks which network element is the propagation source rather than
merely a symptom, in the spirit of MicroRCA \[13\]. Output: a ranked (KPI or
NE, contribution score) list, explicitly still correlational at this stage.

## 4.3 Component 3: Distinguish correlation from causal evidence

Input: Component 2\'s ranked candidates and their raw time series.
Method: a Granger-causality test per candidate (Section 3), plus, where
a configuration-change timestamp coincides with a candidate\'s onset, a
before/after comparison treated as a quasi-experiment. Output: each
candidate labeled correlational only or causal evidence, with its test
statistic attached; Granger causality is necessary-but-not-sufficient
evidence for true causality, and this label is reported as a confidence
tier, not a proof.

## 4.4 Component 4: Explain in engineer-understandable terms

Input: the outputs of Components 1, 2, 3, and 5. Method: a template grounded
strictly in the evidence already computed (not a free-generation LLM call),
following the narrate-don\'t-diagnose pattern from RCACopilot- and
RCAgent-style systems \[17\], \[18\]; an LLM may phrase the narrative fluently but
may not assert any fact absent from the evidence. Output: a short report
stating what is predicted, how confident, which KPI/NE is implicated,
whether that evidence is causal or correlational, and what changed
immediately before onset.

## 4.5 Component 5: Quantify uncertainty and confidence

Input: Component 1\'s raw probability and a held-out calibration set.
Method: split-conformal prediction (Section 3) wraps the probability
into a set with a distribution-free, finite-sample coverage guarantee,
independent of whether the underlying model is well-calibrated on its
own; reliability diagrams / Expected Calibration Error (ECE) are
reported as a diagnostic. Output: a calibrated confidence level plus an
explicit abstain-and-escalate flag for low-confidence (ambiguous)
predictions.
