# Research Paper Feasibility Analysis
### Subject: `Telecom_Trustworthy_AI_Framework.ipynb` (the Question 1 trustworthy-prediction / root-cause notebook)
### Analysis date: 2026-09-01

> **Framing note, read before the rest.** The "existing work" analyzed here is a proof-of-concept
> notebook built to answer an interview/assessment question, not a deployed production system. It
> demonstrates a *design* (5-component framework) on **one machine** from a public, already
> extensively-benchmarked dataset (NetManAIOps SMD), using **standard, off-the-shelf methods**:
> vanilla `LGBMClassifier`, a stock 2-layer `nn.TransformerEncoder`, unmodified SHAP `TreeExplainer`,
> unmodified `statsmodels` Granger causality, unmodified `networkx.pagerank`. No new algorithm, loss
> function, architecture, or dataset was created. This governs the whole analysis below — several
> phases in your template (production hardware, proprietary-data ethics, live user load) don't apply
> yet because no such deployment exists.

---

## Phase 1 — System Reconstruction

**Problem → Inputs → Method → System → Output → Evaluation**

```
Problem:  Predict telecom service degradation early, identify its probable root cause,
          separate correlation from causal evidence, explain it to an engineer, and
          quantify confidence — jointly, not as isolated tasks.

Input:    Multivariate KPI time series (38-dim server-metric proxy for real telecom
          KPIs: latency, packet loss, jitter, CPU/mem, etc.), windowed (W=20 min).

Method:   [LightGBM | Transformer] → probability  ─┐
          SHAP TreeExplainer → per-KPI attribution ─┼─► chained, not learned jointly
          Granger causality → causal/correlational label
          Split-conformal (LAC) → calibrated prediction set
          Personalized PageRank → topology-based root-cause rank (illustrative only)

System:   Sequential pipeline; each stage consumes the previous stage's output.

Output:   P(degradation), ranked root-cause KPI dims, causal verdict, calibrated
          prediction set, templated natural-language explanation.

Evaluation: AUROC/AUPRC (prediction), precision@k/recall@k vs. ground-truth root-cause
          labels (attribution), empirical coverage + ECE (uncertainty), lead time
          (early-warning usefulness) — all on ONE held-out block of ONE machine.
```

1. **Problem solved:** joint prediction + diagnosis + causal filtering + explanation + calibrated confidence for KPI degradation.
2. **Inputs:** 38-dim KPI vector per minute (SMD stand-in for real telecom KPIs).
3. **Outputs:** probability, ranked root causes, causal/correlational labels, prediction set, text explanation.
4. **Components:** LightGBM, Transformer, SHAP, Granger test, split-conformal LAC, PageRank.
5. **Standard engineering:** data download/parsing, windowing/feature engineering, training loops, plotting, the LightGBM/Transformer/SHAP/Granger/PageRank calls themselves (all library defaults).
6. **Potential novelty region:** *not* any single component — the **chaining and joint validation** of attribution → causal filtering → calibration as one pipeline, validated against ground-truth root-cause labels (most papers validate one stage at a time).
7. **Assumptions:** SMD generic server metrics transfer to telecom RAN KPIs; one machine's incident distribution generalizes; Granger causality (lags 1–5 min) is an adequate causal proxy without real interventions.
8. **Constraints:** no topology data, no configuration-change log (no real quasi-experiments), single incident family per machine, single dataset.
9. **Real-world motivation:** operators need root-cause explanations engineers can act on and trust enough to *not* blindly automate remediation on — not just an accuracy number.
10. **Bottlenecks observed:** root-cause precision@k plateaus ~0.53 (far from 1.0); lead time is highly heterogeneous (one incident type had **zero** usable precursor signal); no real interventional causal evidence available in public data.

---

## Phase 2 — Engineering vs. Scientific Contribution

### Engineering Contributions (present, solid, not publishable alone)
- Data loading/windowing/leakage-safe chronological split.
- Training LightGBM and a small Transformer to predict an early-warning label.
- Wiring SHAP, Granger causality, split-conformal, and PageRank onto the trained models' outputs.
- Producing a template-based natural-language report and diagnostic plots.

None of this is a research contribution by itself — it is correct, careful application of existing tools to one dataset.

### Potential Scientific Contributions — strict assessment
| Candidate | Verdict |
|---|---|
| New algorithm / architecture | **No.** `TinyTransformer` is a vanilla encoder; no architectural novelty. |
| New loss function | **No.** Standard weighted BCE. |
| New causal-discovery method | **No.** Off-the-shelf Granger test. |
| New conformal method | **No.** Textbook split-conformal LAC. |
| New dataset/benchmark | **No.** SMD is public and already used in ≥15 papers (OmniAnomaly, USAD, TranAD, GDN, Anomaly Transformer, …). |
| New evaluation methodology (jointly scoring attribution+causal+calibration against ground truth, as one pipeline) | **Plausible, not yet demonstrated at scale** — done here on 1 machine only. |
| Empirically demonstrated phenomenon (causal filtering measurably prunes false SHAP positives; conformal wrapping fixes measurable over-confidence) | **Plausible, shown qualitatively on one segment/one block — not statistically validated.** |

**Explicit statement, as required:** *In its current form, this project contains no publishable Q1 novelty.* It is a well-executed engineering demonstration. The only defensible path to novelty is **methodological/empirical**, not architectural: rigorously showing, across multiple datasets/baselines/seeds, that jointly chaining attribution → causal filtering → conformal calibration changes measurable outcomes (false-positive root causes, selective-prediction risk, calibration error) versus using any one stage alone — which nobody currently reports as a joint ablation (see Phase 3).

---

## Phase 3 — Literature Gap Discovery (live-searched, not fabricated)

Searches run: joint conformal+causal RCA frameworks; RCA benchmark/survey comparisons; human-trust evaluation of XAI for RCA.

| # | Theme | What researchers do | What's missing | How this project could address it |
|---|---|---|---|---|
| 1 | Causal-discovery RCA (Granger/PC/PAG) | Build causal graphs from metrics (e.g. arXiv:2411.06990, 2606.20912) | Rarely paired with a *calibrated confidence* layer | Add conformal wrapping on top of causal-filtered candidates |
| 2 | RCA benchmarking (RCAEval, arXiv:2412.17015; LEMMA-RCA, arXiv:2406.05375) | Standardize PR@K/MRR/MAP@K across microservice/IT-OT domains | No telecom-RAN domain; no benchmark scores attribution+causal+calibration jointly | Extend the joint-pipeline ablation onto these existing public benchmarks instead of inventing a new one |
| 3 | LLM-based AIOps RCA (survey arXiv:2507.12472) | Use LLMs to retrieve/narrate root causes | Confidence calibration of LLM-proposed causes is largely unaddressed | Position the conformal/causal layers as a *grounding* mechanism an LLM narrator must respect |
| 4 | Human-centered XAI evaluation (PMC11525002 systematic review) | Propose explanation UIs | Review explicitly finds **no standardized evaluation framework** (19/73 papers share one) | A controlled study of engineer trust/accuracy under causal-labeled vs. correlation-only explanations would target a documented gap |
| 5 | Transformer/GNN telecom RCA (arXiv:2406.15638) | Combine GNN+Transformer for 5G RAN anomaly RCA | Real topology/RAN data, no causal-vs-correlation distinction, no calibrated uncertainty | Closest published analogue — the differentiator would be adding the causal+conformal layers and validating against its results |

**Candidate gap statements (ranked):**

1. **(High novelty)** *"Existing RCA benchmarks (RCAEval, LEMMA-RCA) evaluate attribution methods in isolation and report point accuracy; they do not measure whether adding a causal-precedence filter and a conformal confidence layer changes false-positive root-cause rates or selective-prediction risk. This work provides a controlled, multi-dataset ablation quantifying that."* — feasible, public data only.
2. **(High novelty, higher risk)** *"XAI-for-RCA human evaluation lacks a standardized protocol (per PMC11525002); no study isolates whether an explicit causal-vs-correlational label plus an abstain-on-uncertainty flag changes engineer diagnostic accuracy or miscalibrated trust."* — needs a human-subjects study.
3. **(Moderate)** *"Published RAN-specific RCA work (arXiv:2406.15638) predicts and localizes but does not report calibrated uncertainty or a correlation/causation distinction; this work would add both and re-benchmark against it if RAN data or its released artifacts are accessible."* — gated on data/artifact access, unverified.
4. **(Moderate)** *"Attention-based attribution is contested (attention ≠ explanation); a causally-regularized attention objective — penalizing attention mass on Granger-non-causal features during training — has not been benchmarked against post-hoc SHAP for RCA."* — real methodological angle, but enters contested territory.
5. **(Incremental)** *"A framework paper proposing the 5-component trustworthy pipeline as a reference architecture for telecom AIOps."* — position/vision paper, weak alone, only useful as an intro/motivation section for #1.

I found **no existing paper that jointly combines causal discovery + conformal prediction + attribution as one evaluated pipeline** — stated as "not found in this search," not as a certainty; a full systematic search before committing is still required (Phase 26, Stage 2).

---

## Phase 4 — Candidate Research Directions

### Direction 1 — Joint Trustworthy-RCA Pipeline: A Multi-Dataset Ablation Study ⭐ (recommended, see Phase 5)
**Research problem:** Does chaining attribution → causal filtering → conformal calibration measurably outperform any single stage alone, and by how much, across fault types and datasets?
**Research gap:** RCA benchmarks (RCAEval, LEMMA-RCA) score attribution methods in isolation; no joint ablation of attribution+causal+calibration exists.
**Contribution:** A reproducible, multi-dataset (SMD, PSM, SWaT, RCAEval, LEMMA-RCA) empirical study + ablation protocol, released as open evaluation code.
**Novelty:** Not a new algorithm — a new **systematic evaluation** of a combination nobody has jointly ablated.
**RQs:** RQ1 (effectiveness): does the full pipeline beat single-stage baselines on precision@k/recall@k? RQ2 (SOTA comparison): how does it compare to MicroRCA/CausalRCA/RCD/ε-Diagnosis? RQ3 (efficiency): what latency overhead do the causal+conformal stages add? RQ4 (generalization): does the effect hold across all 5 datasets and 11+ fault types? RQ5 (practical): does it reduce false-positive root-cause alerts at fixed recall?
**Hypotheses:** H1 — causal filtering significantly reduces false-positive root-cause candidates vs. attribution alone (paired test across fault cases). H2 — conformal calibration significantly improves the risk-coverage trade-off vs. a fixed-threshold rule.
**Experiments:** cross-dataset benchmark; 4-way ablation (full / −causal / −conformal / attribution-only); baseline comparison; ≥5 seeds.
**Dataset:** public only — SMD (all 28 machines, not 1), PSM, SWaT, RCAEval, LEMMA-RCA.
**Baselines:** MicroRCA, CausalRCA, RCD, ε-Diagnosis, SHAP-only, Granger-only.
**Metrics:** Precision@k, Recall@k, MRR, MAP@k, AUROC/AUPRC, ECE, conformal coverage, false-alarm rate, latency.
**Ablations:** remove causal filter; remove conformal layer; swap Transformer↔GBM predictor.
**Stats:** paired Wilcoxon signed-rank across fault cases, bootstrap CIs, Cohen's d effect size, ≥5 seeds.
**Publication potential:** 7/10. **Difficulty:** 7/10. **Novelty risk:** Medium. **Journal area:** network/systems management, applied AI.

### Direction 2 — Human-Trust Study of Causal-Labeled, Uncertainty-Aware Explanations
**Research problem:** Does labeling candidates as "causal evidence" vs. "correlational only," plus an explicit abstain flag, change engineer diagnostic accuracy and calibrated trust vs. a bare attribution ranking?
**Research gap:** Systematic review (PMC11525002) finds no standardized XAI-for-RCA human-evaluation protocol.
**Contribution:** A controlled within/between-subject study + a reusable evaluation protocol for this specific niche.
**Novelty:** methodological (HCI) contribution, not a model contribution.
**RQs:** RQ1 effectiveness (accuracy with vs. without causal labeling); RQ2 comparison (vs. attribution-only baseline UI); RQ4 robustness (does it hold across incident difficulty); RQ5 practical (does trust stay calibrated, i.e. not over-trusted, on wrong predictions).
**Hypotheses:** H1 — causal-tier labels reduce acceptance of false root causes. H2 — abstention flag reduces over-trust on wrong high-confidence predictions.
**Experiments:** vignette study using incidents from RCAEval/SMD, N engineers, pre/post trust survey (e.g. Trust-in-Automation scale), accuracy + time-to-decision measured.
**Dataset:** new — human-response data collected from participants (requires recruitment/ethics approval).
**Baselines:** explanation-format is the manipulated variable (raw score / attribution-only / +causal / full).
**Metrics:** diagnostic accuracy, time-to-decision, trust-calibration gap, SUS score.
**Publication potential:** 6.5/10 (genuine gap, but methodology-heavy and small-N risk). **Difficulty:** 8/10. **Risk:** Medium-High.

### Direction 3 — Real Telecom-KPI Validation (RAN data)
**Research problem:** Does the pipeline hold on real RAN KPIs (latency/jitter/handover failure) instead of the generic SMD proxy, compared directly against the published GNN+Transformer RAN-RCA system (arXiv:2406.15638)?
**Research gap:** existing RAN RCA work has no causal/correlational distinction or calibrated uncertainty.
**Contribution:** first (or one of few) validated trustworthy-RCA pipeline on real RAN data with direct SOTA comparison.
**Novelty:** high **if** real/semi-real data access exists; otherwise this collapses to "same demo, different dataset."
**Blocker:** RAN KPI datasets with root-cause labels are operator-confidential — this direction is **gated on data access you do not currently have**, and I am not assuming you do.
**Publication potential:** 8/10 conditional on data access; **effectively 3/10 unconditional. Difficulty:** 9/10. **Risk:** High (external dependency).

### Direction 4 — Causally-Regularized Attention as a Learned Attribution Mechanism
**Research problem:** Can a training-time penalty on attention mass assigned to Granger-non-causal KPIs produce better root-cause attribution than post-hoc SHAP, in one model instead of two?
**Research gap:** attention-as-explanation is contested; no work regularizes attention using causal-discovery signal specifically for RCA.
**Contribution:** a new loss term (methodological novelty, if it works).
**Novelty:** moderate — real algorithmic contribution, but enters a literature that has actively pushed back on "attention is explanation" claims, so the bar for justification is high.
**Publication potential:** 5/10 as scoped; up to 7/10 if the causal-regularization idea is developed rigorously with its own ablations. **Difficulty:** 6/10. **Risk:** Medium-High.

### Direction 5 — Reference-Architecture / Position Paper
**Research problem:** Propose the 5-component pipeline as a reference architecture for trustworthy telecom AIOps.
**Contribution:** conceptual, not empirical.
**Novelty:** low, standalone. **Publication potential:** 3–4/10 alone; useful only as framing/§2 of Direction 1, not as its own paper for a strong Q1 venue.

---

## Phase 5 — Direction Comparison & Selection

| Direction | Novelty | Research Value | Experiment Feasibility | Dataset Availability | Q1 Potential | Difficulty | Risk |
|---|---|---|---|---|---|---|---|
| D1 Joint pipeline ablation | Medium-High | High | High | High (all public) | 7/10 | 7/10 | Medium |
| D2 Human-trust study | High (gap-confirmed) | High | Medium | Low (must collect) | 6.5/10 | 8/10 | Med-High |
| D3 Real RAN validation | High (conditional) | High | Low (blocked) | Very Low | 8/10* / 3/10 | 9/10 | High |
| D4 Causal-regularized attention | Medium | Medium | Medium | High | 5–7/10 | 6/10 | Med-High |
| D5 Position paper | Low | Low | High | N/A | 3–4/10 | 2/10 | Low (but low reward) |

**BEST RESEARCH DIRECTION: D1 — Joint Trustworthy-RCA Pipeline, Multi-Dataset Ablation Study.**

Why: it is the only direction that is simultaneously (a) buildable directly from the existing pipeline by *generalizing* it rather than replacing it, (b) fully supported by public data with no external dependency, (c) statistically validatable with standard significance testing, and (d) targets a gap independently confirmed by literature search (Phase 3, item 1). D3 would be stronger *if* real RAN data existed, but is not currently within your control; D2 is a strong, legitimate follow-on paper but requires a different skillset and ethics process — recommended as **Paper 2**, not Paper 1.

---

## Phase 6 — Transforming the System into a Research Framework

```
Existing System
   (5-stage pipeline, 1 machine, 1 dataset, no baselines, no stats)
        ↓
Identified Limitation
   (no evidence the joint chaining beats single-stage baselines; unvalidated generalization)
        ↓
Scientific Hypothesis
   (H1/H2 above: causal filtering cuts false positives; conformal calibration
    improves risk-coverage trade-off — both significantly, across datasets)
        ↓
Proposed Method
   (same pipeline, now dataset-agnostic; add ablation switches to disable
    causal/conformal/predictor-choice independently)
        ↓
Experimental Setup
   (5 public datasets, 6 baselines, ≥5 seeds, held-out splits per dataset)
        ↓
Baselines
   (MicroRCA, CausalRCA, RCD, ε-Diagnosis, SHAP-only, Granger-only)
        ↓
Results  [not yet obtained — placeholders in Phase 18]
        ↓
Statistical Validation
   (paired Wilcoxon, bootstrap CI, effect size, per-dataset and pooled)
        ↓
Scientific Contribution
   (quantified evidence for/against joint chaining, released as an
    open evaluation protocol other researchers can reuse)
```

**Required additions, categorized:**
- **CRITICAL:** re-run on all 5 datasets (not 1 machine); implement the 6 baselines; add ablation switches; add statistical testing across seeds/fault cases; add a real SOTA comparison table.
- **IMPORTANT:** extend SMD evaluation to all 28 machines, not just `machine-1-1`; report per-fault-type breakdown, not pooled-only; release code/protocol publicly for reproducibility.
- **OPTIONAL:** add the LLM-narration layer as a qualitative case study; add Direction 4's causally-regularized attention as a secondary ablation arm; add a small human-rater sanity check on explanation quality (light version of D2).

---

## Phase 7 — Core Novel Contribution

**"The primary novelty of this research is a systematic, multi-dataset empirical demonstration of whether — and under what conditions — chaining feature attribution, causal-precedence filtering, and conformal calibration into one root-cause pipeline measurably reduces false-positive root-cause attributions and improves calibrated confidence, compared to using any of these mechanisms alone, an ablation nobody currently reports."**

Main contributions:
1. We propose an ablation-driven evaluation protocol for jointly scoring attribution+causal+calibration RCA pipelines against ground-truth root-cause labels across heterogeneous public benchmarks.
2. We introduce a false-positive-rate and risk-coverage analysis of causal filtering and conformal calibration as add-on layers to existing attribution methods.
3. We demonstrate experimentally which of these layers contributes measurable value, and which fault types/datasets the effect does or does not generalize to.

(Explicitly avoided: "we implemented...", "we integrated...", "we built an application..." — none of those are load-bearing claims here.)

---

## Phase 8 — Research Questions (final, for D1)

- **RQ1 (effectiveness):** Does the full attribution+causal+conformal pipeline achieve significantly higher root-cause precision@k / recall@k than any single stage alone?
- **RQ2 (SOTA comparison):** How does it compare to MicroRCA, CausalRCA, RCD, and ε-Diagnosis on the same benchmarks?
- **RQ3 (efficiency/scalability):** What is the added latency/compute cost of the causal and conformal stages relative to attribution alone, and does it scale to larger metric counts?
- **RQ4 (robustness/generalization):** Does the effect hold consistently across datasets (SMD, PSM, SWaT, RCAEval, LEMMA-RCA) and fault categories (resource, network, code-level)?
- **RQ5 (practical implications):** At a fixed recall target, how much does the pipeline reduce false-positive root-cause alerts an engineer would have to triage?

---

## Phase 9 — Experimental Design

**Hardware:** CPU: any modern multi-core (models are small — LightGBM/Transformer here trained in ~80s on CPU). GPU: optional, useful only to speed up re-runs across 5 datasets × 6 baselines × 5 seeds. RAM: ≥16GB for the larger datasets (SWaT/RCAEval). OS: any (Linux used in this prototype).

**Software:** Python; `lightgbm`, `torch`, `shap`, `statsmodels`, `networkx`, `scikit-learn`; baseline implementations from the RCAEval/original-paper repos where available (do not reimplement from scratch if authors released code — cite and reuse).

**Dataset:** SMD (28 machines), PSM, SWaT, RCAEval (735 failure cases, 11 fault types), LEMMA-RCA. Splits: dataset-native train/test where defined; otherwise chronological 50/20/30 as used in this notebook, generalized per-machine/per-service. Preprocessing: per-dataset z-score normalization fit on the guaranteed-normal historical portion only (as done here). No augmentation (real time series). No PII (all public, already-anonymized industrial/IT telemetry).

**Baseline methods:** MicroRCA (graph+PageRank), CausalRCA (causal DAG), RCD, ε-Diagnosis, SHAP-only, Granger-only — plus this notebook's own LightGBM/Transformer pair as the prediction-stage baseline.

**Proposed method:** the existing 5-stage pipeline, generalized across datasets with ablation switches.

**Repetitions:** ≥5 random seeds per dataset/method combination.

**Statistical analysis:** paired Wilcoxon signed-rank test across matched fault cases (non-parametric, appropriate for skewed precision@k distributions), bootstrap 95% CIs, Cohen's d effect size, Bonferroni or Holm correction across the 5 RQs' multiple comparisons.

---

## Phase 10 — Ablation Study Design

| Configuration | Scientific question answered |
|---|---|
| Full pipeline | Upper-bound performance |
| − causal filter | Does causal filtering add value over attribution alone? (H1) |
| − conformal layer | Does calibration change selective-prediction risk, or just add a UI feature? (H2) |
| Transformer → LightGBM swap | Does the predictor choice interact with downstream attribution quality? |
| Baseline architecture (MicroRCA alone) | Where does the full pipeline sit relative to an established single-method baseline? |

---

## Phase 11 — Robustness Testing (relevant subset only)

- **Domain shift:** cross-machine/cross-dataset generalization (train on SMD, evaluate root-cause attribution quality on RCAEval-style faults) — directly relevant, already flagged as this notebook's biggest unproven assumption.
- **Missing/noisy telemetry:** randomly drop KPI dimensions or inject sensor noise, measure attribution/precision degradation — relevant, telemetry gaps are common in real operations.
- **Alarm-storm conditions:** simulate concurrent overlapping faults, check whether attribution/causal filtering still isolates the correct dimension — relevant to telecom's stated context.
- *Not relevant here:* adversarial inputs, OCR errors, image compression, demographic/lighting variation — none apply to this KPI time-series problem.

---

## Phase 12 — Efficiency / Production Evaluation

**No production deployment currently exists**, so P50/P95/P99 latency, concurrent-user throughput, and cost/request are **not measurable today** and should not be claimed. What *can* legitimately be reported: per-inference compute time for each pipeline stage (already near-instant for LightGBM/SHAP; the Transformer forward pass and Granger test are the dominant costs) as a **feasibility** argument, not a production SLA claim. A genuine production-evaluation section would require an actual deployed pilot — flag this as future work, not a Phase-1-paper requirement.

---

## Phase 13 — Explainability and Error Analysis

Already partially present (SHAP, root-cause precision@k breakdown by segment) — extend to:
- False positive/negative breakdown by fault type (once multiple datasets are used).
- Failure-mode analysis for the "missed" incident (20786–21195 in this notebook) — characterize *why* some faults have no learnable precursor pattern (is it duration, magnitude, KPI subset?).
- Model-confidence vs. error correlation (does low-conformal-confidence correlate with wrong root-cause attribution, not just wrong prediction?).
- SHAP already used appropriately; do not add LIME/Grad-CAM — not relevant to tabular/time-series KPI data.

---

## Phase 14 — Dataset Contribution Opportunity

**Assessment: no new dataset should be proposed.** SMD/PSM/SWaT/RCAEval/LEMMA-RCA already cover this space well; introducing "yet another RCA dataset" adds no value and would face immediate reviewer skepticism (Phase 23). The one legitimate artifact worth releasing is the **evaluation harness/protocol** itself (code that runs the ablation across all 5 datasets with the 6 baselines) — a benchmarking-code contribution, not a data contribution.

---

## Phase 15 — Mathematical Formulation

- **Input:** $X_t \in \mathbb{R}^{d}$, KPI vector at time $t$ ($d{=}38$ here); window $X_{t-W+1:t}$.
- **Output:** $\hat{y}_t = P(\exists\, \text{anomaly in } (t, t+H]\mid X_{t-W+1:t}) \in [0,1]$.
- **Predictor:** $\hat{y}_t = f_\theta(X_{t-W+1:t})$, $f_\theta \in \{\text{GBDT}, \text{Transformer}\}$, $\theta$ fit by minimizing weighted BCE $\mathcal{L}(\theta) = -\sum_t w_{y_t}\big[y_t\log\hat y_t + (1-y_t)\log(1-\hat y_t)\big]$.
- **Attribution:** Shapley value $\phi_i(x) = \sum_{S \subseteq N\setminus\{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}\big[f(S\cup\{i\}) - f(S)\big]$, aggregated per KPI dimension.
- **Causal filter:** for candidate $i$ and target $j$, Granger-causal if $\text{SSR-F-test}\big(\text{Var}(Y_j \mid Y_j^{<t}) \text{ vs. } \text{Var}(Y_j \mid Y_j^{<t}, X_i^{<t})\big)$ rejects $H_0$ at $p<0.05$ for some lag $\ell \in \{1..5\}$.
- **Conformal set:** $\mathcal{C}(x) = \{c : 1-\hat p_c(x) \le \hat q\}$, $\hat q = $ the $\lceil (n{+}1)(1{-}\alpha)\rceil/n$ empirical quantile of calibration nonconformity scores — guarantees $P(y \in \mathcal{C}(x)) \ge 1-\alpha$ marginally.
- **Objective for the proposed study (not the model):** maximize precision@k of the *joint* pipeline's root-cause set subject to maintaining the conformal coverage guarantee — an evaluation objective, not a trained optimization.

(No new loss/optimization is proposed for D1 — this section documents the existing math for the paper's Problem Formulation section, not a novel derivation.)

---

## Phase 16 — Proposed Architecture (layers)

```
Input Layer            KPI window X_{t-W+1:t}
        ↓
Preprocessing          per-dataset normalization (existing)
        ↓
Prediction Layer       LightGBM / Transformer            [existing]
        ↓
Attribution Layer      SHAP TreeExplainer                [existing]
        ↓
Causal Filter Layer    Granger-causality test              [existing]
        ↓
Calibration Layer      Split-conformal (LAC)               [existing]
        ↓
Decision/Reasoning     ablation-switchable combination      [NOVEL: the paper's contribution
                        of the 4 layers above                is evaluating THIS combinatorics]
        ↓
Output                 root-cause set + confidence + verdict
        ↓
Evaluation             precision@k/recall@k vs. ground truth, across 5 datasets [NOVEL: scale]
```

Existing components: everything except the ablation harness and multi-dataset evaluation, which are what must be built.

---

## Phase 17 — Candidate Titles (15, top 5 ranked)

1. *"Does Causal Filtering and Conformal Calibration Improve Root-Cause Attribution? A Multi-Dataset Ablation Study"* ★
2. *"Attribution Is Not Enough: An Empirical Study of Causally-Filtered, Calibrated Root-Cause Analysis"* ★
3. *"Toward Trustworthy Root-Cause Analysis: Quantifying the Value of Causal and Uncertainty-Aware Layers"* ★
4. *"A Systematic Ablation of Attribution, Causality, and Calibration in Multivariate Time-Series Root-Cause Analysis"* ★
5. *"When Does Causal Reasoning Help Root-Cause Analysis? Evidence from Five Public Benchmarks"* ★
6. "Trustworthy Root-Cause Analysis for Networked Systems: Design, Ablation, and Evaluation"
7. "From Correlation to Confidence: A Layered Framework for Root-Cause Attribution"
8. "Calibrated and Causally-Filtered Root-Cause Ranking: An Empirical Evaluation"
9. "Beyond Point Accuracy: Evaluating Trustworthiness in Root-Cause Analysis Pipelines"
10. "A Comparative Ablation of Correlational and Causal Root-Cause Attribution Methods"
11. "Conformal Confidence for Root-Cause Analysis: Does It Change What Engineers Should Trust?"
12. "On the Value of Causal Precedence Testing in Multivariate Anomaly Root-Cause Analysis"
13. "Evaluating the Composability of Attribution, Causality, and Calibration for AIOps"
14. "A Benchmarking Protocol for Trustworthy Root-Cause Analysis Across Heterogeneous Systems"
15. "Rethinking Root-Cause Evaluation: Precision, Causality, and Calibrated Confidence"

(★ = top 5, ranked in order shown)

---

## Phase 18 — Abstract Blueprint

**Background:** Root-cause analysis (RCA) for networked/IT systems increasingly relies on feature-attribution methods (e.g., SHAP), but attribution alone conflates correlation with causation and offers no calibrated confidence.
**Gap:** Existing RCA benchmarks (e.g., RCAEval, LEMMA-RCA) score attribution methods in isolation; no study jointly evaluates whether adding causal-precedence filtering and conformal calibration measurably improves root-cause correctness or the reliability of reported confidence.
**Objective:** Determine whether, and under what conditions, chaining attribution, causal filtering, and conformal calibration improves root-cause precision/recall and calibrated risk-coverage over any single stage alone.
**Method:** A 4-layer pipeline (predict → attribute → causally filter → calibrate) evaluated with a 4-way ablation against 6 baselines (MicroRCA, CausalRCA, RCD, ε-Diagnosis, SHAP-only, Granger-only) on 5 public datasets (SMD, PSM, SWaT, RCAEval, LEMMA-RCA).
**Experimental setup:** ≥5 seeds per configuration; paired Wilcoxon significance testing; precision@k, recall@k, MRR, MAP@k, ECE, conformal coverage, and false-alarm rate as metrics.
**Main findings (to be measured):** "[the joint pipeline improves precision@k by X% (95% CI [Y,Z]) over attribution-only, p<0.05, on N/5 datasets]" — placeholder, not invented.
**Contribution:** an ablation-driven evaluation protocol and empirical evidence for when causal filtering and calibration are, and are not, worth the added complexity in RCA pipelines.

---

## Phase 19 — Full Paper Structure

| Section | Purpose | Needs |
|---|---|---|
| 1. Introduction | Motivate trust gap in RCA | Telecom/AIOps incident-cost framing, 3–5 citations |
| 2. Related Work | Position vs. RCAEval, LEMMA-RCA, MicroRCA, CausalRCA, conformal-prediction literature | Lit-review matrix (Phase 22) |
| 3. Problem Formulation | Phase 15 math | Equations, notation table |
| 4. Proposed Method | The 4-layer pipeline + ablation switches | Fig. 1, Fig. 2 |
| 5. System Architecture | Phase 16 layers | Fig. 1 |
| 6. Experimental Setup | Phase 9 | Table 2, 3 |
| 7. Results | RQ1–RQ5 answers | Fig. 4–7, Table 4 |
| 8. Ablation Studies | Phase 10 | Fig. 6, Table 5 |
| 9. Discussion | Interpret which layers help, when | — |
| 10. Limitations | Single-domain proxy datasets; Granger ≠ real intervention; no human study yet | — |
| 11. Threats to Validity | Dataset selection bias, baseline reimplementation fidelity, seed variance | — |
| 12. Practical Implications | False-alarm reduction for on-call engineers | — |
| 13. Conclusion | Summarize RQ answers | — |
| 14. Future Work | Direction 2 (human trust), Direction 3 (real RAN data) as follow-ons | — |

---

## Phase 20 — Figure Plan

1. **Fig 1 — Pipeline architecture** (Phase 16 diagram): shows what's existing vs. novel-to-evaluate.
2. **Fig 2 — Ablation configurations** visual (full / −causal / −conformal / baseline).
3. **Fig 3 — Experimental pipeline** (datasets → splits → methods → metrics).
4. **Fig 4 — Precision@k/Recall@k comparison** across datasets and methods (grouped bar).
5. **Fig 5 — Risk-coverage curve** (conformal vs. fixed-threshold) — evidence for H2.
6. **Fig 6 — Ablation results** (delta precision@k per removed component) — evidence for H1.
7. **Fig 7 — Per-fault-type breakdown** (where does the pipeline help most/least).
8. **Fig 8 — Reliability diagram** before/after conformal wrapping (already prototyped in this notebook).

---

## Phase 21 — Table Plan

1. **Table 1** — Comparison with existing RCA research (method, dataset, metrics, causal?, calibrated?).
2. **Table 2** — Dataset statistics (5 datasets, machines/services, fault counts, fault types).
3. **Table 3** — Experimental configuration (hyperparameters, seeds, splits).
4. **Table 4** — Baseline comparison (precision@k, recall@k, MRR, MAP@k per method).
5. **Table 5** — Ablation study results (full vs. each component removed).
6. **Table 6** — Compute cost per pipeline stage.
7. **Table 7** — Statistical significance (p-values, effect sizes per RQ).
8. **Table 8** — Error analysis (false positive/negative breakdown by fault type).

---

## Phase 22 — Related Work Literature Matrix (real papers only)

| Paper | Year | Problem | Method | Dataset | Metrics | Strength | Limitation | Relevant Gap |
|---|---|---|---|---|---|---|---|---|
| Su et al., OmniAnomaly (KDD) | 2019 | MTS anomaly detection | Stochastic RNN+VAE | SMD (introduced) | F1 | Source of SMD | No RCA, no causal, no calibration | Baseline dataset only |
| RCAEval (arXiv:2412.17015) | 2024 | RCA benchmarking | Multi-method eval harness | 3 microservice systems, 735 cases | PR@K, MRR, MAP@K | Standardized RCA metrics | No causal/conformal ablation, no telecom domain | Direct gap target |
| LEMMA-RCA (arXiv:2406.05375) | 2024 | Multi-domain RCA dataset | Multi-modal | IT+OT (microservices, water) | — | Domain breadth | No telecom RAN; no causal+calibration study | Extend, don't replace |
| RUN / Neural Granger RCA (arXiv:2402.01140) | 2024 | Causal-graph RCA | Neural Granger discovery | Microservices | PR@K | Causal-graph building | No calibrated confidence layer | Direct precedent for causal layer |
| 5G RAN GNN+Transformer RCA (arXiv:2406.15638) | 2024 | RAN anomaly RCA | GNN+Transformer | Real 5G RAN (not public) | — | Closest telecom-domain analogue | No causal/correlation split, no calibration | Motivates Direction 3 |
| Conformal Prediction intro (arXiv:2107.07511) | 2021 | Distribution-free UQ | Split-conformal | General ML | Coverage | Method source for calibration layer | Not applied to RCA specifically | Direct precedent for conformal layer |
| XAI human-eval systematic review (PMC11525002) | 2023 | XAI evaluation methodology | Review | N/A | N/A | Documents fragmentation | No RCA-specific protocol | Grounds Direction 2's gap |

Categories still needing review before submission: microservice RCA surveys (arXiv:2408.00803), LLM-based AIOps RCA survey (arXiv:2507.12472), MicroRCA/CausalRCA/RCD/ε-Diagnosis original papers (for correct baseline reimplementation), and any 2025–2026 RCA benchmark papers published after this search.

---

## Phase 23 — Reviewer Attack (hostile Q1 review)

| # | Reviewer concern | Fix before submission |
|---|---|---|
| 1 | "This is an engineering pipeline with off-the-shelf components — where's the methodological novelty?" | Frame explicitly as an empirical/ablation contribution (Phase 7), not a new-algorithm claim; lead with RQ1/H1 evidence, not the pipeline description. |
| 2 | "Evaluated on one machine of one dataset — no generalization evidence." | Mandatory: run on all 5 datasets/28 SMD machines before submission (Phase 6 CRITICAL item). |
| 3 | "No comparison against established RCA baselines (MicroRCA, CausalRCA, RCD)." | Implement/reuse all 6 baselines; report head-to-head, not just internal ablation. |
| 4 | "No statistical significance testing — differences could be noise." | Add paired Wilcoxon + bootstrap CIs + effect sizes across ≥5 seeds, as specified in Phase 9. |
| 5 | "Granger causality is not real causality — the causal claims are overstated." | Explicitly caveat every result as "temporal-precedence evidence," never claim proven causation; discuss as a limitation, not hide it. |
| 6 | "SMD's 38 dimensions aren't telecom KPIs — external validity to the claimed domain (telecom) is unproven." | Either drop telecom-specific framing and present as general AIOps/RCA, or explicitly scope claims to "generic multivariate KPI monitoring" with telecom as motivating context only, not as a validated domain. |
| 7 | "Root-cause precision@k of ~0.5 is unimpressive — is the method actually good?" | Report chance-level baselines and effect sizes, not just raw numbers; the contribution is the *ablation comparison*, not absolute SOTA performance. |
| 8 | "Reproducibility: is code/data released?" | Publish full evaluation harness, seeds, configs (Phase 29). |
| 9 | "The conclusions ('conformal calibration helps') may not generalize beyond this narrow setup." | Explicitly test and report per-dataset breakdowns (Fig 7); do not pool results if effect is inconsistent — report the inconsistency itself as a finding. |
| 10 | "Production relevance is anecdotal — no real deployment evidence." | Do not claim production readiness; frame Phase 12 findings as feasibility, and list human-in-the-loop validation (Direction 2) explicitly as future work, not as done. |

---

## Phase 24 — Novelty Stress Test (conservative scoring, for Direction 1 as scoped)

| Dimension | Score /10 |
|---|---|
| Originality | 5 |
| Technical depth | 5 |
| Experimental rigor (once Phase 9 is executed) | 7 |
| Reproducibility | 7 (public data + released code) |
| Dataset quality | 8 (established benchmarks) |
| Practical significance | 6 |
| Theoretical contribution | 3 (no new theory — empirical/ablation study) |
| Generalizability | 5 (unproven until multi-dataset run completes) |
| Writing potential | 7 |
| Q1-journal suitability | 6 |
| **Overall (unweighted mean)** | **5.9 → "6–6.9: substantial improvements required"** |

**Current state (this notebook only, not the redesigned study): 3–4/10** — a single-dataset, single-machine, baseline-free demo. The 5.9 above is the *achievable* score **after** executing Phase 6/9's required additions, not the current score.

---

## Phase 25 — Q1 Readiness Checklist

| Item | Status | What moves it to READY |
|---|---|---|
| Clear research gap | PARTIAL | Confirm via full systematic search (Phase 26 Stage 2), not just this session's spot-check |
| Novel scientific contribution | PARTIAL | Execute Phase 6 additions; contribution is empirical, must be demonstrated not asserted |
| Strong literature review | PARTIAL | Complete Phase 22's remaining paper categories |
| Reproducible methodology | READY | Chronological split, fixed seeds, documented already in this notebook's pattern |
| Appropriate dataset | READY | 5 public datasets identified |
| Strong baselines | MISSING | Implement MicroRCA/CausalRCA/RCD/ε-Diagnosis |
| SOTA comparison | MISSING | Same as above |
| Ablation experiments | MISSING | Build the 4-way ablation harness |
| Statistical significance | MISSING | Add Wilcoxon/bootstrap/effect-size reporting |
| Robustness experiments | MISSING | Domain-shift + missing-telemetry tests (Phase 11) |
| Scalability analysis | MISSING | Compute-cost table (Phase 20 Fig/Table 6) |
| Error analysis | PARTIAL | Present for 1 machine; extend across datasets |
| Limitations discussed | READY | Already modeled in this notebook's own §13 |
| Ethics/privacy | READY (N/A) | All data public/anonymized already; revisit only if Direction 2/3 pursued |
| Reproducibility info | PARTIAL | Needs a public code release (Phase 29) |
| Source code possibility | READY | No proprietary blockers for Direction 1 |
| Dataset availability | READY | Confirmed public |
| Real-world relevance | PARTIAL | Currently framed via SMD-as-proxy; strengthen with explicit scope caveats (reviewer concern #6) |

---

## Phase 26 — Publication Roadmap

**Stage 1 — Literature Review:** Full systematic search (not spot-check) across IEEE/ACM/Springer/Elsevier for RCA+causal+conformal combinations, 2021–2026; complete Phase 22 matrix; confirm gap still holds.
**Stage 2 — Gap Validation:** Contact/read RCAEval and LEMMA-RCA papers in full; verify no unpublished/in-review work already covers this ablation (check recent workshop papers, arXiv preprints weekly for 4–6 weeks).
**Stage 3 — Research Method Design:** Finalize RQs/hypotheses (Phase 8); pre-register the ablation design and statistical tests before running experiments (strengthens rigor claims).
**Stage 4 — Implementation:** Generalize this notebook's pipeline into a dataset-agnostic library with ablation switches; implement/wrap the 6 baselines.
**Stage 5 — Dataset Preparation:** Acquire and preprocess SMD (all machines), PSM, SWaT, RCAEval, LEMMA-RCA into a common evaluation interface.
**Stage 6 — Experiments:** Run full ablation × baseline × dataset × seed grid; log all raw results.
**Stage 7 — Statistical Analysis:** Wilcoxon/bootstrap/effect-size computation per RQ; multiple-comparison correction.
**Stage 8 — Paper Writing:** Draft using Phase 19 structure; write Results only after Stage 7 completes (no result-shaped-by-narrative writing).
**Stage 9 — Internal Review:** Have a colleague/advisor run the Phase 23 hostile-reviewer pass before submission.
**Stage 10 — Journal Selection:** Finalize target from Phase 27, confirm current quartile at submission time (rankings shift yearly).
**Stage 11 — Submission:** Prepare reproducibility package (Phase 29), submit, track revisions.

---

## Phase 27 — Journal Matching (quartiles verified via SCImago, 2026-08-28/29 session)

| Journal | Publisher | Scope fit | Verified Quartile (SJR) | Notes |
|---|---|---|---|---|
| IEEE Transactions on Network and Service Management | IEEE | Strong — network/service management, AIOps-adjacent | **Q1** (SJR 1.500, 2024) | Best scope fit for Direction 1; expects strong systems evaluation rigor |
| Journal of Network and Systems Management | Springer | Strong — network management empirical studies | **Q1** (SJR 0.876, 2024) | Slightly lower bar than IEEE TNSM but still competitive |
| Expert Systems with Applications | Elsevier | Good — applied AI/ML empirical studies, broader scope | **Q1** (SJR 1.854, 2024) | Higher SJR but less networking-specific; would need to foreground the ML/evaluation-methodology angle over telecom framing |

Verify quartiles again at actual submission time — SJR/JCR rankings are recalculated annually and can shift. No acceptance is guaranteed by any of these being Q1.

---

## Phase 28 — Ethical, Security, Regulatory Considerations

All 5 candidate datasets (SMD, PSM, SWaT, RCAEval, LEMMA-RCA) are already public and anonymized industrial/IT telemetry — **no new ethics approval needed for Direction 1.** If Direction 2 (human-subjects trust study) or Direction 3 (real RAN/operator data) is pursued later: informed consent and institutional ethics approval required for Direction 2; data-sharing agreements, anonymization, and operator NDAs would apply for Direction 3. Given the user's fintech/eKYC-adjacent employer context noted in a prior conversation turn (not part of this notebook's dataset), no regulatory (fintech) considerations apply to this specific paper — the notebook uses public server telemetry, not financial data.

---

## Phase 29 — Reproducibility Package

- README describing the ablation protocol and how to reproduce each table/figure.
- Source code for the pipeline + all baseline wrappers.
- `requirements.txt` / environment spec (this session already confirms exact library versions: `statsmodels 0.15.0`, `shap 0.52.0`, etc.).
- Fixed random seeds (≥5, listed explicitly).
- Dataset download/preprocessing scripts (all 5 datasets are scriptable from public URLs, as already demonstrated for SMD in this notebook).
- Raw experiment logs (per seed, per dataset, per method) to support the statistical analysis.
- Architecture diagrams (Phase 16, already drafted in `others/model_architectures.md`).
- No proprietary components in Direction 1 — full open release is feasible, which itself strengthens the Q1 reproducibility checklist.

---

## Phase 30 — Final Research Verdict

### Current Verdict
**C. Mainly an engineering project at present.** (Not D — there is a credible, well-grounded path to B/A via Direction 1, but that path requires substantial additional work, not reframing.)

### Strongest Research Opportunity
Direction 1: a multi-dataset ablation study quantifying whether chaining causal filtering and conformal calibration onto attribution-based root-cause analysis measurably improves correctness and calibrated confidence — a gap independently confirmed by literature search, buildable entirely on public data, with no external dependencies blocking execution.

### Proposed Research Problem
Root-cause analysis pipelines increasingly stack attribution, causal-discovery, and uncertainty-quantification techniques in practice, but the literature evaluates each technique in isolation; it is currently unknown whether, how much, and under what conditions combining them yields a measurable improvement over simpler pipelines, leaving practitioners without evidence to justify the added complexity.

### Research Gap
Existing RCA benchmarks (RCAEval, LEMMA-RCA) and causal-discovery RCA papers report point performance for single methods; no identified study performs a controlled, multi-dataset ablation isolating the marginal contribution of causal-precedence filtering and conformal calibration layered on top of attribution-based root-cause ranking, nor reports the resulting effect on false-positive rates and calibrated risk-coverage trade-offs.

### Proposed Novelty
The novelty is empirical and methodological rather than architectural: a reproducible ablation protocol and the first (to this search's knowledge) systematic, statistically validated evidence for when causal filtering and conformal calibration are and are not worth adding to an attribution-based root-cause pipeline, evaluated consistently across five heterogeneous public benchmarks.

### Top Research Questions
- RQ1: Does the full pipeline significantly outperform single-stage baselines on precision@k/recall@k?
- RQ2: How does it compare to established RCA methods (MicroRCA, CausalRCA, RCD, ε-Diagnosis)?
- RQ3: What compute/latency overhead do the added layers introduce?
- RQ4: Does any observed benefit generalize across datasets and fault types, or is it dataset-specific?

### Required New Development
Dataset-agnostic pipeline refactor with ablation switches; wrappers for 6 baseline methods; multi-seed experiment harness; statistical-testing module; expansion from 1 machine to all of SMD plus 4 additional datasets.

### Required Experiments
4-way ablation × 6 baselines × 5 datasets × ≥5 seeds; domain-shift and missing-telemetry robustness checks; compute-cost profiling.

### Required Baselines
MicroRCA, CausalRCA, RCD, ε-Diagnosis, SHAP-only, Granger-only.

### Required Dataset
Public only: SMD (all machines), PSM, SWaT, RCAEval, LEMMA-RCA. No new data collection needed for Direction 1.

### Expected Scientific Contribution
Evidence-based guidance — currently absent from the literature — on whether combining causal filtering and calibrated uncertainty with attribution-based RCA is worth its added engineering complexity, and under which conditions, for researchers and practitioners designing AIOps/RCA systems.

### Publication Risk
**Medium.** The empirical/ablation framing is defensible and gap-grounded, but reviewers may still push back on "is this novel enough" for a pure evaluation study (Phase 23, concern #1) — mitigated by rigorous execution, not by scope alone.

### Q1 Potential
**6/10**, conservatively, contingent on full execution of Phase 6/9's required additions. The current notebook alone: 3/10.

### Recommended Paper Title
*"Does Causal Filtering and Conformal Calibration Improve Root-Cause Attribution? A Multi-Dataset Ablation Study"*

### One-Sentence Research Pitch
"We propose a controlled ablation protocol combining causal-precedence filtering and conformal calibration with attribution-based root-cause ranking to address the untested assumption that layering these mechanisms improves trustworthiness, demonstrating [measured precision@k / false-alarm-rate change, TBD] over attribution-only and established RCA baselines (MicroRCA, CausalRCA, RCD, ε-Diagnosis) across five public multivariate-KPI benchmarks."

### Next 10 Actions
1. Run a full systematic literature search (not a spot-check) to confirm the gap still holds and no 2025–2026 paper has already closed it.
2. Read RCAEval and LEMMA-RCA papers in full; adopt their evaluation harness/metrics rather than inventing new ones.
3. Refactor this notebook's pipeline into a dataset-agnostic module with ablation on/off switches for causal filtering and conformal calibration.
4. Implement or vendor MicroRCA, CausalRCA, RCD, and ε-Diagnosis as baseline wrappers.
5. Acquire and preprocess all 5 datasets into a common evaluation interface.
6. Pre-register RQs, hypotheses, and the statistical test plan before running experiments.
7. Run the full ablation × baseline × dataset × seed grid; log everything.
8. Compute paired significance tests, bootstrap CIs, and effect sizes per RQ.
9. Draft the paper following the Phase 19 structure, writing Results only from completed Stage-7 numbers.
10. Run the Phase 23 hostile-reviewer pass internally (ideally with your advisor) before submitting to a Phase-27 target journal.
