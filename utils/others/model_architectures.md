# Model Architectures Used in `Telecom_Trustworthy_AI_Framework.ipynb`

Six distinct models/algorithms are used across the notebook's five components. Two are trained
models (LightGBM, the Transformer); the other four are deterministic algorithms wrapped around
those two models' outputs (SHAP, Granger causality, split-conformal prediction, Personalized
PageRank). This file documents each one's architecture, exact hyperparameters, and where it sits
in the pipeline.

## Pipeline overview

```
KPI window (W=20 min x 38 dims)
        │
        ├──► [1a] LightGBM  ──► P(degrade) ──► [SHAP TreeExplainer] ──► per-KPI root-cause ranking
        │      (tabular,           │                                          │
        │       114 feats)         │                                          ▼
        │                          │                              [Granger causality test]
        └──► [1b] Transformer ──► P(degrade) ──► [Split-conformal (LAC)] ──► prediction set +
               (raw window)                             ▲                    escalation flag
                                                          │
                                                   CALIB block only

Separately (illustrative): NE topology graph ──► [Personalized PageRank] ──► root-cause node ranking
```

---

## 1. LightGBM classifier — Component 1 (auditable baseline) & Component 2 (SHAP target)

**Type:** Gradient-boosted decision tree ensemble (GBDT), `lightgbm.LGBMClassifier`.

```
Input: x ∈ R^114   (38 KPI dims × {rolling mean, rolling std, last value}, W=20-step window)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  Tree 1 (depth ≤ 5, leaf-wise)  →  Tree 2  →  ...  → Tree 300  │
│  each tree fit on the negative gradient of log-loss           │
│  (standard gradient boosting, leaf-wise-with-max-depth growth) │
└───────────────────────────────────────────────────────────┘
        │  Σ leaf outputs (weighted by learning_rate)
        ▼
   raw score → sigmoid → P(degradation within next H=30 min)
```

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `scale_pos_weight` | neg/pos count ratio (class-imbalance correction) |
| Input dim | 114 (38 mean + 38 std + 38 last) |
| Output | scalar probability, binary |

**Role:** (a) a fast, auditable fallback predictor benchmarked against the Transformer in
Component 1; (b) the model actually explained in Component 2 — `shap.TreeExplainer` needs a tree
ensemble to compute **exact** Shapley values cheaply, which a transformer does not offer without a
slower approximate explainer.

---

## 2. TinyTransformer — Component 1 (primary predictor)

**Type:** Custom Transformer encoder classifier, PyTorch (`nn.TransformerEncoder`).

```
Input: x ∈ R^(20 × 38)         (raw W=20-step window, 38 KPI dims, not aggregated)
        │
        ▼
 Linear(38 → 32)                       "input projection" (d_model=32)
        │
        + learned positional embedding  Parameter(1, 20, 32)
        │
        ▼
┌─────────────────────────────────────────────┐
│ TransformerEncoderLayer × 2                  │
│  ┌─────────────────────────────────────┐    │
│  │ Multi-Head Self-Attention (4 heads)  │    │
│  │        + residual + LayerNorm        │    │
│  ├─────────────────────────────────────┤    │
│  │ Feed-Forward: 32 → 64 → 32 (ReLU)     │    │
│  │        + residual + LayerNorm        │    │
│  └─────────────────────────────────────┘    │
│           dropout = 0.1, batch_first=True    │
└─────────────────────────────────────────────┘
        │
        ▼
 Mean-pool over the 20 time steps  →  h ∈ R^32
        │
        ▼
 Linear(32 → 16) → ReLU → Linear(16 → 1)
        │
        ▼
   logit → sigmoid → P(degradation within next H=30 min)
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 32 |
| `nhead` | 4 (head_dim = 8) |
| encoder layers | 2 |
| `dim_feedforward` | 64 |
| dropout | 0.1 |
| positional encoding | learned (not sinusoidal) |
| pooling | mean over time |
| optimizer | Adam, lr = 1e-3 |
| loss | `BCEWithLogitsLoss(pos_weight=scale_pos_weight)` |
| batch size / epochs | 256 / 6 |

**Role:** the primary Component 1 predictor (higher AUROC/AUPRC than LightGBM on the held-out EVAL
block); its output probability is also what Component 5's conformal wrapper calibrates, and what
Component 4's explanation template quotes.

---

## 3. SHAP TreeExplainer — Component 2 (root-cause attribution)

**Type:** Not a trained model — an *exact* attribution algorithm (TreeSHAP) that walks the LightGBM
ensemble's tree structure to compute Shapley values in polynomial time (no sampling/approximation,
unlike `KernelExplainer`, which is why LightGBM rather than the Transformer is explained here).

```
LightGBM ensemble (300 trees)  +  one input x ∈ R^114
        │
        ▼
 TreeSHAP: exact per-tree, per-feature contribution decomposition
        │
        ▼
 φ ∈ R^114   (one Shapley value per tabular feature)
        │
        ▼
 aggregate |φ| across each KPI's {mean, std, last} triplet
        │
        ▼
 38 per-KPI-dimension importance scores → rank → top-k root-cause candidates
```

**Role:** turns the LightGBM prediction into a ranked list of implicated KPI dimensions, which is
then validated against SMD's ground-truth `interpretation_label` (precision/recall@k) and fed into
the Granger-causality step below.

---

## 4. Granger causality test — Component 3 (correlation vs. causal evidence)

**Type:** Statistical hypothesis test, not a model. `statsmodels.tsa.stattools.grangercausalitytests`
(nested VAR models + F-test on the restriction).

```
target series Y_t (KPI 14)        candidate series X_t (SHAP top-k KPI)
        │                                   │
        ▼                                   ▼
 fit  Y_t ~ Y_{t-1..t-L}         fit  Y_t ~ Y_{t-1..t-L} + X_{t-1..t-L}   for L = 1..5
        │                                   │
        └──────────────► SSR F-test: does adding X's lags significantly reduce residual error? ◄──┘
                                   │
                                   ▼
                     p_min = min p-value across lags 1..5
                                   │
                    p_min < 0.05 ?  ──Yes──►  "causal evidence" (temporal precedence)
                                   │
                                   No
                                   ▼
                          "correlational only"
```

**Role:** filters Component 2's correlational SHAP candidates down to the subset with temporal
predictive power — necessary-but-not-sufficient evidence for true causality (no intervention is
performed), reported as a confidence tier rather than a proof.

---

## 5. Split-conformal prediction (LAC) — Component 5 (uncertainty quantification)

**Type:** Distribution-free calibration wrapper around the Transformer's probability output — not a
model, a post-hoc statistical procedure (Angelopoulos & Bates' "least ambiguous set" classifier).

```
Transformer probability p₁ = P(degrading)  on the CALIB block, with known true labels y
        │
        ▼
 nonconformity score sᵢ = 1 − P(true class)  for each calibration point
        │
        ▼
 q̂ = the ⌈(n+1)(1−α)⌉ / n empirical quantile of {sᵢ}     (α = 0.10 → target 90% coverage)
        │
        ▼
 at inference, for a new p₁: keep class c in the set if 1 − p(c) ≤ q̂
   keep "normal"    if (1 − p₁) ≤ q̂
   keep "degrading" if p₁       ≤ q̂
        │
        ▼
 prediction SET ∈ { {normal}, {degrading}, {normal,degrading} (ambiguous→escalate), {} (rare) }
```

**Role:** converts a raw (and, per the reliability diagram, over-confident) probability into a
calibrated set with a finite-sample coverage guarantee, and produces the explicit
"abstain-and-escalate-to-human" signal used in Component 4's explanation.

---

## 6. Personalized PageRank over topology graph — Component 2, graph view (illustrative)

**Type:** Graph-propagation algorithm (`networkx.pagerank`), not a learned model — the fixed-graph
precursor to a learned GNN (e.g. GDN, GAT, or the transformer+GNN 5G-RAN model cited in the
notebook's related work).

```
NE dependency graph G (Core → Aggregation → BaseStation_{A,B} → Cell_*)
        │
observed per-node severity (simulated fault propagation + noise)
        │
        ▼
 personalization vector = severity / Σ severity
        │
        ▼
 PageRank on REVERSED graph G⁻¹, restarting at personalization
   (so random-walk mass flows from symptom nodes back toward upstream causes)
        │
        ▼
 node ranking, highest score = most likely root-cause NE
```

**Role:** demonstrates the topology-aware half of Component 2 — ranking *which network element*
is the propagation source, complementing SHAP's *which KPI dimension* ranking. In production, the
fixed graph and hand-set severities would be replaced by a learned GNN whose edge weights and
node-deviation scores come from training data.

---

## Summary table

| # | Name | Category | Trained? | Library | Input | Output |
|---|---|---|---|---|---|---|
| 1 | LightGBM | Gradient-boosted trees | Yes | `lightgbm` | 114-d tabular window | P(degrade) |
| 2 | TinyTransformer | Transformer encoder | Yes | `torch` (custom) | 20×38 raw window | P(degrade) |
| 3 | SHAP TreeExplainer | Exact attribution (TreeSHAP) | No (deterministic given #1) | `shap` | 114-d tabular window | 38 per-KPI importances |
| 4 | Granger causality test | Statistical hypothesis test | No | `statsmodels` | 2 time series | p-value / causal verdict |
| 5 | Split-conformal (LAC) | Calibration wrapper | No (calibrated given #2) | numpy (hand-rolled) | probability + CALIB set | prediction set |
| 6 | Personalized PageRank | Graph propagation | No | `networkx` | topology graph + severities | node ranking |
