# 2. Related Work

Classical machine learning. Tree ensembles (isolation forests, random
forests, gradient-boosted trees) remain a strong, auditable baseline for
KPI anomaly detection; a 2024 benchmark of ML methods on network anomaly
detection confirms they remain competitive with deep alternatives on
tabular telemetry \[8\].

Deep learning and Transformers. Recurrent and variational architectures
dominate multivariate KPI anomaly detection: OmniAnomaly \[7\] models KPI
vectors with a stochastic RNN and is also the source of the SMD dataset used
in this study; LSTM-NDT \[9\] forecasts KPI values and flags deviation with
a non-parametric dynamic threshold. Self-attention architectures improve on
recurrence for long-range temporal dependence \[10\]; TranAD \[11\] uses an
adversarially trained transformer encoder for multivariate anomaly
detection. Closest to our domain, \[12\] fuses a transformer temporal
encoder with a graph neural network over 5G RAN topology for anomaly
root-causing. That is effectively Components 1 and 2 of our framework, evaluated
on real (non-public) RAN data but without a causal-versus-correlational
distinction or calibrated uncertainty.

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
microservice systems; LEMMA-RCA \[6\] extends multi-domain, multi-modal RCA
evaluation to IT and operational-technology systems. Neither benchmark
ablates the joint contribution of causal filtering and calibrated
uncertainty on top of attribution, which is the gap this paper targets.

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
