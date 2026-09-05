# 9. Practical Implications and Future Work

For practitioners, four design choices in TrustNet-RCA are directly
actionable regardless of the specific models used. First, treat prediction,
attribution, causal filtering, explanation, and calibration as separately
monitored pipeline stages, so that a regression in one is not masked by
another. Second, report lead time, root-cause accuracy, and per-fault-type
performance per incident rather than only pooled: Section 6.1 shows AUPRC
spanning 0.101 to 0.823 across anomaly types while AUROC stays within a
narrow 0.877-0.971 band, so an aggregate score (particularly an aggregate
AUROC) can conceal that a system is close to unusable on entire fault
families. Third, use a conformal or similarly calibrated abstention signal
to route low-confidence cases to a human rather than resolving them
silently. Fourth, and following directly from Section 6.5: do not gate that
escalation on a composite trust score assembled from the model's own
confidence and explanation-quality signals until that score has been
validated against localization ground truth, because on this evidence it
would not separate correct diagnoses from incorrect ones.

Four experiments follow immediately from what Section 6 does not establish,
in rough order of how much they would change the paper's claims.

*Score localization on the telecom legs.* This is the largest gap. Both
TelecomTS (named `affected_kpis`) and RCAEval (root-cause service and
indicator) ship localization ground truth, and both loaders already expose
it; the fixed-k metrics of Section 5.6 are implemented. Until this is run,
attribution quality is a claim about server telemetry only, and the
confidence-quality finding of Section 6.5 rests on eight events.

*Extend the conformal layer to the telecom legs.* CALIB blocks are built
and held out for TelecomTS and RCAEval but unused. The coverage guarantee is
distribution-free in theory; demonstrating it at 10 Hz on radio telemetry is
the test that matters operationally.

*Statistical validation.* Multi-seed runs with paired significance tests and
effect sizes across all three legs, which no result in this paper currently
carries, and which the eight-event Component 4 analysis needs most.

*Broaden the RCA leg and add external baselines.* RE1 / Online Boutique is
123 of RCAEval's 735 cases; Sock Shop, Train Ticket, and the RE2/RE3 multi-
source suites remain untouched. Comparison against MicroRCA \[13\],
CausalRCA, RCD, and epsilon-Diagnosis is required before any comparative
claim, as is the joint ablation that this paper's related-work review found
no prior study to have reported: does causal filtering plus conformal
calibration measurably improve root-cause precision and selective-prediction
risk over attribution alone?

Beyond that, two lines are worth naming. The causal-evidence signal of
Component 3 should be tested as the localization-quality signal that Section
6.5's candidates failed to provide. And a controlled human-subjects study
should test whether exposing the causal-versus-correlational distinction and
the abstention flag actually changes engineer diagnostic accuracy and
calibrated trust, a gap independently identified by a systematic review of
XAI human-evaluation methodology \[21\]. A third, higher-risk line,
contingent on data access, is validation on real RAN-level KPI telemetry
against the published GNN+Transformer RAN RCA system \[12\].