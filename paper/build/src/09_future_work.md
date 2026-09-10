# 9. Practical Implications

For practitioners, four design choices in TrustNet-RCA are directly
actionable regardless of the specific models used. First, treat prediction,
attribution, causal filtering, explanation, and calibration as separately
monitored pipeline stages, so that a regression in one is not masked by
another. Second, report lead time, root-cause accuracy, and per-fault-type
performance per incident rather than only pooled: Section 6.1 shows AUPRC
spanning 0.101 to 0.823 across anomaly types while AUROC stays within a
narrow 0.877-0.971 band, so an aggregate score (particularly an aggregate
AUROC) can conceal that a system is close to unusable on entire fault
families.

Third, use a conformal or similarly calibrated abstention signal to route
low-confidence cases to a human rather than resolving them silently. Fourth,
and following directly from Section 6.5: do not gate that escalation on a
composite trust score assembled from the model's own confidence and
explanation-quality signals until that score has been validated against
localization ground truth, because on this evidence it would not separate
correct diagnoses from incorrect ones.
