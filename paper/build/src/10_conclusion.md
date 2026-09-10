# 10. Conclusion

We presented TrustNet-RCA, a five-component framework that treats
trustworthiness as a set of separately measurable properties (prediction,
attribution, causal filtering, explanation, and calibrated uncertainty)
rather than as an emergent quality of one opaque model, and evaluated it
under one identical method across a 5G RAN benchmark (TelecomTS), a
service-management RCA benchmark (RCAEval), and an out-of-domain server telemetry
benchmark (SMD).

Three results are worth carrying forward. First, gradient-boosted trees beat
attention models on all three legs at a fraction of the training cost; we
report this negative result as one, and note that it keeps exact
TreeExplainer attribution available throughout. Second, accuracy and
calibration pull against each other, and conformal prediction resolves the
tension rather than trading it off, delivering tree accuracy under a
distribution-free coverage guarantee, which is the strongest case this work
makes for conformal methods as a management-plane primitive. Third, and most
consequentially, none of the confidence or explanation-quality signals the
framework produces distinguishes a correct root-cause localization from an
incorrect one. A system can be maximally confident, stable under
perturbation, and faithful by deletion test, and still name the wrong KPI.

That third finding is the one we would most want an operator to take
seriously, and it argues against the intuitive design it rules out: a
composite trust score assembled from predicted probability, attribution
entropy, and rank stability would not, on this evidence, tell an engineer
when to doubt the diagnosis. The causal-evidence layer is the one signal not
yet tested in that role and is the natural next candidate.

These results establish that the proposed decomposition is measurable and
instrumentable against ground truth on a public 5G benchmark. They do not
establish that it outperforms simpler alternatives, that localization
transfers beyond SMD, or that any of it survives contact with an operator
network. Those claims require the multi-seed, multi-system, statistically
validated study defined in Section 8, which we intend as the next stage of
this work.