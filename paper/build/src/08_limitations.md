# 8. Limitations and Future Work

Three limitations most affect what this paper can claim, each paired with
the work that would remove it.

**Evaluation depth stops at SMD.** Section 6.3's attribution results and the
conformal coverage guarantee in Section 6.5 are both demonstrated on SMD
only. TelecomTS and RCAEval ship localization ground truth and hold out
CALIB calibration blocks, but neither the root-cause ranking nor the
conformal layer is yet scored on the legs assigned to carry the paper's
telecom and RCA claims. This is the largest gap in the paper and also the
most immediately available: the fixed-k metrics of Section 5.6 are already
implemented and both loaders already expose the labels. Running it would
also give the eight-event Component 4 audit the larger sample it needs to
move from descriptive to population-level.

**No statistical validation.** Every leg is a single seed and a single
split, with hyperparameters fixed by hand on SMD and reused verbatim
elsewhere; no significance testing, confidence intervals, or effect sizes
are computed, so all reported numbers are untuned-baseline numbers. The
work is multi-seed runs with paired significance tests and effect sizes
across all three legs.

**No external baselines.** Only one of nine available RCAEval systems
(RE1/Online Boutique) is used, and the full pipeline is compared within its
own component hierarchy rather than against established RCA methods such as
MicroRCA \[13\], CausalRCA, RCD, or epsilon-Diagnosis, so no comparative
claim is made. Closing this gap should include the joint ablation this
paper's related-work review found no prior study to have reported: whether
causal filtering plus conformal calibration measurably improves root-cause
precision and selective-prediction risk over attribution alone.
