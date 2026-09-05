# 7. Discussion

Taken together, the results in Section 6 show that each layer of
TrustNet-RCA produces measurable, distinguishable evidence rather than a
single opaque score. The predictor's discrimination (Section 6.1) is
separable from the attribution stage's correctness against ground truth
(6.3), which is separable again from the causal filter's ability to prune
spurious candidates (6.4), from the explanation layer's faithfulness and
stability (6.5), from the calibration layer's coverage guarantee (6.6), and
from the system's actual, uneven early-warning behavior (6.7). That
separability is the practical value of decomposing trustworthiness: a 0.920
AUROC headline alone would not have revealed that root-cause precision
plateaus at 0.529, that per-anomaly-type AUPRC spans an eightfold range,
that no TelecomTS incident affords a 30 s runway, or that 20% of predictions
should be escalated rather than acted on automatically.

Two findings cut against the paper's own initial expectations, and both are
more useful than the confirmations. The first is architectural: attention
models lose to gradient-boosted trees on all three legs, at 4-82x the
training cost. Reporting this rather than burying it has a concrete payoff
for the framework, because a tree predictor makes Component 2's
TreeExplainer attribution exact rather than approximate, and keeps
attribution comparable across datasets instead of switching explainer family
per leg. The negative result and the design are aligned rather than in
tension.

The second is the one an operator should act on. Section 6.5 finds that none
of the four confidence and explanation-quality signals the framework
produces (predicted probability, attribution entropy, rank stability,
deletion-test faithfulness) separates the events where localization was
correct from those where it was wrong; two of the four point marginally in
the wrong direction. This matters because the intuitive next engineering
step, once a pipeline emits several quality signals, is to combine them into
a composite trust score and gate escalation on it. On this evidence that
score would not work, and would be worse than no score at all, since it
would attach unwarranted confidence to the dangerous cases: prediction
correct, diagnosis wrong. The one signal not yet tested in that role is the
causal-evidence agreement of Component 3, which is the natural next
candidate precisely because it draws on temporal structure rather than on
the attribution the other signals all derive from.

There is a methodological point here too. The confidence-quality gap is only
visible because localization ground truth existed to partition events
against. A study without such labels would have observed stable, faithful,
confident explanations and concluded the explanation layer was working. This
is the argument for insisting on datasets with root-cause ground truth even
when they are less realistic than the alternative. Conversely, it is the
reason the localization scores in this paper carry a caveat: they exist for
SMD, the least telecom-like of the three legs.

These results establish that a joint pipeline is measurable and
instrumentable across three domains under one method, not that it
outperforms simpler alternatives. Section 8 states the limitations directly,
and Section 9 defines the multi-seed, multi-baseline study required to make
a comparative claim.
