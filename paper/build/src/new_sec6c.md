## 6.5 Component 4 \-- Explanation faithfulness, stability, and the confidence-quality gap

Components 1-3 and 5 are scored against ground truth. Component 4 \-- the
narrative an engineer actually reads \-- is the one stage where "it looks
reasonable" is the easiest failure to miss, so we audit it directly rather
than presenting a sample output and moving on. The explainer is a
deterministic template: every clause is a lookup into a number already
computed by another component, following the narrate-don't-diagnose
discipline that RCACopilot- and RCAgent-style systems arrived at. An LLM may
phrase it more fluently, but it is structurally prevented from asserting any
fact not present in the evidence dictionary. A representative output for
SMD segment 19367-20088 reads:

> \[t=19366\] Predicted P(degradation within next 30 min) = 0.997.
> 90%-coverage conformal prediction set = {degrading} -> high confidence
> degradation. Early-warning behavior on this incident: 98 min before the
> labeled onset. Top-5 implicated KPI dimensions (SHAP): \[1, 7, 13, 14, 30\].
> Ground-truth root-cause dimensions: \[1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15,
> 16, 25, 28\] -> 3/5 flagged dimensions confirmed correct.

Grounding the text is necessary but not sufficient: the numbers it grounds
against can themselves be unreliable. We therefore measure two properties of
the underlying attribution.

*Faithfulness (deletion test).* For each event we zero out (impute the
standardized training mean) the top-5 SHAP-ranked dimensions' mean/std/last
entries and measure the resulting drop in predicted probability, against the
drop from deleting five random dimensions, averaged over 200 repeats. A
faithful explanation should cause the larger drop.

*Stability (perturbation test).* For each event's pre-onset window we add
Gaussian noise (sigma = 0.05 in standardized units, roughly 5% of one
standard deviation), recompute features and the SHAP ranking, and compare to
the unperturbed ranking by Spearman rho, averaged over 20 draws.

*Attribution entropy* is a third, perturbation-free proxy: low normalized
entropy means SHAP concentrates attribution on a few dimensions.

By the deletion test the explanations pass on all eight events. But the
margin varies enormously \-- from 0.9999 against 0.194 for random deletion on
segment 15849-16368, to 0.208 against 0.200 on segment 20786-21195, a
difference that is essentially nothing. Three of the eight events also begin
from a base probability above 0.999, i.e. the model is already at its output
ceiling before any deletion, so "top-5 deletion drops probability more than
random" is close to a floor effect on those events rather than strong
evidence. Rankings are highly stable (Spearman rho 0.990-0.998 across all
eight events), but stability is a property of the explanation, not of its
correctness.

That distinction is what the next test isolates. Every one of the eight
events is predicted correctly and, on seven of eight, confidently. The
critical-trustworthiness matrix therefore collapses to exactly two cells:
four **ideal** events (prediction correct, root cause correct) and four
**dangerous diagnostic errors** (prediction correct, root cause wrong), with
zero detection failures. The operative question is whether any signal
available to the system distinguishes the second group from the first.

**Table 12. Component 4 confidence and explanation-quality signals,
partitioned by whether root-cause localization was correct (SMD, n = 8).**

| **Signal** | **RCA-correct mean (n=4)** | **RCA-wrong mean (n=4)** | **Difference** |
|---|---|---|---|
| p_hat (predicted probability) | 0.995 | 0.991 | 0.004 |
| attribution_entropy (SHAP spread) | 0.712 | 0.696 | 0.016 |
| stability_rho (rank stability) | 0.992 | 0.995 | **-0.003** |
| faithfulness_margin (deletion test) | 0.510 | 0.515 | **-0.005** |

None of the four separates the groups by any meaningful margin, and two of
them \-- stability and faithfulness \-- point in the *wrong* direction,
scoring marginally higher on the events where the root cause was wrong. The
conformal layer does no better: the single ambiguous event does happen to be
an RCA-wrong case, but three of the remaining four RCA-wrong events are
still labeled confidently degrading.

This is the paper's central trustworthiness finding, and it is negative.
TrustNet-RCA can be maximally confident that a problem exists, produce an
explanation that is stable under perturbation and faithful by deletion test,
and still name the wrong KPI \-- with no signal in its own outputs marking
that case as different. The practical implication is a warning about a
tempting design: naively combining predicted probability, attribution
entropy, and stability into a single composite trust score would **not**, on
this evidence, separate correct from incorrect localization. Building such a
score requires either a signal not tested here \-- causal-evidence agreement
from Component 3, or cross-machine variance \-- or enough data to resolve an
effect that four-versus-four groups cannot. We report this as it came out
rather than reframing it as a partial success.

The sample is eight events on one machine, and every statistic above is
descriptive of that machine rather than a confirmatory population-level
test. The direction of the result is nonetheless the safer one to be wrong
about: it counsels against trusting a confidence score that has not been
validated against localization ground truth.

## 6.6 Component 5 \-- Uncertainty quantification

Split-conformal calibration on the CALIB block yields q_hat = 0.985 and
achieves 0.955 empirical coverage on EVAL against the 90% target \-- a
valid, if conservative, guarantee. 20.0% of EVAL windows are routed to the
ambiguous ({normal, degrading}) set, i.e. an explicit escalate-to-human
signal, and 0.0% to the empty set. The raw probability's ECE on EVAL is
0.118: Figure 8 shows the model is overconfident in its top bin (mean
predicted probability 0.986, empirical positive rate 0.752), exactly the
failure mode conformal wrapping is designed to survive without requiring the
underlying model to be well calibrated.

![](figures/fig8_reliability_diagrams.png){width="7.0in" height="2.9in"}

*Figure 8. Reliability diagrams for the raw probability, showing
overconfidence in the top bin that the conformal layer is designed to
tolerate.*

The conformal layer has been run on SMD only. CALIB blocks are constructed
and held out for TelecomTS and RCAEval, but the coverage guarantee is not
yet demonstrated on either, which Section 8 records.

## 6.7 Early-warning lead time

Table 13 reports, per SMD ground-truth segment, the gap between the first
sustained risk-score crossing of 0.5 and official onset. Out-of-sample
(EVAL + CALIB) the mean lead time is 42.2 minutes, with 4 of 5 incidents
caught and one missed entirely.

**Table 13. Component 1 early-warning lead time on SMD. TRAIN-block
segments are in-sample and shown only for contrast.**

| **Segment** | **Block** | **Lead time (min)** |
|---|---|---|
| 15849-16368 | train (in-sample) | 43 |
| 16963-17517 | train (in-sample) | 40 |
| 18071-18528 | train (in-sample) | 64 |
| 19367-20088 | eval | 98 |
| 20786-21195 | eval | **missed** |
| 24679-24682 | calib | 2 |
| 26114-26116 | calib | 27 |
| 27554-27556 | calib | 42 |

![](figures/fig11_case_study_timeline.png){width="3.4in" height="2.4in"}

*Figure 11. Risk score against ground truth for segment 19367-20088, the
incident with the longest measured lead time.*

Lead time is highly heterogeneous rather than uniform: one incident is
flagged 98 minutes early with a clean ramp-up, one CALIB-block incident is
caught only ~2 minutes early (essentially at onset), and one incident's risk
score jumps from ~0.001 to ~0.98 in a single step with no gradual precursor
within the model's 20-minute lookback and is missed by the lead-time rule
entirely. We report this variance explicitly rather than only the mean,
since averaging it away would misrepresent the system's actual, uneven
early-warning capability \-- itself a trustworthiness finding, not only a
performance number. The TelecomTS runway analysis of Section 6.1 is the same
observation at a different time scale: what the system can anticipate is
bounded by what the incident telegraphs.
