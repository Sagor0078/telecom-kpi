# 1. Introduction

Modern telecom networks generate continuous telemetry \-- latency,
packet loss, throughput, jitter, signal strength, handover-failure rate,
CPU/memory utilization, alarms, and configuration-change events \-- and
operators increasingly rely on machine learning to detect degradation
before it escalates into a service disruption. The dominant research
emphasis, however, has been point predictive accuracy: does the model
flag the right minute? This framing is insufficient for an operational
setting in which an automated flag triggers costly human or automated
remediation. An engineer facing a flagged degradation needs four
additional answers that accuracy alone does not provide: which KPI is
implicated, whether that KPI\'s relationship to the fault is causal or
merely coincidental, why the system believes so in terms the engineer
can audit, and how much to trust the flag at all.

Feature-attribution methods such as SHAP \[1\] answer the first of these
questions but are, by construction, correlational \-- a KPI can score
highly because it moves together with the fault without driving it.
Causal-discovery methods such as Granger causality \[2\] and
constraint-based causal graphs \[3\] address this but are rarely paired
with a calibrated confidence signal, so a causal claim is reported with
the same false certainty as a correlational one. Conformal prediction
\[4\] supplies exactly that calibrated confidence but is, to our
knowledge, not evaluated jointly with attribution and causal filtering.
Existing RCA benchmarks \[5\], \[6\] score attribution methods against
each other but do not ablate the value that a causal filter or a
calibration layer adds on top.

This paper makes three contributions. First, we propose
**TrustNet-RCA**, a five-component framework \-- predict, attribute,
causally filter, calibrate, explain \-- that makes each of these questions
an explicit, separately-evaluable pipeline stage rather than an implicit
property of one end-to-end model. Second, we implement it with standard,
off-the-shelf methods at each stage and evaluate it under one identical
method on three public datasets with complementary jobs: TelecomTS \[22\],
a 5G testbed dataset with 18 named PHY/MAC/network-layer KPIs, as the
primary 5G RAN benchmark; RCAEval \[5\], with annotated microservice
failures, for service-management and RCA rigor; and the Server Machine
Dataset (SMD) \[7\] strictly as an out-of-domain generalization check.
Running one method across three domains is what allows the paper to
separate what is a property of the framework from what is a property of a
dataset. Third, we report what we found rather than what we hoped, and two
of the most useful results are negative: attention models lose to
gradient-boosted trees on every leg at 4-82x the training cost, and none of
the confidence or explanation-quality signals the framework produces
distinguishes a correct root-cause localization from an incorrect one. We
define the concrete experimental programme (Section 9) required to elevate
these findings to generalizable evidence.

The remainder of this paper is organized as follows. Section 2 reviews
related work across classical ML, deep learning, transformer,
graph-neural-network, causal, and LLM-based approaches to RCA, and
situates the identified gap. Section 3 formalizes the prediction,
attribution, causal-filtering, and calibration problems. Section 4
describes the proposed framework and its five components. Section 5 details the three-dataset experimental setup. Section 6 reports results for each component across all three legs. Section 7 discusses the findings, Section 8 states
limitations and threats to validity, Section 9 outlines practical
implications and future work, and Section 10 concludes.
