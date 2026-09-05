# 8. Limitations and Threats to Validity

**The early-warning anchor rests on a single dataset.** Section 6.1 records
why: RCAEval's scheduled fault injections make anticipation ill-posed, so
TelecomTS alone carries the 5G early-warning claim. A single dataset
carrying a core claim is a real concentration of risk, not a formality.

**TelecomTS anomalies are largely synthetic.** Ten of its eleven anomaly
types are injected; Jamming is the one real over-the-air phenomenon, and
those seven events have zero pre-onset runway. Results on injected faults
may not transfer to naturally occurring RAN degradation. The dataset is also
testbed-derived, not operator-network data, and involves no O-RAN interfaces
or components; we make no O-RAN claim.

**SMD is not telecom data and is not treated as such.** Its 38 dimensions
are generic server metrics, not named radio KPIs. It is reported strictly as
an out-of-domain generalization check, and no SMD number in this paper
should be read as a measurement about radio behavior.

**Localization is scored on one leg only.** Section 6.3's attribution
results are SMD-only. TelecomTS and RCAEval both ship localization ground
truth, and both loaders expose it, but the root-cause ranking is not yet
scored on either, so "RCA rigor" is not yet earned on the leg assigned to
carry it.

**One RCAEval system of nine.** Only RE1 / Online Boutique is used: 123 of
735 available cases. Sock Shop, Train Ticket, and the RE2/RE3 multi-source
suites are untouched.

**No conformal layer on the new legs.** CALIB blocks are built and held out
for TelecomTS and RCAEval, but the coverage guarantee is demonstrated on SMD
only.

**No statistical validation.** Every leg is a single seed and a single
split. No significance testing, confidence intervals, or effect sizes are
computed, and hyperparameters were fixed by hand on SMD and reused verbatim
elsewhere. That was deliberate, to keep the legs comparable, but it means no
leg is tuned and all numbers are untuned-baseline numbers.

**The Component 4 audit is underpowered.** The confidence-quality result in
Section 6.5 rests on eight events on one machine, partitioned four against
four. It is descriptive of that machine, not a population-level test. We
draw a cautionary conclusion from it rather than a quantitative one; a
larger sample could reveal a small effect these groups cannot resolve.

**No external baselines.** We compare models within our own hierarchy but do
not compare the full pipeline against established RCA methods such as
MicroRCA \[13\], CausalRCA, RCD, or epsilon-Diagnosis. No comparative claim
is made.

**Granger causality is not proof of causation.** The verdicts in Section 6.4
reflect temporal-precedence evidence from an observational test, not
intervention. Neither dataset provides a configuration-change log that could
serve as a quasi-experiment.

**Illustrative, not fitted, topology experiment.** Section 6.8 uses a
hand-specified synthetic topology and injected severities. It demonstrates a
mechanism, not a validated capability on real network topology.

**No production or robustness evaluation.** No deployed system exists to
report latency or throughput SLAs against, and no adversarial-input,
alarm-storm, missing-telemetry, or drift tests were run.
