# 3. Problem Formulation

Let X_t in R\^d denote the KPI vector observed at minute t (d = 38 in
this study; in a real deployment d indexes named telecom KPIs such as
latency, packet loss, jitter, or handover-failure rate). A window of the
last W minutes is written X\_(t-W+1:t).

**Early-warning prediction (Component 1).**

> y_hat_t = f_theta( X\_(t-W+1:t) ) in \[0, 1\]
>
> y_t = 1 iff an anomaly occurs at some tau in (t, t+H\] (H = prediction
> horizon)
>
> L(theta) = - sum_t w\_{y_t} \[ y_t log(y_hat_t) + (1 - y_t) log(1 -
> y_hat_t) \]

f_theta ranges over a gradient-boosted tree ensemble and a Transformer
encoder in this study (Section 4.1); w\_{y_t} is a class-imbalance
weight.

**Attribution (Component 2).**

> phi_i(x) = sum\_{S subseteq N\\{i}} \[\|S\|!(\|N\|-\|S\|-1)!/\|N\|!\]
> \* \[ f(S u {i}) - f(S) \]

phi_i(x) is the Shapley value of feature i for input x; per-KPI
importance aggregates the Shapley values of that KPI\'s derived tabular
features (Section 5.2).

**Causal filter (Component 3).**

For candidate KPI i and target KPI j, i is retained as causal evidence
for j if, at some lag l in {1,\...,L}, the null hypothesis that i\'s
past does not improve the prediction of j\'s future (beyond j\'s own
past) is rejected by an SSR F-test at p \< 0.05:

> H0: Y_j,t \~ Y_j,(t-1..t-l) vs. H1: Y_j,t \~ Y_j,(t-1..t-l) +
> X_i,(t-1..t-l)

**Calibration (Component 5).**

Given a calibration set with known labels, the nonconformity score for
point i is s_i = 1 - P(true class \| x_i). The (1-alpha)-quantile of
{s_i}, denoted q_hat, defines a prediction set for a new point x:

> C(x) = { c : 1 - P(c \| x) \<= q_hat } with marginal guarantee P( y in
> C(x) ) \>= 1 - alpha
