# 3. Problem Formulation

Let $\mathbf{x}_t \in \mathbb{R}^{d}$ denote the KPI vector observed at time
step $t$, with $d = 18$ for TelecomTS, $d = 24$ for RCAEval, and $d = 38$ for
SMD. In deployment, $d$ indexes named telecom KPIs such as latency, packet
loss, jitter, or handover-failure rate. Write the window of the $W$ most
recent observations as $\mathbf{X}_{t-W+1:t} \in \mathbb{R}^{W \times d}$.
Throughout, $\hat{\cdot}$ denotes an estimated quantity and $H$ the
prediction horizon.

## 3.1 Early-warning prediction

Component 1 learns a predictor $f_\theta$ mapping a window to a degradation
probability,

$$\hat{y}_t = f_\theta\!\left(\mathbf{X}_{t-W+1:t}\right) \in [0, 1],$$

against the forward-looking label

$$y_t = \mathbf{1}\!\left[\,\exists\, \tau \in (t,\, t+H] \ \text{anomalous}\,\right],$$

so that a positive label at $t$ asserts a future event and the predictor
observes nothing after $t$. Parameters are fitted by minimizing the
class-weighted cross-entropy

$$\mathcal{L}(\theta) = -\sum_{t} w_{y_t}\!\left[\,y_t \log \hat{y}_t + (1 - y_t)\log\!\left(1 - \hat{y}_t\right)\right],$$

where $w_{y_t}$ compensates for class imbalance. In this study $f_\theta$
ranges over the model hierarchy of Section 5.5.

## 3.2 Attribution

Component 2 assigns each feature $i$ its Shapley value with respect to the
fitted predictor $f$,

$$\phi_i(x) = \sum_{S \subseteq N \setminus \{i\}} w_S \left[ f(S \cup \{i\}) - f(S) \right],$$

$$w_S = \frac{|S|!\,(|N| - |S| - 1)!}{|N|!},$$

where $N$ is the feature set and $w_S$ is the Shapley weight. Per-KPI importance aggregates $|\phi_i|$ over
the derived features of that KPI (Section 5.3). For tree ensembles this is
computed exactly rather than approximated, which Section 6.1 shows is
available on every leg because the strongest predictor is a tree model.

## 3.3 Causal filtering

Component 3 retains a candidate KPI $i$ as causal evidence for target KPI
$j$ only if $i$'s past improves the prediction of $j$'s future beyond $j$'s
own past. For lag $\ell \in \{1, \dots, L\}$ the test contrasts the null and alternative

$$\mathcal{H}_0:\ Y_{j,t} \sim Y_{j,\,t-1:t-\ell},$$

$$\mathcal{H}_1:\ Y_{j,t} \sim Y_{j,\,t-1:t-\ell} + X_{i,\,t-1:t-\ell},$$

rejecting $\mathcal{H}_0$ by an SSR $F$-test at $p < 0.05$ under
Benjamini-Hochberg control across the tested dimensions. This is
temporal-precedence evidence, not intervention; Section 8 states the
consequence.

## 3.4 Calibration

Component 5 wraps the predictor in a distribution-free coverage guarantee.
Given a calibration set with known labels, the nonconformity score of point
$i$ is

$$s_i = 1 - \hat{P}\!\left(y_i \mid x_i\right),$$

and the $(1-\alpha)$-quantile of $\{s_i\}$, written $\hat{q}$, defines the
prediction set for a new point $x$,

$$C(x) = \bigl\{\, c : 1 - \hat{P}(c \mid x) \le \hat{q} \,\bigr\},$$

which satisfies the marginal coverage guarantee

$$\mathbb{P}\bigl(y \in C(x)\bigr) \ge 1 - \alpha.$$

A set of size two, $\{\textit{normal}, \textit{degrading}\}$, is the
system's explicit abstention: it signals that the evidence does not
separate the classes at the requested confidence and the case should be
escalated rather than resolved automatically.
