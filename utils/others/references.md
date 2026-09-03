# Related Research — Annotated Bibliography

Supporting literature for `Telecom_Trustworthy_AI_Framework.ipynb` (Question 1: *Trustworthy prediction and root-cause identification*). Grouped by the modelling family each paper informs. Links marked (arXiv) were located and confirmed via live web search on 2026-08-28; a few foundational works are cited by author/venue only because this session could not re-verify a specific link.

## 1. Classical ML for network/KPI anomaly detection

- **"Machine Learning-Based Network Anomaly Detection: Design, Implementation, and Evaluation"**, MDPI *Telecom* 5(4):143, 2024. https://www.mdpi.com/2673-2688/5/4/143 — benchmarks tree-based and classical ML models (Isolation Forest, Random Forest, gradient boosting) for network KPI anomaly detection; useful as the "simple, auditable baseline" tier of the framework.
- Lundberg, S. & Lee, S. "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017 — the attribution method used throughout the framework's explainability component.

## 2. Deep learning for multivariate KPI time series

- Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., Pei, D. **"Robust Anomaly Detection for Multivariate Time Series via Stochastic Recurrent Neural Network"** (OmniAnomaly), KDD 2019 — introduces the **Server Machine Dataset (SMD)** used as the public dataset in this notebook, and a VAE+RNN approach to multivariate KPI anomaly detection.
- Hundman, K. et al. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (LSTM-NDT), KDD 2018 — sequence-to-sequence LSTM forecasting + dynamic thresholding for early-warning style anomaly flags, the direct ancestor of the "predict degradation before disruption" component.
- Deng, A. & Hooi, B. "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series" (GDN), AAAI 2021 — combines a learned sensor-dependency graph with attention-based forecasting and per-sensor deviation scores, bridging the deep-learning and GNN families below.

## 3. Transformer-based approaches

- Wen, Q. et al. **"Transformers in Time Series: A Survey"**, arXiv:2202.07125. https://arxiv.org/pdf/2202.07125 — taxonomy of transformer variants (positional encoding, attention modules) for forecasting/anomaly/classification; used to justify the encoder choice in Component 1.
- Tuli, S., Casale, G., Jennings, N. **"TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data"**, VLDB 2022, arXiv:2201.07284. https://arxiv.org/abs/2201.07284 — transformer encoder with adversarial (two-phase) training and self-conditioning; strong reference for the forecasting/anomaly-scoring backbone.
- **"Root Cause Analysis of Anomalies in 5G RAN Using Graph Neural Network and Transformer"**, arXiv:2406.15638. https://arxiv.org/html/2406.15638v1 — directly telecom-RAN, combines a transformer encoder with a GNN over cell/site topology for anomaly root-causing; closest published analogue to the framework proposed here.
- **"Classification of Anomalies in Telecommunication Network KPI Time Series"**, arXiv:2308.16279. https://arxiv.org/pdf/2308.16279 — taxonomy of KPI anomaly types in real operator data (point, contextual, collective) that motivates the labeling scheme used in Component 1/2.
- **"Time Series Network Utilization KPI Forecasting Using Advanced AI/ML Models"**, arXiv:2607.19974. https://arxiv.org/html/2607.19974 — forecasting utilization-type KPIs with modern AI/ML models; supports the choice of probabilistic forecasting for the prediction component.

## 4. Graph Neural Networks for topology-aware root cause analysis

- Wu, L. et al. "MicroRCA: Root Cause Localization of Performance Issues in Microservices", IFIP/IEEE IM 2020 — builds an attributed service-dependency graph and uses Personalized PageRank to rank root-cause candidates; the pattern reused (with a cell/NE topology graph) in Component 2.
- **"Cellular Network Fault Diagnosis Method Based on a Graph Convolutional Neural Network"**, PMC10459609. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10459609/ — GCN over cell-site topology for telecom fault diagnosis, with an LSTM front end for temporal priors.
- **"CHASE: A Causal Hypergraph based Framework for Root Cause Analysis in Multimodal Microservice Systems"**, arXiv:2406.19711. https://arxiv.org/pdf/2406.19711 — hypergraph extension for multimodal (metrics+logs+traces) causal RCA.
- **"A Comprehensive Survey on Root Cause Analysis in (Micro) Services: Methodologies, Challenges, and Trends"**, arXiv:2408.00803. https://arxiv.org/html/2408.00803v1 — survey covering rule-based, ML, GNN and causal RCA methods; used to cross-check that the framework's component 2 covers the mainstream method families.

## 5. Causal reasoning: separating correlation from causal evidence

- Spirtes, P., Glymour, C., Scheines, R. *Causation, Prediction, and Search* — origin of the PC/FCI constraint-based causal discovery algorithms used in Component 3.
- **"Root Cause Analysis In Microservice Using Neural Granger Causal Discovery"** (RUN), arXiv:2402.01140. https://arxiv.org/abs/2402.01140 — neural Granger causal discovery + contrastive learning to build a causal graph from monitoring metrics; direct template for the Granger-causality step used in this notebook.
- **"Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?"**, arXiv:2408.13729. https://arxiv.org/html/2408.13729v1 — empirically compares PC, FCI, Granger, fGES, LiNGAM as causal-graph builders for RCA; informs the choice/limits of Granger causality used here.
- **"Graphical Causal Reasoning for Root Cause Analysis in Cloud Networks"**, arXiv:2606.13532. https://arxiv.org/html/2606.13532 — constructs a causal graph from binary event time series via bivariate Granger causality + conditional-independence tests, then walks the graph to rank causes — the same two-step (build graph → rank) pattern as Component 2+3 here.

## 6. LLM-based approaches for diagnosis / operator-facing explanation

- **"Exploring LLM-based Agents for Root Cause Analysis"**, arXiv:2403.04123. https://arxiv.org/pdf/2403.04123 — autonomous LLM agents with tool use for cloud incident RCA (context management, trajectory stabilization); the design pattern used for the "explanation-generation" LLM in Component 4 (tool-grounded, not free-generation).
- **"RCA Copilot: Transforming Network Data into Actionable Insights via Large Language Models"**, arXiv:2507.03224. https://arxiv.org/pdf/2507.03224 — LLM pipeline that aggregates multi-modal diagnostic evidence and generates root-cause narratives for network incidents; closest analogue to Component 4's report generator.
- **"Automatic Root Cause Analysis via Large Language Models for Cloud Incidents"**, arXiv:2305.15778. https://arxiv.org/pdf/2305.15778 — LLM reads retrieved telemetry/runbooks to propose root causes; motivates keeping the LLM strictly as a *narrator over verified evidence* rather than an independent diagnoser, to avoid hallucinated causes.

## 7. Uncertainty quantification & trustworthy AI

- Angelopoulos, A. & Bates, S. **"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"**, arXiv:2107.07511. https://arxiv.org/abs/2107.07511 — the conformal-prediction method used for Component 5 (distribution-free coverage guarantees, no distributional assumptions on KPI noise).
- **"Conformal Prediction and Trustworthy AI"**, arXiv:2508.06885. https://arxiv.org/abs/2508.06885 — reviews why calibrated, guaranteed-coverage uncertainty (vs. bare softmax confidence) is a prerequisite for trustworthy deployment.
- **"Uncertainty Aware Deep Learning Model for Secure and Trustworthy Channel Estimation in 5G Networks"**, arXiv:2305.02741. https://arxiv.org/pdf/2305.02741 — a 5G-specific example of separating epistemic/aleatoric uncertainty in a deep model, mirroring the split used in Component 5.

## Dataset

- Su, Y. et al. (2019), *Server Machine Dataset (SMD)*, https://github.com/NetManAIOps/OmniAnomaly — 28 machines × 38 metrics (CPU, memory, network, load metrics sampled every minute), with anomaly-interval labels **and** ground-truth root-cause dimension labels (`interpretation_label`). Used in the notebook as a public, license-open proxy for telecom NE/CPU-memory/traffic KPI monitoring, chosen specifically because it is one of the few public datasets with root-cause ground truth, which lets the notebook *quantitatively* validate the root-cause component rather than only the anomaly-detection component. True RAN-level KPI datasets (latency/jitter/handover-failure with root-cause labels) are held by operators/vendors and are not public.
