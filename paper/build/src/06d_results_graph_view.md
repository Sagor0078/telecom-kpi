## 6.8 Component 2, graph view \-- illustrative topology-based ranking

Neither SMD nor the reconstructed TelecomTS sessions provide a
network-element topology, so this experiment is explicitly illustrative: a
synthetic Core $\rightarrow$ Aggregation $\rightarrow$ BaseStation $\rightarrow$ Cell graph with an injected
root cause at Aggregation and severity decaying (with noise) along
descendants. Personalized PageRank on the reversed graph, seeded by observed
node severity, ranks Aggregation first (score 0.319), correctly recovering
the injected root cause and demonstrating the topology-propagation mechanism
that a learned graph neural network would generalize with data-driven edge
weights. It demonstrates a mechanism, not a validated capability.
