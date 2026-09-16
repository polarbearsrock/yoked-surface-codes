# Papers that improve union-find decoding accuracy

The strongest starting points for our experiment are changes to UF's growth
schedule, belief-find, and union-intersection UF. More recent work also explores
local refinement after UF and ensembles of UF forests. The relevant distinction
is whether a method can change the clusters or expose connections between them.

This is a targeted literature review, not a reproduction of published results.
The search retrieved 55 results across growth/refinement, probabilistic
preprocessing, and ensemble/hardware directions, representing 47 distinct URLs
before consolidating different versions of the same paper. Primary paper pages
and full texts support the summaries below. Search date: 2026-09-16.

## Direct algorithmic precedents

### Grow smaller frontiers first

Nicolas Delfosse and Naomi H. Nickerson, **Almost-linear time decoding algorithm
for topological codes**, Quantum 5, 595 (2021), initially posted in 2017.
[Paper, Section 5](https://arxiv.org/html/1709.06218v3).

Their weighted-growth variant prioritizes the cluster with the smallest
frontier, rather than growing every odd cluster uniformly. They report toric
code threshold changes from 9.2% to 9.9% for phase errors with perfect syndrome
measurements, and from 2.4% to 2.6% with faulty syndrome measurements. The
implementation preserves almost-linear complexity.

**Relevance to our code:** this is a scheduling choice, distinct from edge
probability weights and from union-by-size in the disjoint-set data structure.
Our decoder currently contributes unit growth from each active endpoint
cluster. A small-frontier-first variant could change which connections complete
before a yoke-containing cluster stops. Our prior frontier-normalized diagnostic
is related in motivation but is not an implementation of their schedule.

### Weight graph edges using the noise model

Shilin Huang, Michael Newman, and Kenneth R. Brown, **Fault-tolerant weighted
union-find decoding on the toric code**, Physical Review A 102, 012419 (2020).
[Paper](https://arxiv.org/abs/2004.04693).

Under their circuit-level depolarizing model, edge weighting raises the UF
threshold from 0.38% to 0.62%; weighted matching reaches 0.72%. This is a useful
baseline demonstration of how much noise-aware growth can improve UF, while
leaving an accuracy gap to matching. Our implementation already has weighted
edges and correlation reweighting, so this is not by itself a new intervention
for the current experiment.

### Belief propagation followed by weighted UF: belief-find

Oscar Higgott, Thomas C. Bohdanowicz, Aleksander Kubica, Steven T. Flammia, and
Earl T. Campbell, **Improved decoding of circuit noise and fragile boundaries
of tailored surface codes**, Physical Review X 13, 031007 (2023).
[Paper](https://arxiv.org/abs/2203.04948),
[full text](https://arxiv.org/pdf/2203.04948).

BP operates on the circuit fault model. If it fails to return a syndrome-valid
answer, its posterior information supplies weights for a weighted UF solve.
Their belief-find and belief-matching thresholds are about 0.94%, compared
with 0.82% for standard MWPM in the same circuit-noise study. They report very
similar accuracy for the two hybrid decoders.

This is stronger probabilistic preprocessing than conditioning weights on one
hard UF correction. It can change the eventual partition. BP also adds message
memory and repeated updates; the paper explicitly notes that its linear-time
stage can be computationally intensive. Almost-linear asymptotic complexity
does not establish lower latency than modern MWPM on our workload.

### Share cluster information across X and Z: UIUF

Tzu-Hao Lin and Ching-Yi Lai, **Union-Intersection Union-Find for Decoding
Depolarizing Errors in Topological Codes** (2025 preprint).
[Paper, Sections III–V](https://arxiv.org/html/2506.14745v1).

The first stage builds valid X and Z clusters. Physical-qubit locations covered
in both sectors become erasures for subsequent UF decoding. Thus the method
uses overlapping cluster support instead of trusting only one peeled correction.
The authors report more than an order-of-magnitude LER improvement over UF in
some tested regimes, with an additional UF-scale pass and an error/erasure
distance guarantee. They also explain why naive iterative hard-correction
feedback can lose that guarantee.

The evaluations use code-capacity, phenomenological, and biased noise models.
They do not establish the improvement for our circuit-level SI1000 model with
yokes. Adapting the qubit-level X/Z intersection to circuit faults and DEM
components requires care. Nevertheless, this is a particularly relevant
candidate when retaining UF's basic operations is a priority.

### Preserve alternatives and refine locally: BP + UF + BP

Xingyu Qiao et al., **Fault-Tolerant Hybrid Decoder for Quantum Surface Codes
on Probabilistic Inference and Topological Clustering**, Applied Sciences
16(5), 2586 (2026).
[Paper, Sections 3.3–4.1](https://www.mdpi.com/2076-3417/16/5/2586).

Global BP supplies soft weights; a modified clustering stage keeps selected
cycle-forming edges; local BP then refines the retained fault subgraph.
The authors report improved LER on circuit-noise rotated surface codes and
a threshold around 0.72%. Their experiments use 30 global and 15 local BP
iterations. Their MWPM comparison explicitly excludes correlated matching.

This is a close architectural precedent for preserving competing corrections
instead of immediately reducing everything to one forest. Its extra edges are
inside the clusters selected by the modified growth process. Applying only
that internal augmentation to our existing failed partitions would not evade
our logical-space certificate. The complete BP-driven procedure can produce
different partitions, so that limitation does not rule out the full method.

### Sample several forests: coset ensemble decoding

Shuang Liang et al., **Coset Ensemble Decoder for Quantum Error Correction
with Algorithm–Hardware Co-Design** (2026; manuscript states acceptance at
ISCA 2026).
[Paper, Sections III-A and VI-A](https://arxiv.org/html/2606.11076).

The decoder generates forests using different priorities, extracts candidate
corrections, and votes over logical outcomes among candidates with the smallest
correction size. It includes FPGA architecture and synthesis results. With
24 candidates under circuit depolarizing noise, the paper reports better
accuracy than its UF baseline, but a remaining MWPM gap that grows with distance.
At p=0.002 the reported LER ratio to MWPM reaches about 2.1 at d=19.

The paper explicitly limits its optimization to the solution space allowed by
clustering. Random-priority frequencies are not, by themselves, a demonstration
of exact physical posterior sampling. For our 256 diagnosed failures, any
ensemble restricted to the same final partitions would retain the same wrong
logical prediction. It would need different growth, expanded support, or both.

## Related work with a different role

- **Hardware with adaptive noise information:** Abbas B. Ziad et al.,
  [Local clustering decoder as a fast and adaptive hardware decoder for the
  surface code](https://www.nature.com/articles/s41467-025-66773-x), Nature
  Communications (2025). A UF-based FPGA decoder adapts to control signals such
  as heralded leakage. The reported accuracy/resource benefits concern a
  leakage-dominated circuit model. This is evidence that adaptive clustering
  can work in hardware, rather than evidence of the same gain for our
  leakage-free SI1000 comparison.
- **Confidence from additional growth:** Kaito Kishi et al.,
  [Even More Efficient Soft-Output Decoding with Extra-Cluster Growth and Early
  Stopping](https://arxiv.org/html/2602.03336v1) (2026). Bounded cluster gaps and
  extra-cluster growth estimate reliability for decoder switching or other
  downstream uses. This is closely related to our earlier confidence work,
  but computing a better soft output does not itself repair UF's hard answer.
- **Local neural preprocessing:** Christopher Chamberland et al.,
  [Techniques for combining fast local decoders with global decoders under
  circuit-level noise](https://arxiv.org/abs/2208.01178), Quantum Science and
  Technology 8, 045011 (2023). Local convolutional decoding reduces syndrome
  density before a global decoder and the paper considers FPGA costs. This
  adds a trained front end, a different implementation tradeoff from changing
  UF growth or performing another UF pass.
- **Why UF differs from matching:** Yue Wu, Namitha Liyanage, and Lin Zhong,
  [An interpretation of Union-Find Decoder on Weighted Graphs](https://arxiv.org/abs/2211.03288)
  (2022). This develops the interpretation of UF as an approximation to the
  blossom algorithm and motivates weighted graph variants. It is a useful
  conceptual companion to our growth traces.

## What these papers suggest for our next comparison

My assessment is to test a changed growth schedule first, keeping the present
correlation weights, because it isolates the algorithmic issue at low added
complexity. UIUF provides a second UF-centered design to study. Belief-find is
a well-supported accuracy reference for more sophisticated preprocessing, with
its BP cost measured explicitly. A bounded extension across stopped clusters
remains a separate hypothesis supported by our own traces; the papers above
do not demonstrate its LER on our workload.

Published gains are relative to each paper's baselines and noise models. They
cannot be transferred numerically to our correlated-UF versus correlated-MWPM
comparison. Our [growth diagnosis](../docs/results/correlated_uf_growth_diagnosis_d7_d9.md)
also shows why changing only the correction within an unchanged partition
cannot repair the sampled logical failures.
