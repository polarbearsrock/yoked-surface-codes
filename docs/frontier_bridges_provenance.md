# Provenance of frontier-weighted UF and bounded bridges

Verified against the experiment code, contemporaneous literature notes, and
the primary UF paper on 2026-09-17.

The experiment named **Frontier-weighted UF + bounded bridges** combines two
locally specified modifications. It was not a reproduction of two separately
identified papers named “Frontier” and “Bridges.”

| Experimental component | Published background | What we implemented |
|---|---|---|
| Frontier-weighted growth | Delfosse and Nickerson, *Almost-linear time decoding algorithm for topological codes*, Section 5: weighted growth prioritizes the cluster with the smallest boundary. | Simultaneous growth of all active clusters, with each cluster's per-edge rate scaled by its frontier-edge count using a power of two. |
| Bounded bridges | The repository records no external paper as the source of this rule. Our missing-connection diagnostics motivated it. | Up to two rounds of mutually selected, cost-improving connections between stopped clusters, followed by minimum-cost correction on the resulting forest. |

The relevant paper is **Nicolas Delfosse and Naomi H. Nickerson,
“Almost-linear time decoding algorithm for topological codes,” Quantum 5,
595 (2021)**, first posted in 2017. See
[Section 5](https://arxiv.org/html/1709.06218v3#S5),
[arXiv:1709.06218](https://arxiv.org/abs/1709.06218), and
[DOI:10.22331/q-2021-12-02-595](https://doi.org/10.22331/q-2021-12-02-595).
Its motivation is that growing a larger boundary introduces more extraneous
edges. Its schedule selects the smallest boundary first; it does not specify
our simultaneous reciprocal-frontier rule. The paper tracks boundary
vertices, whereas our normalization counts frontier edges. The paper's
performance and complexity claims therefore do not establish those of our
variant.

The tested rules are defined by the
[frozen protocol](results/parallel_uf_clustering_d7_d9_p003_100k/README.md)
and [native implementation](results/parallel_uf_clustering_d7_d9_p003_100k/kernel.cc):

- **Frontier weighting:** for an active cluster with `b` frontier edges,
  each frontier edge receives growth at rate
  `1 / 2**ceil(log2(max(1, b)))` from that cluster. Contributions from both
  endpoint clusters add. Inactive clusters contribute zero. This slows
  individual edges of large-frontier clusters while allowing every active
  cluster to grow. The earlier diagnostic used exact `1/b`; the 100k-shot
  experiment uses the power-of-two approximation. This concerns growth
  scheduling, independently of the graph's existing probability weights.
- **Bridge candidates:** after growth stops, freeze the intercluster edges
  with at most **0.5 nats of growth remaining**. This is a residual-growth
  limit, not a limit on the edge's original weight.
- **Bridge scoring:** tree min-sum messages give each component's optimum
  correction cost `C_A` and the optimum `C_A(u)` with its endpoint syndrome
  toggled. For an edge `e=(u,v)` between components A and B, the cost change
  when using it is
  `delta(e) = w(e) + C_A(u) - C_A + C_B(v) - C_B`.
  Boundary parity constraints are free.
- **Bridge acceptance:** each component proposes its most negative eligible
  edge, breaking ties by edge ID. Accept only mutual proposals with
  `delta < -1e-9`. Accepted component pairs are disjoint within a round,
  preserving a forest. Recompute messages for the next round, with at most
  **two rounds**, then extract the minimum-cost forest correction.

These changes apply only to the second UF pass. All variants share the
production first-pass UF correction and the correlation weights it produces.
Actual sampled faults, logical truth, and MWPM predictions are not decoder
inputs. Earlier MWPM-guided missing-edge diagnoses motivated the autonomous
bridge rule; those oracle diagnoses are separate from the evaluated decoder.

The frontier rule aims to improve which connections UF reaches before its
clusters become valid. Bridges then admit a limited set of useful connections
that ordinary growth left unfinished. A lower correction cost does not
guarantee the correct logical class. The bridge search also excludes cycles
and combinations of individually unprofitable edges. Its local specification
does not constitute a claim of research novelty.

Parallel growth and disjoint bridge merges motivated these choices. The
experiment is a software simulation with floating-point graph weights;
FPGA/RTL latency, area, throughput, and fixed-point accuracy were not measured.

For provenance and results, see:

- [Literature review](../exa-results/uf-accuracy-2026-09-16.md): explicitly
  distinguishes frontier normalization from the published schedule and calls
  bounded extension across stopped clusters a separate hypothesis.
- [Growth and missing-connection diagnosis](results/correlated_uf_growth_diagnosis_d7_d9.md).
- [100k-shot results: d=7/9, p=0.003](results/parallel_uf_clustering_d7_d9_p003_100k.md).
- [100k-shot results: d=11/13, p=0.003](results/parallel_uf_clustering_d11_d13_p003_100k.md).
- [100k-shot results: d=7/9/11/13, p=0.001](results/parallel_uf_clustering_p001_100k.md).

The literature and initial diagnosis were recorded in commit
`e94dfe1dbfa45fee92fb837fb294b8a8ebfc40fe`; the frozen frontier/bridge
experiments were recorded in `9efa549e27040260b56d476689badaec4fb85421`.

Suggested wording for a report:

> We evaluated a parallel frontier-normalized UF growth heuristic inspired by
> the smallest-boundary-first weighted-growth UF of Delfosse and Nickerson
> (2021, Section 5). We combined it with a locally specified bounded
> intercluster bridge refinement motivated by our UF failure diagnostics.
> The implemented growth schedule differs from the published schedule; the
> bridge rule is defined in our experimental protocol.

BibTeX for the published background:

```bibtex
@article{delfosse2021almostlinear,
  author = {Delfosse, Nicolas and Nickerson, Naomi H.},
  title = {Almost-linear time decoding algorithm for topological codes},
  journal = {Quantum},
  volume = {5},
  pages = {595},
  year = {2021},
  doi = {10.22331/q-2021-12-02-595},
  eprint = {1709.06218},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph}
}
```
