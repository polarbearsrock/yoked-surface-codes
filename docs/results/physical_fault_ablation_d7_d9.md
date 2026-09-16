# Physical faults behind the correlated-UF / correlated-MWPM gap

**CZ gate faults and data-qubit errors accumulated while waiting for measurement
or reset have the largest removal effects in this experiment.** Their relative
order is uncertain. Within CZ faults, the largest group effect comes from Pauli
errors affecting both the data qubit and its check ancilla. There is no consistent
single-Pauli winner across the two distances.

These conclusions come from a new, stratified physical-fault ablation of the
existing SI1000 `p=0.003`, six-patch, two-ideal-yoke, `rounds=4d` experiment.
They are sensitivities of the current decoders to removing whole fault families,
not an additive allocation of logical failures to individual physical events.

## What was measured

At each distance, select 256 shots uniformly without replacement from each
original outcome: both decoders correct, UF alone fails, MWPM alone fails, and
both fail. This gives **1,024 analyzed shots per distance**, sampled from the
original 100,000. Selections were saved before inspecting physical faults.

Recover the actual noise draws, remove all events in one family, update both
the syndrome and the true logical labels, and rerun both full correlated
decoders. Both retain the original DEM and recompute their own correlation
evidence. No noise model is retuned and no production decoder is changed.

The primary metric is the **joint-shot failure-rate gap**,
`G = P(correlated UF fails) - P(correlated MWPM fails)`. It is 9.445 percentage
points at d=7 and 7.322 points at d=9 in the full saved population. Stratum
estimates are weighted by their original population counts. Equal sample sizes
are not treated as equal population frequencies.

| Physical family removed | Reduction in gap, d=7 | Reduction in gap, d=9 |
|---|---:|---:|
| All CZ gate faults | **98.3%** | **95.4%** |
| Extra data depolarization while waiting for measurement/reset | **90.5%** | **98.1%** |
| Ancilla readout flips | 65.5% | 80.8% |
| Reset errors | 27.6% | 35.8% |
| Other single-qubit data depolarization | 31.1% | 19.1% |
| Single-qubit ancilla depolarization | 4.3% | 5.0% |

These are point estimates, not percentages of failures exclusively attributable
to each row. The effects overlap substantially and must not be added.

The physical channels are identified directly in the sampled circuit:
`DEPOLARIZE2(0.003)` after CZ; extra `DEPOLARIZE1(0.006)` on waiting data;
`M(0.015)` readout flips; `X_ERROR(0.006)` after ancilla reset; and other
single-qubit depolarization at `0.0003`. Ancilla post-measurement depolarization
at `0.003` is included in the final row, but its recovered detector and
observable response is zero on every selected shot: the following reset
erases its effect in this circuit.

Readout flips are more numerous than either leading family. The estimated mean
event counts per original shot are 121.7 readout, 98.8 waiting-data, and 84.3 CZ
events at d=7; at d=9 they are 259.5, 209.8, and 186.0. Counting physical faults
alone would therefore give a different ranking from their removal effects.

## Which CZ faults, and which Paulis?

A two-qubit Pauli product is one physical event. Classify its support using the
actual data/ancilla roles, independently of target order in the circuit.

| CZ subfamily removed | Reduction in gap, d=7 | Reduction in gap, d=9 |
|---|---:|---:|
| Nonidentity Pauli on both data and ancilla | **79.5%** | **93.6%** |
| Nonidentity on data only | 22.1% | 56.7% |
| Nonidentity on ancilla only | 22.5% | 10.4% |

The first row includes such events as `XX`, `YX`, and `ZZ`, with data listed
first. It also contains more events: about 50.7 versus 16.7/16.9 per shot at
d=7, and 112.1 versus 37.0/36.9 at d=9. This establishes the largest *group
removal effect*; it does not establish greater harmfulness per individual event.

| Pauli-specific family removed | Reduction in gap, d=7 | Reduction in gap, d=9 |
|---|---:|---:|
| CZ product containing at least one Y | 72.7% | 85.8% |
| CZ product containing no Y | 57.2% | 89.6% |
| Waiting-data X | 42.4% | 67.7% |
| Waiting-data Y | 64.7% | 55.0% |
| Waiting-data Z | 48.8% | 50.6% |

The Y-containing versus Y-free CZ ordering reverses between distances, as does
the largest waiting-data Pauli estimate. Paired intervals do not establish
Y-containing CZ faults as more important than Y-free CZ faults at either
distance. The observations do not justify naming Y, or one specific two-qubit
Pauli product, as the unique leading culprit.

## The failing shots usually involve both leading families

Consider only the 256 sampled shots per distance where original UF fails and
original MWPM succeeds. The following counts require **both** decoders to
succeed after removal:

| Family removed | d=7 repaired / 256 | d=9 repaired / 256 |
|---|---:|---:|
| CZ faults | 253 | 256 |
| Waiting-data errors | 244 | 253 |
| Readout flips | 207 | 223 |

In **241/256 d=7 cases and 253/256 d=9 cases**, either deleting all CZ faults
or deleting all waiting-data errors, in separate interventions, repairs the
same case. This is evidence for a mixed-fault problem. It does not identify a
particular local pair or imply that deleting any one constituent fault suffices.

The earlier [physical-fault investigation](correlated_uf_mwpm_failure_analysis_d7_d9.md)
provides a concrete example of the interaction mechanism. In d=7 shot 44875,
a data-Y/ancilla-X CZ fault and a later readout flip cancel at one detector.
The final UF partition omits a connection used by MWPM, and UF assigns the
logical error to the wrong patches. Removing the readout event repairs UF.
That example is illustrative, not a randomly selected population-frequency
measurement. The ablation above supplies the broader family comparison.

Together with the [growth diagnosis](correlated_uf_growth_diagnosis_d7_d9.md),
the evidence points to the current UF growth and stopping decisions on
multi-fault syndromes. It does not establish that these physical events are
intrinsically uncorrectable by UF, or that correlation reweighting never works.

## Actual LER changes

Removing a leading family makes the problem substantially easier for both
decoders. The gap reduction is not an improvement to the original decoder at
the original physical noise level. Using the preceding experiment's normalized
LER convention, `sinter.shot_error_rate_to_piece_error_rate` with
`pieces=6*rounds, values=8`:

| Intervention | UF LER, d=7 | MWPM LER, d=7 | UF LER, d=9 | MWPM LER, d=9 |
|---|---:|---:|---:|---:|
| Original, full 100k records | 0.0015990 | 0.00089079 | 0.00071825 | 0.00033368 |
| Remove CZ faults, estimated | 0.000041989 | 0.000032383 | 0.000018111 | 0.0000025643 |
| Remove waiting-data errors, estimated | 0.000090241 | 0.000036162 | 0.0000085929 | 0.0000021074 |
| Remove readout flips, estimated | 0.00050608 | 0.00029688 | 0.00016833 | 0.00010120 |

## Uncertainty and interpretation

This is a diagnostic sample, not a complete 100k-shot ablation. The archived
summaries include 20,000 paired stratified bootstrap replicates. In particular,
the CZ-versus-waiting-data ordering is unresolved at both distances.

Bootstrap intervals can be too narrow for the very small residual failure
rates, because they cannot generate outcomes absent from the sampled stratum.
An [independent audit](physical_fault_ablation_d7_d9/audit.py) therefore also
inverts the exact finite-population hypergeometric distribution. Simultaneous
bounds on the two disagreement probabilities in each of four strata yield a
conservative 95% interval for each variant's gap reduction:

| Leading family | d=7, fraction of gap removed | d=9, fraction of gap removed |
|---|---:|---:|
| CZ | 98.3%, conservative interval 66.1–131.1% | 95.4%, conservative interval 59.4–130.7% |
| Waiting-data | 90.5%, conservative interval 55.1–125.9% | 98.1%, conservative interval 65.5–130.8% |

An interval extending beyond 100% allows a reversal of the residual decoder
ordering; it does not mean a negative failure probability. These are marginal
intervals per variant, not simultaneous bounds across all tested variants.
The intervals describe subsampling uncertainty in the fixed saved population,
not uncertainty from generating a new experiment.

Whole-family removals change the total noise burden, and different families
contain different event counts. Decoding with the original DEM is deliberate
for this sensitivity analysis, but does not estimate the best retuned decoder
under a modified physical noise model. A per-event causal ranking would require
additional individual-fault interventions.

## Validation and artifacts

The observer reproduced **all 100,000 original detector and observable records
at each distance bit for bit**. The selected shots contained 417,822 physical
events at d=7 and 893,159 at d=9. For every selected shot, both the all-fault
response and the XOR of the family responses matched its original detector and
observable values; the ideal control was zero.

Independent ordinary-Stim forced circuits checked the original and each of the
six primary interventions on one shot from every outcome stratum, at each
distance, using two seeds and two samples per seed. All original decoder
predictions were reproduced on all 2,048 selected shots. The archive audit also
checks decoded payload hashes, baseline labels, stratum outcome counts, and
population-weighted estimates.

The [analysis directory](physical_fault_ablation_d7_d9/) contains the protocol,
selected row IDs, compact decoded outcomes, event counts, source hashes,
validation records, and summaries. The [runner](physical_fault_ablation_d7_d9/analyze.py)
and [observer source](physical_fault_ablation_d7_d9/trace_cases.cc) are included.
The raw logs and larger response arrays are retained at:

```
$TMPDIR/physical-fault-ablation-d7-d9-3r3wpmuq/analysis
```

Use the [README](physical_fault_ablation_d7_d9/README.md) for the run stages.
Primary decoding used 48 workers at d=7 and 64 at d=9; the Pauli subfamily runs
used the same worker counts. Selection seeds were `2026091707` and `2026091709`.
