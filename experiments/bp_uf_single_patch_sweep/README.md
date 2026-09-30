# Fixed BP + UF distance extension

Extend the completed d=7 study to d=9, 11, and 13. At each distance use one
patch, SI1000 p=0.003, 4d noisy CZ rounds (36, 44, 52), zero yokes, and the
same ideal time boundaries and two logical observables as d=7.

All seven decoders see the same detector rows within each sample:
weighted UF, correlated UF, BP1/2/5/10 + weighted UF, and correlated MWPM.
BP uses the original sum-product flooding implementation with damping 0.5,
LLR clipping +/-30, double precision, and the original posterior-to-edge
projection. Every BP arm executes its entire fixed budget and finishes with
UF on the original syndrome. There is no adaptive stopping or budget choice.
Correlated MWPM enables correlations at both graph import and decode.

Run 10,000 pilot shots and 100,000 independently seeded confirmation shots
per distance. Use seed 202609290000 + 100*d + phase, with phase 1 for pilot,
2 for confirmation, 3 for validation, 4 for timing-row selection, and 5/6
for pilot/confirmation MWPM audits. Parameters and sample sizes remain fixed
throughout the sweep. Sample each phase once and save detector/truth arrays.
Serial timing uses 1,024 predetermined pilot rows shared by all decoders.

Queue distances sequentially and restrict the entire job to 32 distinct
physical cores. Native UF/BP collection uses 32 OpenMP threads across shots.
MWPM collection uses 32 spawned processes in a separate collection stage;
it does not overlap native collection. BP and UF within one shot are serial.

Reuse one BP trajectory per shot to obtain all four fixed budgets, charging
each arm full cumulative initialization and iteration time, its own weight
projection, and its UF pass. Serial timers separately record initialization,
BP rounds, projection, and UF growth/tree/peeling. Correlated UF totals charge
both necessary UF passes plus reweighting. Correlated MWPM records bound C++
decode and public API timing, including both correlation passes. Timing
excludes circuit/model setup and sampling. Native timers additionally exclude
Python orchestration, unpacking, physical audits, and some buffer destruction.
Fixed arm order/shared trajectories affect cache state; d=7 BP10 was timed
standalone, so precise cross-run ratios have that limitation.

Before collecting new distances, verify the new six-arm native wrapper
against saved d=7 deterministic results, including BP10. At each distance
verify 1-thread/32-thread agreement, independent Python BP and production UF,
selected physical corrections and logical masks, and every native collection
correction's syndrome. Audit 64 MWPM physical corrections per phase and
verify every serial timing prediction against collection. Halt on any failure.

Report either-observable failures per complete 4d-round memory shot, per-X/Z
failures, Wilson 95% intervals, paired repairs/regressions, paired differences
and intervals, and McNemar tests. Preserve BP5 versus correlated UF as the
original primary contrast; other comparisons are exploratory and unadjusted
for multiple testing. This extension was selected after the d=7 results.
BP comparisons include the projection change relative to prior-weight UF.

Source `env/workspace.sh`, activate `repos/yoked-surface-codes/.venv`, then run:

```
python experiments/bp_uf_single_patch_sweep/launch.py --reference /data2/s2chitni/projects/dante/results/bp_uf_single_patch_d7_p003_20260929T054325Z --tmux
```

The launcher copies the exact d=7 frozen dependencies and helper sources,
records source hashes and the inherited repository revision, and starts a
detached tmux session on the compute server. Launch outside any ephemeral
execution sandbox when the run must survive disconnecting the client. Its
socket, status command and server process ID are saved in `launch.json`.
Alternatively use `--wait` to keep a managed process supervisor alive.
Temporary builds, raw samples, logs,
and resumable chunks live under `$DANTE_SCRATCH/runs`; retained configuration,
sources, predictions, summaries, validation and reports live in `results/`.
`launch.json` records the exact runner command for resuming. Resuming checks
identities and reuses finished chunks. The runner writes live progress and
reports automatically, without requiring another chat turn.
