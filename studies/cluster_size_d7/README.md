# Cluster-size / predecoder study scripts (d = 7 and d = 9, SI1000 p = 0.003)

Non-claim-bearing study scripts copied verbatim from the git-ignored `out/cluster_size_study_d7/`
directory, where they were developed and run. Data (`cell.npz`, replays, JSON results) and the
generated pages stay under `out/cluster_size_study_d7/<cell>/` and are not tracked.

Pipeline (`run_cell_pipeline.sh CELL D ROUNDS P SEED SHOTS PAGE`, every stage <= 32 processes, resumable):
`collect_cell.py` -> `frontier_families.py` -> `l2_timing.py` -> `microblossom_cycles.py` -> `zerog_cycles.py`
-> `replay_closing_times.py` -> `helios_quantum.py` -> `cap_ladder.py` -> `replay_event_times.py`
-> `helios_budget_cycles.py` -> `build_talk_page.py`; plus `workload_counts.py` and `build_talk_page_v2.py`
for the reviewed page, `presentation_figures.py` for static figures, `astrea_hw.py`, `paper_ler.py`,
`frontier_sweep.py`, `l2_structure.py`, `l2_validate.py`, and the earlier page builders.

Run from the repository root with the venv, `PYTHONPATH=src`, and the six `*_NUM_THREADS=1` variables exported.
