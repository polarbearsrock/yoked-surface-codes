# Retained experiment results

Keep protocols, source recipes, compact statistics, validation records, and final
figures here. Raw shots, compressed prediction arrays, oversized calibration
tables, and redundant workspace runs were permanently deleted at the user's
request after their compact reports were retained. The deletion inventory is in
[archive_index.json](archive_index.json). The d=9 sample and paired reference
predictions needed for the current min-sum comparison remain in the reference
directory recorded there.

The index records each removed repository payload's original path, size, and
SHA-256. Historical manifests remain unchanged; paths in them may refer to removed
payloads. Reproduce old runs from their recorded recipes, source revisions,
configurations and seeds when raw data is needed again.

Use `$DANTE_SCRATCH/runs` for new samples, builds, checkpoints, and per-shot arrays.
Publish only the final report, configuration, source identities, and compact
measurements in this directory.
