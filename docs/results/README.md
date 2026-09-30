# Retained experiment results

Keep protocols, source recipes, compact statistics, validation records, and final
figures here. Raw shots, compressed prediction arrays, oversized calibration
tables, and complete workspace runs have been moved to the local archive recorded
in [archive_index.json](archive_index.json). They were preserved, not deleted.

The index records each archived repository payload's original path, size, and
SHA-256. Restore a payload to its recorded original path when replaying an old
recipe that expects it there. Historical manifests remain unchanged; absolute
paths in them may refer to the pre-archive workspace locations.

Use `$DANTE_SCRATCH/runs` for new samples, builds, checkpoints, and per-shot arrays.
Publish only the final report, configuration, source identities, and compact
measurements in this directory.
