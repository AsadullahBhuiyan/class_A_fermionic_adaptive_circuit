# Purification v2 quarantine

`07_maxmix_hard_soft_purification_v2_exact.tar.gz` is the exact repository
bundle that produced the server-visible v2 checkpoints on 2026-09-06.  It is
retained as a compressed artifact so the repository-wide canonical GPU-source
sync rule can remain true for live `.py` copies without mutating the executed
v2 source.

- archive SHA-256: `77cfe0981480182c1034e45e20f7e83eab805e8fcb470f0f94706d9c6e8bd15e`
- v2 sampling revision:
  `maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2`
- v2 configuration SHA-256:
  `2dc0ba9a2a3ec8cc79eebec19bda0a6bcbaf8f3efcd7b47da99eab8decbcffce`
- executed canonical GPU source SHA-256:
  `53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700`

The corrected v3 runner accepts a v2 checkpoint only after matching the full
locked identity above, the other executed source hashes, task/sample identity,
byte count, payload schema, and checksum.  The v2 Drive output is never edited
or deleted by that conversion.
