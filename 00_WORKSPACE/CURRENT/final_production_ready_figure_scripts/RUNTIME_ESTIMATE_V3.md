# Provisional v3 A100 runtime report

The only provenance-complete timing calibration currently present in the transferred
production tree is frozen P1.  Across its completed five-trajectory A100 shards, median
elapsed times are 28.2 s (`12x12`), 65.4 s (`16x16`), 174.5 s (`20x20`), 541.0 s
(`24x24`), 1493.5 s (`28x28`), and 3685.5 s (`32x32`).  The 226 transferred shards total
58.0 measured A100-hours.  These are useful kernel-scale anchors, but they are not direct
fixed-`Nx` timings for every v3 observer stack.

Until one shard from each v3 kernel class has run, budget **180–480 aggregate A100-hours**
for the 458–498 ordinary stochastic shards, plus **2–8 A100-hours** for the two H3
descendants.  The intentionally broad range covers the extra tangent, contour, replay, and
controller-frame work absent from simple P1 shards.

That corresponds to roughly **7.6–20.3 days on one continuously available A100**, or an
ideal compute floor of **1.8–4.9 hours on 100 A100s**.  Real 100-GPU elapsed time will be
longer because of scheduling, Drive/object-store traffic, startup, receipt verification,
and the M3 gate barrier.  For planning, use **3–8 hours** for a well-provisioned 100-A100
handoff, followed by the separately gated M3 wall array.

These are planning bounds, not promised runtimes.  The runner records synchronized child
elapsed time and rolling ETA; after the first shard of baseline, parent+tangent, P2,
S2/M2, M3, B1, and H3 completes, replace the corresponding bound with its measured rate.
