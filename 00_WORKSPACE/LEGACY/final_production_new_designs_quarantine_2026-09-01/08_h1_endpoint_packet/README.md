# 08_h1_endpoint_packet

This is the standalone H1-v4 chirality campaign. It replaces the interpretation of the
completed `03_h1_modular_response` experiment without changing or deleting that campaign.
The primary observable is the legacy-validated localized packet launched separately from
both entanglement endpoints at each physical wall. It contains no retarded-response,
static-susceptibility, or Fourier-ridge product.

The physical matrix is unchanged: `Nx=20`, `Ny=40`, `n_shell=1`, 80 cycles, observation cycles
`40,48,56,64,72,80`, hard and soft walls, topological `alpha_1=1`, matched trivial
`alpha_1=3`, and 25 independent Born trajectories in five immutable five-trajectory
shards. Every observation cycle evaluates all 40 translated half-cylinder cuts. Each cut launches
four endpoint packets for each source width: two physical walls times two entanglement
endpoints. The actual wall coordinates come from the instantiated model's `DW_loc`.

For wall `w`, the two conditional endpoint centers define

`D_w(t) = [ybar_(w,lower)(t) + ybar_(w,upper)(t) - (Ay-1)] / 2`.

The exact `Nx=20, Ny=40` benchmark fixes the wall orientation signs to `[-1,+1]`. The
primary handedness is the oriented fixed-time displacement at modular time 2; oriented
fit velocity is secondary. Cuts are reduced inside each trajectory before whole
trajectories are bootstrapped. The final observation cycle is primary and the earlier five are
stationarity diagnostics.

H1-v4 retains the H1-v3 numerical contract without changing the physical trajectories or
handedness estimator. For every modular time and localized source, the observer first
records the raw deviation of total packet probability from one, then enforces the exact
unit-norm identity before computing retention, centers, handedness, and velocity. A raw
error above `1e-8` is retained as a non-gating warning and the queue continues. Only a
non-finite or non-positive norm, a raw error above the hard `1e-6` ceiling, or a
post-normalization residual above `1e-12` is a hard numerical failure. Conditional
eigenvector-Gram diagnostics and the full raw-error argmax are saved. The locked
`complex128` frame and `float64` probability dtypes are also enforced. Every completed,
finite shard that crosses a hard tolerance is archived under `_failed_qualifications` before
qualification is rejected, so a late numerical gate never discards a completed diagnostic
shard. A structurally unusable non-finite packet still aborts immediately. The superseded
H1-v2 output collection remains immutable.

Resume identity is strict over every bundled Python source, not just the configuration and
engine filename. A changed observer, runner, helper, or canonical GPU engine makes an old
archive and A100 qualification stale instead of silently accepting self-consistent old code.

When checksum, engine hash, physical case, and RNG-state checks all pass, the runner
replays the completed response campaign's ordered Born records so the new estimator is
evaluated on the same trajectories. Otherwise it generates a fresh trajectory with the
same preregistered seed. The manifest records which route was used and verifies any replay
against the source record.

Upload the clean two-campaign deployment to `MyDrive/final_production_new_designs_v4`.
No legacy-design source folder or unused sibling bundle is required. The v3 output folder
is read only as data: a pinned migration ledger requires all 12 current archive files and
their receipts to be server-visible and checksum-consistent, rejects the eight receipt-only
entries, credits 11 compatible case/shard slots, and forces the soft/topological shard-zero
qualification to run again under v4. Production therefore computes nine H1-v4 shards and
analysis merges them with the 11 reusable v3 shards. A warning-level raw norm residual is
printed and recorded but does not stop subsequent shards.

Each v4 archive is staged under local `/content`, uploaded resumably, and checked against
the Drive server's parent, filename, byte count, and SHA-256 before its receipt is published.
The parent runner independently repeats that check before advancing the queue. DriveFS is
only a cache and cannot make a shard complete. The notebook obtains one Google authorization
per fresh runtime, checks account-wide quota, and runs a small remote-commit probe before work.

`production_config.json` is immutable run intent. `src/source_manifest.json` records the
canonical engine and helper hashes. Never hand-edit copied files in `src`; update their
repository sources and run `_maintenance/sync_bundle_sources.py`.
