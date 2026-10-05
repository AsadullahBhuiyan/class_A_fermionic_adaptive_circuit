# Alpha_1=3: dissipation on every slab

2026-09-14 user-requested protocol replacement: do not freeze the exterior or
restrict evolution to the active slab. Run the channel and dephasing Lindblad
on the complete physical system, for both wall choices.

Launch four fresh processes:

```bash
python launch_alpha3_endpoints.py --launch --hard-exterior evolve --cpus 8,9,10,11
```

Nx=20, Ny=64, walls x=5,15, nshell=1, alpha_1=3, alpha_2=30, complex128,
perfect correction, 128 cycles/time units. Initialize the **entire system** as
maximally mixed. The hard case keeps support truncation but visits all centers
in the discrete channel and includes all OW jump modes in the Lindblad.
The soft case uses untruncated support and likewise evolves all centers.
All 2,560 physical modes evolve; no exterior preparation, frozen occupations,
or reduced-slab model is used. Channel schedules use the original matched root
seed 20260814; Lindblad uses unit gain/loss/dephasing rates and RK4 dt=0.05.

Fresh outputs and logs live under `results/alpha3_full_system_dissipation_v1_TIMESTAMP`.
The completed frozen-exterior results are preserved. Soft runs are repeated
unchanged scientifically, to provide a complete four-job collection. The
full-system hard evolution is the original channel/Lindblad protocol at alpha_1=3,
not the slab-only pure-trajectory protocol used for the separate heatmaps.

Full-system evolution removes frozen-exterior/twirl artifacts as a preparation
choice; it does not impose an occupation gap or guarantee the absence of modes
near one-half. Endpoint covariances, explicitly twirled ky spectra, spectral
checkpoints, cycle observables, source hashes, and completion checksums retain
the existing data format. In full-system runs `active_indices` contains every
physical mode and the two `frozen_exterior_*` arrays are empty.
