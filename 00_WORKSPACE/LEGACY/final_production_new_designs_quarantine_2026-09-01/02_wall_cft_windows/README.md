# 02_wall_cft_windows

This standalone, versioned bundle compares the two programmable domain-wall constructions and saves only reusable raw CFT-window data. It does not modify or replace the existing W1, S1, P1, or Gate-2 archives.

The ten cases are `Nx=20`, `Ny=20,30,40,50,60`, crossed with a hard support-terminated wall (`dw_truncation=true`, `meas_slab_only=true`) and a soft explicit interface (both flags false). Each case has 25 independent Haar-random half-filled Slater trajectories in five immutable shards of five, runs for `2*Ny` cycles, and observes six checkpoints strictly after `Ny`.

At every checkpoint the bundle saves every periodic full-x window `[0,Nx) x [y0,y0+Ay)` for `Ay=0,...,Ny//2` and every `y0`. Each `Ay_*.npz` holds the natural occupations plus cell-resolved von Neumann entropy and charge-variance contours. `common.npz` holds coordinates and the x-resolved squared correlator. No covariance, eigenvectors, topology, event records, fits, figures, or trajectory-averaged derived quantities are retained.

Production is locked until the A100 qualification completes both `Ny=60`, `Ay=30` hard and soft cases safely. The production queue runs or reuses the current version-matched qualification automatically before its first missing shard; the notebook also provides an optional cell to run it early and inspect the receipt. Run `run_production_bundle.ipynb` from the uploaded campaign folder, or invoke `run_bundle.py` directly.
