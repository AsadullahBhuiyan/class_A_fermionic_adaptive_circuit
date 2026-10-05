# Sequential postselected hard/soft A100 campaign

This standalone bundle runs four deterministic full-postselection trajectories
in one A100 notebook: `alpha1_1/hard`, `alpha1_1/soft`, `alpha1_3/hard`,
then `alpha1_3/soft`. No averaging or trajectory SEM is implied.

## Scientific contract

- `Nx=20`, `Ny=40`, `T=4Ny=160`
- maximally mixed initialization
- hard/support-truncated and soft/untruncated domain walls at `x=[5,15]`
- `nshell=1`, `alpha_1_values=[1,3]`, `alpha_2=30`, raster-y ordering
- `postselect=True`, no feedback, complex128 covariance dynamics
- canonical `classA_U1FGTN_gpu.run_markov_circuit`

Full postselection forces one deterministic outcome record, so there is one
trajectory per construction and no trajectory SEM. The hard construction uses
`meas_slab_only=True` and analyzes its `22Ny=880` active modes after the
exterior is forced into a fixed occupied product sector. The soft construction
uses `meas_slab_only=False` and analyzes all `40Ny=1600` modes.

At every cycle, including zero, the bundle saves total entropy, active charge,
`entropy_contour[cycle,x,y]` (shape 161x20x40),
`entropy_contour_x[cycle,x]` (shape 161x20), and `entropy_closure_error`,
the centered occupation gap, the raw occupation-logit gap, and

```text
Delta_lambda(t) = min_j |atanh(a_j)| / t,
```

where `a_j` are eigenvalues of the centered covariance. The rate is undefined
at `t=0`. The endpoint active covariance, complete centered/occupation spectra,
and eigenvectors are retained.

The entropy contour is `s(x,y)=sum_(mu,j) |U_(mu,x,y),j|^2 h(nu_j)`,
where `h(nu)=-nu log(nu)-(1-nu)log(1-nu)` is in nats and exact caps
contribute zero. It sums over the two orbitals per cell, not over y;
`entropy_contour_x` additionally sums over y. Divide by Ny offline for
the plotted normalization. Canonical basis order is `mu+2*x+2*Nx*y`.
Hard-wall exterior product modes contribute zero; the active contour is
embedded into the full 20x40 grid. Soft walls use the entire physical grid.
Each contour sums to the saved total entropy. Eigenvectors are computed
transiently every cycle but saved only at the endpoint. This adds eigensolver
work, but only about 1.1 MB of uncompressed contour data per trajectory.

This is observer/result/checkpoint v2. The dynamics, seeds, and postselection
are unchanged. Existing v1 data and checkpoints lack the contour history and
are not resumed or overwritten. All four runs start from cycle zero in the
new v2 output collection; subsequent v2 restarts resume normally.

## Resume and execution

Each construction writes a rolling covariance/NumPy/Torch/CUDA RNG/observer
checkpoint every ten cycles. The previous checkpoint is replaced only after a
DriveFS byte/SHA-256 readback succeeds. A result NPZ is followed by a
checksum-bound completion JSON; its checkpoint is removed only after both
verify. Restarting the notebook skips a completed hard run and proceeds to the
soft run.

Upload this folder to
`MyDrive/final_production_new_designs/19_postselected_hard_soft_n20x40`, open
`run_postselected_hard_soft_n20x40.ipynb` on an A100 40-GB runtime, and run all
cells. Set `MAX_NEW_CONSTRUCTIONS=1` only if you deliberately want to stop after
one pending alpha/wall run; the default `None` runs all four sequentially.
Use a fresh runtime after replacing the bundle files to avoid cached imports.

Outputs are written to
`MyDrive/classA_final_production_outputs/postselected_maxmix_hard_soft_alpha1_3_nx20_ny40_s1_4ny_contours_gpu_v2`,
with separate `alpha1_1` and `alpha1_3` directories, each containing
`results/hard`, `results/soft`, and rolling checkpoints. The `alpha_1_values`
list controls the sweep; the runner resolves `alpha_1` for each run.
