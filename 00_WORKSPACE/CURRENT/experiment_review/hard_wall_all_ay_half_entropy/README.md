# Lane-B half-system entropy collapse

This analysis uses the completed Lane-B portion of
`hard_wall_entropy_contour_all_ay_nx20_ny30-60_s100_2ny_raster_v3`:
`Nx=20`, `Ny=30,35,40,45,55`, and 100 independent endpoint trajectories per
size.  It does not pool the independent hard-wall v2 ensemble.

For every trajectory and strip width, the saved von Neumann contour has already
been aligned in relative `dy` and averaged over all periodic origins `y0`.
The analysis integrates that contour over the two disjoint half systems

```text
left:  x = 0,...,9
right: x = 10,...,19
```

so each half contains one hard wall, at `x=5` or `x=15`.  The two integrated
curves close exactly to the stored full-strip entropy.

Each trajectory is anchored at `Ay*=Ny//2`.  The anchored trajectory curves
are then averaged, and one through-origin slope is fitted jointly across the
five completed circumferences over `Ay=8,...,Ny//2`.  Each circumference has
equal total fit weight.  Error bars and slope errors are ordinary trajectory
sampling SEMs; slope errors propagate the full covariance between strip
widths.  No bootstrap or fit-residual error is used.  The single-wall CFT
target is `m=1/6`.

Run from the repository root:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/hard_wall_all_ay_half_entropy/make_half_entropy_collapse.py
```

The script revalidates all 100 Lane-B result/checksum pairs and all sample IDs
before writing the PDF/PNG figure, curve CSV, fit CSV, and analysis manifest.

## Wall-only variant

`make_wall_only_entropy_collapse.py` repeats the identical mean-first,
half-strip-anchored collapse after restricting the spatial sum to the
three-cell windows centered on the walls:

```text
left wall:  x = 4,5,6
right wall: x = 14,15,16
```

It writes independent wall-only PDF/PNG, curve CSV, fit CSV, and manifest
outputs while leaving the half-system products unchanged.

The default wall-only command uses those three-cell windows.  To retain only
the domain-wall center cells, use:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/hard_wall_all_ay_half_entropy/make_wall_only_entropy_collapse.py --window center
```

This produces separate `wall_center_only` outputs for (x=5) and (x=15).
