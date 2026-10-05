# Sample-resolved effective central charge

This analysis uses the completed hard-wall, pure-state, perfect-correction
campaign at `Nx=20`, `Ny=30,40,50`, and `S=100`.  The saved entropy arrays are
already resolved by trajectory and cycle; only the periodic strip origin `y0`
was averaged within each trajectory.

For every trajectory `xi` and physical cycle, the script fits

```text
S_xi(Ay,t) = m_xi(t) log[(Ny/pi) sin(pi Ay/Ny)] + b_xi(t)
```

on `Ay=8,...,Ny/2`, defines `c_{1,xi}(t)=3m_xi(t)`, and then averages the 100
resolved estimates.  Error bars are the simple sample-wise standard error
`SD(c_{1,xi})/sqrt(100)`.  The translated origins are never treated as
independent samples.

For a fixed fit window, an ordinary least-squares slope is a linear functional
of the entropy curve.  Consequently, averaging the 100 fitted slopes and
fitting the trajectory-averaged curve give the same point estimate up to
floating-point roundoff.  The improvement is statistical: the resolved
reduction exposes the physical sample distribution and supplies a valid
sampling uncertainty, whereas the older plot reported only the regression
residual of the mean curve.

Run from the repository root with:

```bash
python 00_WORKSPACE/CURRENT/experiment_review/entropy_ceff_sample_resolved/analyze_sample_resolved_ceff.py
```

Outputs include a two-column PDF and 300-dpi PNG, an all-cycle summary CSV, a
compressed NPZ containing every derived sample-wise `c_{1,xi}(t)` and its
standard error, and a checksum-bearing analysis manifest.
