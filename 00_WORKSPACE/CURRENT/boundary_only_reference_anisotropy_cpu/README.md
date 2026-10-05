# Boundary-only reference-anisotropy calibration

This campaign applies Zabalo's space--time reference-ancilla matching protocol
to the exact-wall boundary circuit.  It starts from the exact half-filled ground
state of the periodic translation-invariant QWZ Hamiltonian at mass one.  The
initial state has no domain wall.  Domain-wall OW functions appear only in the
measurement--feedback controller, and only the two wall columns at `x=4,12`
are visited.

The spatial branch inserts two Gaussian two-mode references simultaneously on
one wall at separation `Ny/2`.  The temporal branch inserts them at the same
wall cell with separation `delta_tau`.  Their late-time mutual informations are
matched to determine

```text
alpha(Ny) = asinh(1) * Ny / (pi * t_star).
```

The independent statistical unit is one complete base Born trajectory.  Cycle
points, temporal branches, and reference modes are never treated as independent
samples.

## Run order

```bash
pytest -q tests
python run_campaign.py preflight --workers 1 --cpu-list 0
./launch_audit.sh
```

The audit screens burn-ins `4Ny,8Ny,12Ny,16Ny` at `Ny=24`, confirms the first
stable pair with 100 trajectories, and tests a `3Ny` reference plateau with a
locked `5Ny` extension.  It writes `analysis/audit_decision.json`.  Production
is deliberately blocked unless that file says `passed`.

After a passed audit:

```bash
./launch_two_lanes.sh
python analyze_campaign.py
```

Lane A uses physical CPUs `0-19` for `Ny=32,24,16`; lane B uses `28-47` for
`Ny=28,20`.  Every base checkpoint and reference branch has a checksum-bound
completion JSON.  Rerunning either lane verifies and skips completed work.

Production locates the crossing with 50 trajectories, retains one guard point
on each side, and increases the ensemble through `S=100,150,...,400` until the
crossing, 10% confidence-width, and wall-agreement criteria pass.  Failure at
the locked limits is an unresolved scientific result, not a software failure.

## Scientific interpretation

The actual and spectrally flattened uniform QWZ Hamiltonians have the same
negative-band Slater determinant; using the actual Hamiltonian makes the energy
and ground-state definition explicit.  The material change relative to the
previous boundary-only free-energy campaign is the absence of an initial domain
wall.

The final analysis fits `alpha(Ny)=alpha_infinity+b/Ny^2`, checks omission of
`Ny=16` and an added `1/Ny^4` term, and produces one PDF/PNG plus CSV/JSON
results.  Any conversion of the existing Casimir coefficient to `c_eff` is
marked diagnostic because anisotropy calibration cannot repair its fit-window
instability.
