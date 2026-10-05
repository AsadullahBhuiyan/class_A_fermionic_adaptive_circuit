# Kraus boundary-contour analysis

This project tests a spatial decomposition of the many-body squared-singular-
value spectrum of a realized Gaussian Kraus operator.  It keeps the four input
ensembles separate:

1. the legacy `Nx=20, Ny=20, S=10, T=40` complete covariance history;
2. the completed hard-v2 and soft-v3 purification endpoints (`S=100` at
   `Ny=20,30,40`);
3. the completed hard-wall bundle-13 soft-mode data (`S=100` at seven sizes);
4. a new deterministic CPU trajectory/replay with event-resolved record weight.

For a normalized max-mix output with occupations `nu_a`, the leading many-body
level is

```text
ell_0 = log Z + sum_a log(max(nu_a,1-nu_a)).
```

The first term receives an exact event-wise contour from the chain rule for the
record probability.  The second receives an exact spectral contour from the
occupation eigenvector weights.  Flip costs

```text
d_a = abs(log(nu_a/(1-nu_a)))
```

give additive gap contours.  Near-degenerate modes are interpreted through
aggregate wall subspaces, not individual eigenvector labels.

The analysis never pools historical sampling revisions.  Generated products
are written under `analysis_outputs/kraus_boundary_contours_v1/`.

## Reproduction

Run the four inputs and then build the combined report:

```bash
python analyze_legacy_history.py
python analyze_purification_endpoints.py
python analyze_bundle13_gap_contours.py
python run_event_contour_replay.py
python build_report.py
```

The reader-facing products are
`analysis_outputs/kraus_boundary_contours_v1/kraus_boundary_contour_report.pdf`
and `analysis_outputs/kraus_boundary_contours_v1/RESULTS.md`.  The report is a
single two-page PDF containing both the primary four-panel diagnostic and the
supplementary contour/gap-ratio panel.
