# Raw-separation log-log variant

Generate with `python ../../build_correlator_summary_3x1.py --raw-separation`.
This standalone copy leaves the technical report and original figure unchanged.

Panels (a,b) show the same arithmetic ensemble means against physical separation
ry=1..30, using logarithmic x and y axes. Axis labels give the untransformed
quantities, not log chord. The display-only C>1e-8 cutoff, marker/color assignments,
wall annotation, and 100 trajectories per ensemble are unchanged. Panel (c)
retains the normalized log-chord collapse, the 8..Ny/2 fit, gray ranges, and beta=2.1903.

This is an axis-only remake from the v2 preserved compact data export; its mean
curves are checked against the v2 plotted CSV. It does not rerun data generation
or rely on current engine copies matching historical engine hashes. The new
summary records SHA-256 identities of the compact export, plotted CSV, and original
provenance summary. The original summary retains the scientific-input provenance.

For captions, replace descriptions of (a,b) as log C versus log chord by
"squared correlators versus separation ry on logarithmic axes." Panel (c)'s
caption and fitting interpretation do not change. Connecting lines in (a,b)
remain guides to the eye; no new raw-separation power-law fit is introduced.
