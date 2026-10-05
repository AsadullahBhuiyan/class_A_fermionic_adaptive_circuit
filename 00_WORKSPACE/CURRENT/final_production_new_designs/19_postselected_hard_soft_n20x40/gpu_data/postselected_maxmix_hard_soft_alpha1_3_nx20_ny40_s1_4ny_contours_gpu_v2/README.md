# Complete postselected contour campaign

Four checksum-verified results: hard/soft walls at alpha1=1 and 3, Nx=20,
Ny=40, T=160, one fully postselected trajectory each. Original NPZ and
completion JSON bytes are preserved in the alpha/wall hierarchy.

- `entropy_contour[cycle,x,y]`: 161x20x40 cell entropy in nats, orbitals summed.
- `entropy_contour_x[cycle,x]`: 161x20 entropy summed over y and orbitals.
- `total_entropy_nats[cycle]`, charge and occupation-gap diagnostics.
- Full endpoint active covariance, centered/occupation spectra, eigenvectors,
  and physical basis indices (880 modes hard, 1600 modes soft).

All histories include t=0. Divide contours or total entropy by Ny=40 offline
for the normalized plots. The Lyapunov gap at zero time is intentionally NaN.
Hard-wall exterior product sectors have zero entropy. The complete spatial
grid is retained for both wall types. No time-dependent covariance or
eigenvector histories were saved.

See DOWNLOAD_MANIFEST.json for source/config identity, original Drive file
IDs, checksums, and validation. No simulations or averaging were performed
during import, and nothing on Drive was changed.
