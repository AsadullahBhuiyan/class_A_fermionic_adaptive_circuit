# Downloaded Campaign 22: clipped alpha1=3 continuation

Source account: `abhuiyan2398@gmail.com`.
Source folder: https://drive.google.com/drive/folders/1-olBIfvnwaCj3ObzmTykaM269CHJ7jzn

This independent output collection contains the **continuation of the original
ensemble**, not a new independent ensemble: Nx=20, Ny=30, alpha1=3, alpha2=30,
100 samples, full-system measurements, hard walls, complex128, 60 cycles.
Cycles 0–30 retain the original unclipped observations. The cycle-30 state was
spectrally clipped at handoff; cycles 31–60 use cycle-end spectral clipping.
`fork_provenance.json` identifies the original checkpoint and handoff corrections.

The original 20 NPZ files and 20 completion JSONs are retained byte-for-byte
under `results/hard/alpha1_3/Ny030/`. Each shard holds five trajectories, including
occupations and scalar entropy/charge/variance at cycles 0–60, record-probability
histories, clipping diagnostics, and the final complex128 centered covariance.
Alpha1=3 does not contain spatial contours or slow-mode vectors; those were
requested only for the separate alpha1=1 run.

`drive_inventory.json` records remote file identities and byte counts.
`DOWNLOAD_MANIFEST.json` is written only after all result checksums, completion
identities, exact sample/cycle coverage and numerical consistency checks pass.
Reproduce the local verification with `python ../../verify_download.py` from
this directory. No original Drive outputs or campaign-21 data are modified.
