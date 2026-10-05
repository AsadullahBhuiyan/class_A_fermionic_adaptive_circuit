from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import nbformat
OUT=Path(__file__).resolve().parent
old=OUT.parent/'centered_spectral_window_vs_subsystem_n20x32_hard_alpha1_v1'
with np.load(OUT/'subsystem_spectra.npz') as z,np.load(old/'subsystem_spectra.npz') as previous:
    assert np.array_equal(z['sample_ids'],np.arange(100))
    assert np.array_equal(z['origins'],np.arange(32))
    assert np.array_equal(z['widths'],np.arange(1,17))
    errors=[]
    for a in range(1,17):
        ev=z[f'eigenvalues_Ay{a:02d}']
        assert ev.shape==(100,32,40*a)
        assert np.isfinite(ev).all() and ev.min()>=-1-1e-8 and ev.max()<=1+1e-8
        errors.append(float(np.max(abs(ev[:,0]-previous[f'eigenvalues_Ay{a:02d}']))))
    assert max(errors)<1e-10
with np.load(OUT/'window_statistics.npz') as z:
    counts=z['central_counts'];mixed=z['mixed_counts']
    comp=int(np.max(abs(counts[:,-1,:16]-counts[:,-1,16:])))
    assert comp==0
    assert counts.shape==(100,16,32) and (counts<=mixed).all()
r=json.loads((OUT/'window_diagnostics.json').read_text());t=pd.read_csv(OUT/'window_integrals.csv')
individual=(counts/mixed).mean(-1)
assert np.allclose(t.mixed_normalized_integral,individual.mean(0),rtol=0,atol=1e-15)
assert np.allclose(t.mixed_normalized_integral_sem,individual.std(0,ddof=1)/10,rtol=0,atol=1e-15)
previous_table=pd.read_csv(OUT.parent/'centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v1/window_integrals.csv')
assert np.array_equal(t.mode_count,previous_table.mode_count)
check=dict(fixed_origin_spectrum_max_difference=max(errors),half_width_complement_central_count_max_difference=comp,
           total_spectra=51200,total_raw_eigenvalues=17408000,
           max_change_from_v1=float(np.max(abs(t.mixed_normalized_integral-previous_table.mixed_normalized_integral))),
           raw_count_unchanged=True,trajectory_integral_mean_and_sem_validated=True,
           max_difference_origin_average_vs_global_pool=float(np.max(abs(t.mixed_normalized_integral-t.globally_pooled_integral))),
           max_difference_origin_average_vs_individual_normalization=float(np.max(abs(t.mixed_normalized_integral-t.mean_individually_normalized_integral))))
(OUT/'cross_checks.json').write_text(json.dumps(check,indent=2)+'\n')
f=r['fits']['mode_count'];half=r['half_system'];primary=f[str(r['default_fit_min_width'])]
report=f'''# Origin-averaged central spectral weight, 20 by 32

All 100 hard-wall alpha1=1 pure endpoints at cycle 64, subsystem widths 1–16, all 32 periodic origins, L={r['L']}. All x columns and both orbitals are retained. No simulation or origin subsampling is used.

Normalize each trajectory-origin mixed-mode spectrum after excluding |lambda| >= 1 - 1e-8. Average the 32 normalized densities inside each trajectory, then average the 100 trajectories. Its central-window integral is mean_s mean_origin [n(s,origin)/m(s,origin)]. This v2 implements the requested equal trajectory and origin weighting; the earlier v1 pooled-sample estimator is preserved in the adjacent v1 directory. The raw count averages all origins inside each trajectory, then averages the 100 independent trajectories. Error bars retain the correlations among cuts and widths; origins are not additional independent samples. CSV columns also provide all-mode normalization and the alternative orders of conditional normalization.

At half width, the normalized integral is {half['mixed_normalized_integral']:.8f} +/- {half['mixed_normalized_integral_sem']:.8f} (one SEM), the raw mean count is {half['mode_count']:.6f} +/- {half['mode_count_sem']:.6f}, and the all-mode normalized integral is {half['all_modes_integral']:.8f}.

The descriptive fit n = a + b log[(32/pi) sin(pi Ay/32)], over widths {r['default_fit_min_width']}–16, gives a={primary['intercept']:.6f}, b={primary['slope']:.6f} +/- {primary['slope_sem']:.6f}, R-squared={primary['R_squared']:.6f}. Changing the lower width to 2 gives b={f['2']['slope']:.6f} +/- {f['2']['slope_sem']:.6f}; to 4 gives b={f['4']['slope']:.6f} +/- {f['4']['slope_sem']:.6f}. These finite-size fits do not establish an asymptotic law. A normalized fraction has an additional width-dependent denominator and need not have the same scaling as a count.

Validation: source receipts, configuration, bytes, hashes, exact sample IDs 0–99; finite matrices, frame orthonormality, Hermiticity before symmetrizing, spectral bounds and trace sums. Origin-zero spectra match the preceding calculation within {max(errors):.3g}; half-width complementary cuts give identical central counts. Unmodified spectra are cached, allowing L to change without new diagonalizations. Fixed-origin products remain in the adjacent `centered_spectral_window_vs_subsystem_n20x32_hard_alpha1_v1` directory.

## Files

- `spectral_window_vs_subsystem.ipynb`: editable CPU notebook; plotting commands immediately above figures.
- `subsystem_spectra.npz`: eigenvalues indexed by width, with axes trajectory, origin, mode; sample IDs, origins and widths included.
- `window_integrals.csv`: central spectral weights, mean counts, SEMs and normalization alternatives.
- `sample_window_counts.csv`: all sample/width/origin counts.
- `window_statistics.npz`: count arrays and full covariance matrices across widths.
- `spectra_diagnostics.json`, `window_diagnostics.json`, `cross_checks.json`: provenance, numerical and statistical checks.
- `spectral_window_vs_log_chord.pdf/png`, `spectral_window_vs_width.pdf/png`: vector and 300-dpi figures.

Figure caption: S=100 independent trajectories initialized as pure states with the exterior prepared as a product frame; hard-wall production endpoints at cycle 64. Nx=20, Ny=32, alpha1=1, alpha2=30, nshell=1. Each periodic subsystem contains all x and both orbitals, with Ay=1–16. Centered covariance eigenvalues are computed separately for each trajectory and origin. Each mixed-mode spectrum is normalized separately, then origins are averaged inside each trajectory before trajectories are averaged. Raw counts average origins within trajectories before ensemble averaging. Error bars are ordinary SEMs of the 100 trajectory-level origin averages. Fits use widths {r['default_fit_min_width']}–16; full cross-width trajectory covariance is propagated to coefficient uncertainties.
'''
(OUT/'README.md').write_text(report)
print(json.dumps(check,indent=2));print(report.split('## Files')[0])
