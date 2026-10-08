"""Verified even-circumference curves and fits for the current manuscript."""
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import entropy_charge_support as fitting

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/endpoint_even_20261008'
SIZES = (24,28,32,40,50,60)
STYLES = {24:('#D92725','^'), 28:('#F08050','<'), 32:('#8FC1E3','v'),
          40:('#2CA02C','s'), 50:('#6B6B6B','D'), 60:('#1F77B4','o')}

def load_fits():
    provenance = json.loads((DATA/'provenance.json').read_text())
    for name, field in [('sample_curves.npz','compact_sha256'), ('fits.json','reported_fits_sha256')]:
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == provenance[field]
    reported = json.loads((DATA/'fits.json').read_text())
    analyzer = SimpleNamespace(CURVE_KEYS={k:k for k in reported},
                               PREFACTOR={k:v['prefactor'] for k,v in reported.items()})
    cases = {}
    with np.load(DATA/'sample_curves.npz',allow_pickle=False) as z:
        for ny in SIZES:
            cases[ny] = {'ay_values':np.arange(ny//2+1)}
            for key in reported:
                a = z[f'{key}_Ny{ny}'].copy()
                assert a.shape == (100,ny//2+1) and np.isfinite(a).all()
                cases[ny][key] = a
        contour = z['half_contour_Ny32'].copy()
    results = {}
    for key in reported:
        groups, summary = fitting.anchored_mean_curve_fit(analyzer,cases,key)
        for our, original in [('slope','slope'), ('slope_covariance_SEM','slope_covariance_SEM'),
                              ('converted_coefficient','coefficient'),
                              ('converted_covariance_SEM','coefficient_SEM'), ('R0_squared','R0_squared')]:
            np.testing.assert_allclose(summary[our],reported[key][original],rtol=2e-13,atol=1e-14)
        results[key] = groups, summary
    np.testing.assert_allclose(contour.sum((1,2)),cases[32]['entropy'][:,-1],atol=2e-8,rtol=0)
    return results, contour, provenance
