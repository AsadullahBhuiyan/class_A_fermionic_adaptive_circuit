"""Bundle the imported endpoint observables without running dynamics."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[2] / 'experiment_review/endpoint_figures_import_20261008/extracted/Nx20_Ny24_28_32_40_50_60_endpoint_figures'
DATA = ROOT / 'data/endpoint_even_20261008'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    DATA.mkdir(exist_ok=True)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    payload = {}
    inputs = []
    for ny in manifest['config']['Ny_values']:
        chunks = []
        for path in sorted((SOURCE/f'results/Ny{ny:03}').glob('*.npz')):
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert digest(path) == receipt['sha256']
            assert path.stat().st_size == receipt['bytes']
            assert receipt['config_sha256'] == manifest['config_sha256']
            with np.load(path, allow_pickle=False) as values:
                chunk = {key: values[key].copy() for key in values.files}
            chunks.append(chunk)
            inputs.append({'path':str(path.relative_to(SOURCE)), 'sha256':digest(path),
                           'receipt_sha256':digest(path.with_suffix('.json'))})
        ids = np.concatenate([c['sample_ids'] for c in chunks])
        np.testing.assert_array_equal(ids, np.arange(100))
        contour = np.concatenate([c['endpoint__contour_von_neumann_y0avg'] for c in chunks])
        entropy = np.concatenate([c['endpoint__entropy_von_neumann'] for c in chunks])
        variance = np.concatenate([c['endpoint__charge_variance'] for c in chunks])
        np.testing.assert_allclose(contour.sum((2,3)), entropy, atol=2e-8, rtol=0)
        payload[f'entropy_Ny{ny}'] = entropy
        payload[f'variance_Ny{ny}'] = variance
        payload[f'left_Ny{ny}'] = contour[:,:,[5,6],:].sum((2,3))
        payload[f'right_Ny{ny}'] = contour[:,:,[14,15],:].sum((2,3))
        if ny == 32:
            payload['half_contour_Ny32'] = contour[:,-1].transpose(0,2,1)
    np.savez_compressed(DATA/'sample_curves.npz', **payload)
    for name in ['fits.json','curves.csv']:
        shutil.copy2(SOURCE/'analysis'/name, DATA/name)
    provenance = {'source_root':str(SOURCE), 'source_manifest_sha256':digest(SOURCE/'manifest.json'),
                  'config':manifest['config'], 'config_sha256':manifest['config_sha256'],
                  'canonical_entry_point':manifest['canonical_entry_point'], 'inputs':inputs,
                  'compact_sha256':digest(DATA/'sample_curves.npz'),
                  'reported_fits_sha256':digest(DATA/'fits.json'),
                  'reported_curves_sha256':digest(DATA/'curves.csv'),
                  'extraction_source_sha256':digest(Path(__file__)),
                  'contour_axes':['sample','relative_dy','x'], 'no_simulation':True}
    (DATA/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Bundled 600 trajectories; source observables preserved.')

if __name__ == '__main__':
    main()
