#!/usr/bin/env python3
"""Extract and verify saved untwirled occupations; no circuit simulations."""
from pathlib import Path
import hashlib,json,numpy as np
from scipy.linalg import eigvalsh
root=Path(__file__).resolve().parents[1]
data=root/'data/mean_channel'
summary=json.loads((data/'spectrum_summary.json').read_text())
outputs={}; receipts={}
for a in (1,3):
 info=summary[f'alpha{a}_hard'];p=Path(info['source'])
 assert hashlib.sha256(p.read_bytes()).hexdigest()==info['source_sha256']
 with np.load(p,allow_pickle=False) as saved:
  assert saved['spectral_cycles'][-1]==128
  G=saved['G_final'][0]
  np.testing.assert_array_equal(saved['active_indices'],np.arange(2560))
  np.testing.assert_allclose(G,saved['active_G_final'],atol=1e-14,rtol=0)
  nu=np.sort(saved['active_occupations'][-1])
  herm=float(np.max(np.abs(G-G.conj().T)))
  exact=eigvalsh(G,check_finite=False)
  np.testing.assert_allclose(nu,exact,atol=2e-12,rtol=0)
  assert nu.min()>-2e-12 and nu.max()<1+2e-12
  outputs[f'alpha{a}']=nu
  receipts[str(a)]={'source':str(p),'source_sha256':info['source_sha256'],'saved_key':'active_occupations[-1]','cycle':128,'modes':len(nu),'hermiticity_max_error':herm,'maximum_eigenvalue_verification_error':float(np.max(np.abs(exact-nu))),'last_cycle_normalized_frobenius_change':info['last_cycle_normalized_frobenius_change']}
np.savez_compressed(data/'untwirled_spectra.npz',**outputs)
(data/'untwirled_spectra_provenance.json').write_text(json.dumps({'description':'Ascending eigenvalues of the untwirled full G_final; saved eigenvalues independently checked against G_final. No dynamics rerun.','cases':receipts,'sha256':hashlib.sha256((data/'untwirled_spectra.npz').read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps(receipts,indent=2))
