"""Count spectral-window modes from saved pure-state frames; no dynamics."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib, json, os
import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/entanglement_spectrum'
REPO = next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def count_sample(task):
    sid, padded, rank = task
    frame = padded[:, :int(rank)]
    assert np.max(abs(frame.conj().T@frame-np.eye(rank))) < 1e-8
    q = 2*(frame@frame.conj().T)-np.eye(1280)
    counts = np.empty((15,32), dtype=np.int32)
    for j, width in enumerate(range(1,16)):
        for origin in range(32):
            indices = (((origin+np.arange(width))%32)[:,None]*40+np.arange(40)).ravel()
            block = q[np.ix_(indices, indices)]
            v = eigvalsh((block+block.conj().T)/2, subset_by_value=(-.99,.99),
                         driver='evr', check_finite=False)
            assert not np.any(abs(v)==.99)
            counts[j,origin] = np.count_nonzero(abs(v)<.99)
    return sid, counts

def main():
    original=json.loads((DATA/'provenance.json').read_text())
    assert sha(DATA/'inputs.npz') == original['compact_input_sha256']
    with np.load(DATA/'inputs.npz',allow_pickle=False) as z:
        top=z['counts_by_sample_width_origin'].copy()
    pooled=json.loads((DATA/'pooled_half_strip_provenance.json').read_text())
    assert sha(DATA/'pooled_half_strip_spectra.npz') == pooled['compact_input_sha256']
    with np.load(DATA/'pooled_half_strip_spectra.npz',allow_pickle=False) as z:
        half=(abs(z['centered_occupations'])<.99).sum(-1)
    np.testing.assert_array_equal(top[:,-1,:],half[0])
    triv=np.empty_like(top);triv[:,-1,:]=half[1]
    sources=[];seen=[]
    workers=min(8,len(os.sched_getaffinity(0)))
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pooled['alpha1_3_sources']:
            path=REPO/record['result']; receipt=record['completion']
            assert sha(path)==receipt['result_sha256']
            sources.append(dict(path=str(path),sha256=sha(path)))
            with np.load(path,allow_pickle=False) as z:
                assert z['Nx']==20 and z['Ny']==32 and z['cycles_total']==64 and z['alpha_1']==3
                frames,ranks,ids=z['final_frame'],z['final_ranks'],z['case_sample_indices']
            jobs=((int(sid),frames[j],int(ranks[j])) for j,sid in enumerate(ids))
            for sid,counts in tqdm(pool.map(count_sample,jobs),total=len(ids),
                                   desc='Saved alpha=3 window counts',unit='trajectory'):
                triv[sid,:15,:]=counts;seen.append(sid)
    assert sorted(seen)==list(range(100))
    counts=np.stack([top,triv])
    path=DATA/'window_count_comparison.npz'
    np.savez_compressed(path,alpha_1=[1,3],sample_ids=np.arange(100),
                        widths=np.arange(1,17),origins=np.arange(32),counts=counts,
                        Nx=20,Ny=32,cycle=64,window=.99)
    report=dict(no_simulations=True,window='abs(2nu-1)<0.99',shape=list(counts.shape),
                sources=sources,alpha1_1_preserved=True,half_strip_counts_verified=True,
                compact_sha256=sha(path),renderer_sha256=sha(__file__),workers=workers)
    (DATA/'window_count_comparison_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
