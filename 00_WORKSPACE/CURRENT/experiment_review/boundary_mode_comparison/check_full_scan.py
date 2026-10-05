"""Readback, cut-window sensitivity, and independent eigensolver checks."""
import csv
import json
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from full_scan import OUT
from analyze import REPO, WALL_X, sha


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    samples=[]
    for artifact in tqdm(manifest['outputs'],desc='Readback and cut-distance checks'):
        p=OUT/artifact['file']
        if sha(p)!=artifact['sha256']:raise ValueError('output hash mismatch')
        with np.load(p) as z:
            xy=z['xy'];half=xy.shape[1]
            np.testing.assert_allclose(xy.sum(axis=(1,2)),1,atol=1e-9)
            w=z['wall_weight'];sep=z['individually_separated'];nu=z['occupations'][z['finite_indices']]
            base=(w>.8)&sep
            s={k:artifact[k] for k in ['wall','Ny','alpha1','sample']}
            for depth in [1,2,3,4]:
                joint=xy[:,depth:half-depth,:][:,:,WALL_X].sum(axis=(1,2))
                cut=1-xy[:,depth:half-depth,:].sum(axis=(1,2))
                keep=base&(cut<.5)
                s[f'modes_wall80_cut{depth}_below50']=int(keep.sum())
                s[f'modes_wall80_joint_depth{depth}_above50']=int((base&(joint>.5)).sum())
                s[f'modes_wall80_joint_depth{depth}_above50_cap1e8']=int((base&(joint>.5)&(nu>1e-8)&(nu<1-1e-8)).sum())
                s[f'max_wall_joint_depth{depth}']=float(joint[base].max(initial=0))
            samples.append(s)
    with (OUT/'cut_distance_sample_counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    summary=[]
    for key in sorted({(s['wall'],s['Ny'],s['alpha1']) for s in samples}):
        subset=[s for s in samples if (s['wall'],s['Ny'],s['alpha1'])==key]
        r=dict(zip(['wall','Ny','alpha1'],key))
        for k in subset[0]:
            if k.startswith('modes_'):
                r[k]=sum(s[k] for s in subset)
                r[k+'_samples']=sum(s[k]>0 for s in subset)
        summary.append(r)
    (OUT/'cut_distance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    # Independent LAPACK driver for the four plotted examples and deepest candidate.
    examples=json.loads((OUT/'examples.json').read_text())
    with np.load(OUT/'modes/soft/Ny032/alpha1_3/sample_086.npz') as z:
        xy=z['xy'];joint=xy[:,4:12,:][:,:,WALL_X].sum(axis=(1,2))
        selected=np.flatnonzero((z['wall_weight']>.8)&z['individually_separated'])
        k=int(selected[np.argmax(joint[selected])])
        examples.append(dict(file='modes/soft/Ny032/alpha1_3/sample_086.npz',row=k,sample=86))
    checks=[]
    for example in examples:
        with np.load(OUT/example['file']) as z:
            k=example['row'];source=REPO/str(z['source_file']);ny=int(z['Ny'])
            i=int(z['finite_indices'][k]);v=z['eigenvectors'][:,k];oldnu=float(z['occupations'][i])
        with np.load(source) as raw,threadpool_limits(limits=2):
            row=int(np.flatnonzero(raw['case_sample_indices']==example['sample'])[0])
            f=raw['final_frame'][row,:20*ny,:int(raw['final_ranks'][row])]
            nu,u=eigh(f@f.conj().T,driver='evr')
            overlap=float(abs(np.vdot(v,u[:,i]))**2)
            if overlap<.999:raise ValueError('independent eigenvector mismatch')
            checks.append(dict(file=example['file'],index=i,occupation_difference=abs(float(nu[i])-oldnu),
                squared_overlap=overlap))
    (OUT/'validation.json').write_text(json.dumps(dict(samples_readback_verified=len(samples),
        independent_driver_checks=checks,script_sha256=sha(__file__)),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
