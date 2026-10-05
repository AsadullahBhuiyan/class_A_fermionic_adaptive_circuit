from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.special import xlogy
from tqdm.auto import tqdm
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v2'

def analyze():
    cache=SOURCE/'subsystem_spectra.npz'
    meta=json.loads((SOURCE/'spectra_diagnostics.json').read_text())
    digest=hashlib.sha256(cache.read_bytes()).hexdigest();assert digest==meta['cache_sha256']
    windows=np.array([.5,.9,.95,.99,.999,.9999,.99999,1.])
    widths=np.arange(1,17);origin_entropy=np.empty((100,len(windows),16,32))
    raw_min=1.;raw_max=-1.
    with np.load(cache) as z:
        assert np.array_equal(z['sample_ids'],np.arange(100)) and np.array_equal(z['origins'],np.arange(32))
        for j,a in enumerate(tqdm(widths,desc='Entropy contributions by spectral window',unit='width')):
            ev=z[f'eigenvalues_Ay{a:02d}'];assert ev.shape==(100,32,40*a) and np.isfinite(ev).all()
            raw_min=min(raw_min,float(ev.min()));raw_max=max(raw_max,float(ev.max()))
            assert ev.min()>=-1-1e-8 and ev.max()<=1+1e-8
            lam=np.clip(ev,-1,1);nu=(1+lam)/2
            entropy=-xlogy(nu,nu)-xlogy(1-nu,1-nu)
            for k,L in enumerate(windows):origin_entropy[:,k,j]=np.sum(entropy*(np.abs(lam)<=L),axis=-1)
    assert np.all(np.diff(origin_entropy,axis=1)>=-1e-12)
    trajectory_entropy=origin_entropy.mean(-1)
    mean=trajectory_entropy.mean(0);sem=trajectory_entropy.std(0,ddof=1)/10
    x=np.log(32/np.pi*np.sin(np.pi*widths/32));mask=widths>=5
    X=np.column_stack([np.ones(mask.sum()),x[mask]]);operator=np.linalg.pinv(X)
    trajectory_coeff=np.einsum('pa,ska->skp',operator,trajectory_entropy[:,:,mask])
    coef=trajectory_coeff.mean(0);coef_sem=trajectory_coeff.std(0,ddof=1)/10
    residuals=mean[:,mask]-coef@X.T
    total=((mean[:,mask]-mean[:,mask].mean(1,keepdims=True))**2).sum(1)
    r2=1-(residuals**2).sum(1)/total
    table=pd.DataFrame(dict(L=windows,intercept=coef[:,0],slope=coef[:,1],slope_sem=coef_sem[:,1],R_squared=r2,
          residual_rms=np.sqrt((residuals**2).mean(1)),half_entropy=mean[:,-1],half_entropy_sem=sem[:,-1],
          half_entropy_retained_fraction=mean[:,-1]/mean[-1,-1]))
    table.to_csv(OUT/'entropy_window_fits.csv',index=False)
    np.savez_compressed(OUT/'entropy_window_statistics.npz',windows=windows,widths=widths,sample_ids=np.arange(100),origins=np.arange(32),
                        origin_entropy=origin_entropy,trajectory_entropy=trajectory_entropy,mean=mean,sem=sem,log_chord=x,
                        trajectory_coefficients=trajectory_coeff,coefficient_covariance=np.cov(trajectory_coeff.reshape(100,-1),rowvar=False)/100,
                        residuals=residuals)
    previous=OUT.parent/'stochastic_equilibrium_spectral_comparison_v1/findings.json'
    prior=json.loads(previous.read_text())['entropy_origins']['mean']
    assert abs(mean[-1,-1]-prior)<1e-10
    diag=dict(Nx=20,Ny=32,samples=100,origins=32,cycles=64,windows=windows.tolist(),fit_widths=list(range(5,17)),
      estimator='sum binary entropy for abs(centered eigenvalue)<=L; average origins within trajectories, then trajectories; no renormalization',
      fit='unanchored ordinary least squares: a + b log[(Ny/pi)sin(pi Ay/Ny)]',
      uncertainty='SEM of 100 trajectory-level origin averages; full cross-width correlations propagated',
      comparison_scope='same fixed-size two-parameter fits as mode-count analysis, not the report joint multi-size anchored fit',
      input_cache=str(cache),input_sha256=digest,input_provenance=meta,
      raw_min=raw_min,raw_max=raw_max,full_entropy_previous_result_difference=float(mean[-1,-1]-prior),
      clipping='Only bound excursions within 1e-8 clipped for entropy evaluation; raw cache unchanged',fits=table.to_dict('records'))
    (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    return table,diag

if __name__=='__main__':
    import os
    os.sched_setaffinity(0,[8,9]);print(analyze()[0].to_string(index=False))
