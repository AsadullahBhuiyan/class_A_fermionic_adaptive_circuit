"""Matched 2x2 endpoint occupation spectra: alpha rows and wall columns."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from campaign_schema import sha256_file
from observables import exact_y_twirl, ky_spectrum_and_x_weights, translation_residual

HERE=Path(__file__).resolve().parent
OLD=HERE/'results/matched_markov_lindblad_perfect_correction_v1_20260821T222115Z_afb274a039e6'
NEW=HERE/'results/alpha3_full_system_dissipation_v1_20260915T004928Z'
OUT=HERE/'analysis_outputs/alpha1_vs3_hard_soft_full_system_ky'


def read_endpoint(family,alpha,hard):
    if alpha==1:
        root=OLD/'cases'/f'N20x64_ain1p0_aout30p0_nsh1_dwtrunc{int(hard)}_{family}_deph1_pc1'
        assert (root/'_SUCCESS').exists()
        meta=json.loads((root/'metadata.json').read_text())
        path=root/'observables.npz';expected=meta['observables_sha256']
        model=meta['case']['model'];dyn=meta['case']['dynamics']
        assert model['Nx']==20 and model['Ny']==64 and model['nshell']==1
        assert model['alpha_run_in']==1 and model['alpha_run_out']==30
        assert model['wall_locations']==[5,15] and model['dw_truncation']==hard
        assert dyn['dephasing'] and dyn['perfect_correction'] and dyn['cycles']==128
        assert dyn['init_mode']=='maxmix'
        seed=dyn['sample_seeds'][0]
    else:
        root=NEW/(family+('_hard' if hard else '_soft'))
        meta=json.loads((root/'completion.json').read_text());cfg=meta['config']
        assert meta['status']=='complete'
        path=root/meta['result_filename'];expected=meta['result_sha256']
        assert path.stat().st_size==meta['result_bytes']
        assert cfg['evolution_domain']=='full_system' and not cfg['meas_slab_only']
        assert cfg['exterior_preparation']=='none' and cfg['dw_truncation']==hard
        assert cfg['Nx']==20 and cfg['Ny']==64 and cfg['alpha_1']==3 and cfg['alpha_2']==30
        assert cfg['nshell']==1 and cfg['dephasing'] and cfg['cycles']==128 and cfg['perfect_correction']
        seed=cfg['schedule_seed']
    assert seed==2257147520926244099
    assert sha256_file(path)==expected
    with np.load(path,allow_pickle=False) as z:
        final=z['G_final'][0]
        distance=float(z['successive_state_distance'][0,-1])
    assert final.shape==(2560,2560)
    return final,dict(source=str(path),sha256=expected,last_step_distance=distance,
                     schedule_seed=seed,source_metadata=str(root/('metadata.json' if alpha==1 else 'completion.json')))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    summary=[];products={}
    for family in ('markov_channel','lindblad'):
        fig,axes=plt.subplots(2,2,figsize=(7.05,5.35),sharex=True,sharey=True)
        for row,alpha in enumerate((1,3)):
            for col,hard in enumerate((True,False)):
                label=f'{family}_alpha{alpha}_'+('hard' if hard else 'soft')
                print('[spectrum]',label,flush=True)
                final,info=read_endpoint(family,alpha,hard)
                twirled=exact_y_twirl(final,20,64)
                ky,occ,weights=ky_spectrum_and_x_weights(twirled,20,64)
                assert occ.min()>=-1e-10 and occ.max()<=1+1e-10
                trace_error=float(abs(occ.sum()-np.trace(final).real));assert trace_error<1e-8
                order=np.argsort(ky);ky=ky[order];occ=occ[order];weights=weights[order]
                ax=axes[row,col]
                ax.scatter(np.repeat(ky/np.pi,40),occ.ravel(),s=6,
                           marker='o' if hard else '^',facecolors='none',
                           edgecolors='#2468ad' if hard else '#b3312d',linewidths=.45)
                ax.axhline(.5,color='.5',ls='--',lw=.8)
                ax.set(xlim=(-1.03,1.03),ylim=(-.035,1.035))
                ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([0,.25,.5,.75,1])
                ax.tick_params(top=True,right=True)
                ax.text(-.13,1.04,f'({chr(97+2*row+col)})',transform=ax.transAxes)
                ax.text(.04,.94,rf'$\alpha_1={alpha}$',transform=ax.transAxes,va='top',
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
                if row==0:ax.set_title('Hard wall' if hard else 'Soft wall')
                if row==1:ax.set_xlabel(r'$k_y/\pi$')
                if col==0:ax.set_ylabel(r'Occupation $\nu_a(k_y)$')
                products[label+'_occupations']=occ;products[label+'_x_weights']=weights
                info.update(family=family,alpha_1=alpha,alpha_2=30,wall='hard' if hard else 'soft',
                    raw_translation_residual=translation_residual(final,20,64),trace_error=trace_error,
                    minimum_distance_to_half=float(abs(occ-.5).min()),
                    k0_central_occupations=occ[np.argmin(abs(ky)),19:21].tolist())
                summary.append(info)
        name='Quantum channel' if family=='markov_channel' else 'Lindblad + dephasing'
        fig.suptitle(name+r': $20\times64$, $n_{\rm shell}=1$, endpoint 128',fontsize=10)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/f'{family}_alpha_wall_ky_2x2.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    np.savez_compressed(OUT/'spectra.npz',ky=ky,physical_x=np.arange(20),**products)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'caption.txt').write_text(
        'Full-system endpoint ky-resolved occupation spectra: top alpha_1=1, bottom alpha_1=3; '
        'left support-truncated hard walls, right untruncated soft walls. Alpha_2=30 throughout. '
        'Nx=20, Ny=64, walls x=5,15 inclusive, nshell=1, full-system maximally mixed initialization, '
        'perfect correction, complex128. All centers evolve in every case; none has a frozen '
        'exterior or active-slab restriction. Include all 40 orbitals per ky, including exterior modes. '
        'Channel: one seeded random site schedule, outcomes averaged analytically, cycle 128. '
        'Lindblad: unit gain/loss/number-dephasing rates, RK4 dt=0.05, t=128. '
        'Each spectrum diagonalizes ky-diagonal blocks of the endpoint after the exact y-translation '
        'twirl. Channel inter-momentum coherences are discarded; no temporal average is used. '
        'There is no trajectory-sampling error bar, fit, or band interpolation. Dashed reference: '
        'occupation 1/2. Alpha_1=1 uses the preserved completed matched campaign; alpha_1=3 uses '
        'the new full-system-dissipation rerun. These are distinct source versions with matched '
        'scientific settings, not one pooled ensemble. Source hashes and residuals are in summary.json.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
