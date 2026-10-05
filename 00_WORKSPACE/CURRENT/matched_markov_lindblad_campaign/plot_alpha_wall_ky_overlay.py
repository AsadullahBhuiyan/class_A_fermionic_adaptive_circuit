"""Overlay alpha_1=1 and 3 using the verified four-panel spectrum data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'analysis_outputs/alpha1_vs3_hard_soft_full_system_ky'


def main():
    path=OUT/'spectra.npz'
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,
                         'xtick.direction':'in','ytick.direction':'in'})
    with np.load(path,allow_pickle=False) as z:
        ky=z['ky']
        for family,title in [('markov_channel','Quantum channel'),('lindblad','Lindblad + dephasing')]:
            fig,axes=plt.subplots(1,2,figsize=(7.05,3.05),sharex=True,sharey=True)
            for col,wall in enumerate(('hard','soft')):
                ax=axes[col]
                for alpha,color,marker,size in [(1,'#2468ad','o',13),(3,'#c0392b','^',9)]:
                    occ=z[f'{family}_alpha{alpha}_{wall}_occupations']
                    assert occ.shape==(64,40) and np.all(np.isfinite(occ))
                    ax.scatter(np.repeat(ky/np.pi,40),occ.ravel(),s=size,
                        marker=marker,facecolors='none',edgecolors=color,
                        linewidths=.6,label=rf'$\alpha_1={alpha}$',zorder=3 if alpha==3 else 2)
                ax.axhline(.5,color='.5',ls='--',lw=.8,zorder=0)
                ax.set(title=wall.capitalize()+' wall',xlabel=r'$k_y/\pi$',
                       xlim=(-1.03,1.03),ylim=(-.035,1.035))
                ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([0,.25,.5,.75,1])
                ax.tick_params(top=True,right=True)
                ax.text(-.13,1.05,f'({chr(97+col)})',transform=ax.transAxes)
                ax.legend(loc='center left',frameon=False,markerscale=1.5,handletextpad=.3)
            axes[0].set_ylabel(r'Occupation $\nu_a(k_y)$')
            fig.suptitle(title+r': $20\times64$, $n_{\rm shell}=1$, endpoint 128',fontsize=10)
            fig.tight_layout()
            for ext in ('png','pdf'):
                fig.savefig(OUT/f'{family}_alpha_overlay_1x2.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)
    caption=(OUT/'caption.txt').read_text()
    caption=('Overlay presentation: left hard wall, right soft wall. Blue open circles '
             'denote alpha_1=1; red open triangles denote alpha_1=3. No data are changed '
             'from the four-panel spectra. All markers denote sampled momenta, without '
             'interpolation. The scientific/data provenance caption follows.\n\n'+caption)
    (OUT/'overlay_caption.txt').write_text(caption)
    sources={}
    for source in (path,OUT/'summary.json'):
        with source.open('rb') as f:sources[source.name]=hashlib.file_digest(f,'sha256').hexdigest()
    (OUT/'overlay_manifest.json').write_text(json.dumps({'input_sha256':sources,
        'presentation':'two panels per family, alpha1=1 blue circles, alpha1=3 red triangles'},indent=2)+'\n')


if __name__=='__main__':main()
