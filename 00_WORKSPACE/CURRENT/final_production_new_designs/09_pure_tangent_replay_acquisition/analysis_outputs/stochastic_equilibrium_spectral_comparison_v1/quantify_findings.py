from analyze_comparison import *
import pandas as pd

def ratio_error(a,b):
    a=np.asarray(a);b=np.asarray(b);r=a.mean()/b.mean()
    return dict(mean=float(r),sem=float((a-r*b).std(ddof=1)/np.sqrt(len(a))/b.mean()))

def quantify():
    small=dict(np.load(OUT/'matched_spectra.npz'));large=dict(np.load(OUT/'extended_size_controls.npz'))
    spatial=dict(np.load(OUT/'spatial_and_origin_controls_all100.npz'))
    all_spectra={**small,**large};rows=[];fitrows=[];countarrays={};gaps={}
    sizes=[24,28,32,40,50,60];cutoffs=np.linspace(.05,.95,181)
    for ny in sizes:
        for protocol in ['stochastic','equilibrium']+(['quarter_cut'] if ny>=40 else []):
            a=all_spectra[f'{protocol}_{ny}'];m=metrics(a)
            rows.append(dict(Ny=ny,protocol=protocol,**{k:float(v.mean()) for k,v in m.items()},
                             S_sem=ms(m['S1'])['sem'],V_sem=ms(m['V'])['sem'],S_over_V=float(m['S1'].mean()/m['V'].mean())))
    for label,a in [('stochastic_y0',spatial['y0_spectra']),('stochastic_y8',spatial['y8_spectra']),
                    ('stochastic_origin_average',spatial['origin_spectra'].reshape(-1,640)),('equilibrium',spatial['equilibrium_eigenvalues'][None,:])]:
        per=np.array([(abs(a)<q).sum(axis=-1) for q in cutoffs]).T
        if label=='stochastic_origin_average':per=per.reshape(100,16,-1).mean(1)
        y=per.mean(0);countarrays[label]=y;countarrays[label+'_sem']=per.std(0,ddof=1)/np.sqrt(len(per)) if len(per)>1 else np.zeros(len(y))
        for family,x in [('constant_lambda_density',cutoffs),('constant_epsilon_density',np.arctanh(cutoffs))]:
            coeff=float(x@y/(x@x));pred=coeff*x
            fitrows.append(dict(protocol=label,model=family,amplitude=coeff,rmse_levels=float(np.sqrt(np.mean((y-pred)**2))),fit_min=.05,fit_max=.95,fit_points=len(x)))
        gap_arrays=[]
        # For origin-averaged data each trajectory's 16 cuts are kept grouped below.
        for vals in a:
            eps=np.sort(-2*np.arctanh(vals[abs(vals)<np.tanh(1.5)]))
            gap_arrays.append(np.diff(eps))
        gg=np.concatenate(gap_arrays)
        gaps[label]=dict(window='abs(epsilon)<3',gaps=len(gg),median=float(np.median(gg)),fraction_below_1e_minus5=float(np.mean(gg<1e-5)))
    write_csv('size_summary.csv',rows);write_csv('spectral_envelope_fits.csv',fitrows)
    np.savez_compressed(OUT/'bin_free_spectral_counts.npz',cutoffs=cutoffs,**countarrays)
    mm=metrics(spatial['origin_spectra']);eq=metrics(spatial['equilibrium_eigenvalues'])
    ds=mm['S1'][:,0]-mm['S1'][:,8];da=mm['S1'][:,0]-mm['S1'].mean(1)
    ent=spatial['entropy_x'];ent8=spatial['entropy_x_y8'];eqent=spatial['equilibrium_entropy_x']
    state=pd.read_csv(OUT/'endpoint_state_diagnostics.csv')
    slope_rows=[]
    for protocol in ['stochastic','equilibrium']:
        selected=[r for r in rows if r['protocol']==protocol]
        x=np.log([r['Ny'] for r in selected]);X=np.column_stack([np.ones(len(x)),x]);y=np.array([r['S1'] for r in selected])
        if protocol=='stochastic':
            se=np.array([r['S_sem'] for r in selected]);cov=np.linalg.inv(X.T@(X/se[:,None]**2));b=cov@(X.T@(y/se**2));chi=float(np.sum(((y-X@b)/se)**2))
            fit=dict(intercept=float(b[0]),slope=float(b[1]),slope_sem=float(np.sqrt(cov[1,1])),chi_squared=chi,dof=len(x)-2)
        else:
            b=np.linalg.lstsq(X,y,rcond=None)[0];fit=dict(intercept=float(b[0]),slope=float(b[1]),slope_sem=None)
        slope_rows.append(dict(protocol=protocol,Ny_min=24,Ny_max=60,**fit))
    # Independent trajectories: origin averaging is performed before the SEM.
    finding=dict(size_fits=slope_rows,level_gap_diagnostics=gaps,
        entropy_y0=ms(mm['S1'][:,0]),entropy_y8=ms(mm['S1'][:,8]),entropy_origins=ms(mm['S1'].mean(1)),
        variance_y0=ms(mm['V'][:,0]),variance_origins=ms(mm['V'].mean(1)),equilibrium={k:float(v) for k,v in eq.items()},
        paired_entropy_y0_minus_y8=ms(ds),paired_entropy_y0_minus_origin_mean=ms(da),
        excess_removed_by_y8=float(ds.mean()/(mm['S1'][:,0].mean()-eq['S1'])),
        excess_removed_by_origin_average=float(da.mean()/(mm['S1'][:,0].mean()-eq['S1'])),
        S_over_V_y0=ratio_error(mm['S1'][:,0],mm['V'][:,0]),S_over_V_origins=ratio_error(mm['S1'].mean(1),mm['V'].mean(1)),
        S_over_V_equilibrium=float(eq['S1']/eq['V']),flat_lambda_reference=3.,constant_epsilon_reference=float(np.pi**2/3),
        wall_cell_entropy_excess_y0=ms((ent-eqent)[:,[5,15]].sum(1)),
        wall_cell_entropy_excess_y8=ms((ent8-eqent)[:,[5,15]].sum(1)),
        topological_interior_entropy_excess_y8=ms((ent8-eqent)[:,6:15].sum(1)),
        equilibrium_exterior_entropy=float(eqent[:5].sum()+eqent[16:].sum()),
        topological_excess_energy=ms(state.topological_excess_energy),
        origin_entropy_at_half_filled_active_rank=ms(mm['S1'][abs(state.topological_charge-352)<1e-8].mean(1)),
        samples_at_half_filled_active_rank=int((abs(state.topological_charge-352)<1e-8).sum()),
        gaussian_entropy_of_averaged_covariance=float(metrics(spatial['averaged_covariance_spectrum'])['S1']),
        caution='The Gaussian entropy of the averaged two-point matrix is not generally the entropy of the non-Gaussian trajectory mixture.')
    # Construct a sector-matched ground-state control for the deterministic postselected trajectory.
    progress=OUT/'matched_postselection_progress.npz'
    if progress.exists():
        f=np.load(progress)['frame_0'];nx=20;ny=32;n=640
        p,h,e,u,ed=equilibrium(nx,ny)
        idx=np.where((np.arange(2*n)//2%nx>=5)&(np.arange(2*n)//2%nx<=15))[0]
        outside=np.setdiff1d(np.arange(2*n),idx);initial=f@f.conj().T
        rank=int(round(np.trace(initial[np.ix_(idx,idx)]).real));et,ut=eigh(hermitian(h[np.ix_(idx,idx)]),driver='evd')
        ps=np.zeros_like(p);ps[np.ix_(idx,idx)]=ut[:,:rank]@ut[:,:rank].conj().T
        ps[np.ix_(outside,outside)]=initial[np.ix_(outside,outside)]
        l,ent,var,c=reduced(ps,nx,ny,vectors=True)
        np.savez_compressed(OUT/'sector_matched_equilibrium.npz',projector=ps,spectrum=l,entropy_x=ent,active_rank=rank,active_energies=et)
        finding['sector_matched_equilibrium']=dict(active_rank=rank,half_filling_active_rank=352,**{k:float(v) for k,v in metrics(l).items()})
    e=np.load(OUT/'sector_matched_equilibrium.npz')['active_energies']
    ranks=np.rint(state.topological_charge).astype(int)
    cost=np.array([e[:r].sum()-e[:352].sum() for r in ranks])
    finding['topological_excess_energy_same_rank']=ms(state.topological_excess_energy.to_numpy()-cost)
    finding['topological_ground_energy_cost_of_charge_offset']=ms(cost)
    finding['sector_matched_equilibrium']['fermi_gap']=float(e[362]-e[361])
    accepted=OUT/'accepted_postselection_cycle32.npz'
    if accepted.exists():
        z=np.load(accepted);post=metrics(z['spectra_32'])['S1'];st=mm['S1'].mean(0)
        eps=-2*np.arctanh(z['spectra_32'][8][abs(z['spectra_32'][8])<np.tanh(1.5)])
        gg=np.diff(np.sort(eps))
        finding['accepted_postselection_control']=dict(cycle=32,independent_initial_states=1,active_rank=362,
            S_y0=float(post[0]),S_origin_average=float(post.mean()),cut_offset=float(post[0]-post.mean()),
            centered_origin_profile_correlation=float(np.corrcoef(st,post)[0,1]),
            median_epsilon_gap_y8=float(np.median(gg)),fraction_gaps_below_1e_minus5=float(np.mean(gg<1e-5)),
            scope='Finite-cycle deterministic control. Late-time stationary postselection is not established.')
    dump('findings.json',finding)
    return finding
if __name__=='__main__':
    print(json.dumps(quantify(),default=native,indent=2))
