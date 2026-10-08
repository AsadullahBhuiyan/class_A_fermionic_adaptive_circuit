"""Derived deterministic estimators, fit tables, and convention checks."""
import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import benchmark as b


def disk_sectors(nx,ny,radius,cy):
    index=np.arange(nx*ny)
    x,y=index%nx,index//nx
    dx=(x-nx/2+nx/2)%nx-nx/2
    dy=(y-cy+ny/2)%ny-ny/2
    angle=np.mod(np.arctan2(dy,dx),2*np.pi)
    inside=dx*dx+dy*dy<=radius**2
    return tuple((2*np.flatnonzero(inside & (angle>=s*2*np.pi/3) & (angle<(s+1)*2*np.pi/3))[:,None]+[0,1]).ravel() for s in range(3))


def disk_chern(g,nx,ny,radius,cy):
    a,c,d=disk_sectors(nx,ny,radius,cy)
    return float(-24*np.pi*np.trace(g[np.ix_(d,a)] @ g[np.ix_(a,c)] @ g[np.ix_(c,d)]).imag)


def occupied_frame(data):
    v=data['vectors'];ny,q,_=v.shape
    phase=np.exp(2j*np.pi*np.arange(ny)[:,None]*np.arange(ny)[None,:]/ny)/np.sqrt(ny)
    return np.einsum('yk,kai->yaki',phase,v).reshape(ny*q,ny*q)[:,data['occupied'].ravel()]


def local_marker(frame,nx,ny):
    u=frame.conj()
    index=np.arange(2*nx*ny)
    x,y=(index//2)%nx,index//(2*nx)
    xx=u.conj().T @ (x[:,None]*u)
    yy=u.conj().T @ (y[:,None]*u)
    diagonal=np.einsum('ij,ij->i',(u @ xx) @ yy,u.conj())
    return (-4*np.pi*diagonal.imag).reshape(ny,nx,2).sum(2)


def topology():
    rows=[];saved={}
    for n in tqdm((20,30,40),desc='Square-system topology',unit='size'):
        data,diag=b.load_case(n,n,1.)
        g=b.dense_from_delta(data['projector_delta'],n).T
        values=np.array([disk_chern(g,n,n,.2*n,y) for y in range(n)])
        mean=float(values.mean())
        row=dict(L=n,x_left=b.interfaces(n)[0],x_right=b.interfaces(n)[1],radius=.2*n,
                 chern=mean,signed_deviation=mean-1,absolute_deviation=abs(mean-1),
                 center_spread=float(np.ptp(values)),fermi_degenerate_count=diag['fermi_degenerate_count'])
        if diag['fermi_degenerate_count']:
            alt=b.dense_from_delta(data['alternative_projector_delta'],n).T
            alt_values=np.array([disk_chern(alt,n,n,.2*n,y) for y in range(n)])
            row['alternative_chern_difference']=float(alt_values.mean()-mean)
            saved[f'alternative_chern_L{n}']=alt_values
        else:
            row['alternative_chern_difference']=0.
        rows.append(row); saved[f'chern_L{n}']=values
        if n==30:
            marker=local_marker(occupied_frame(data),n,n)
            assert abs(marker.sum())<1e-7
            saved['marker']=marker
            saved['marker_display']=np.tanh(marker)
    b.csv_write(b.HERE/'data/chern_table.csv',rows)
    b.npz_write(b.HERE/'data/topology.npz',**saved)
    return rows


def chord(n,width):
    return n/np.pi*np.sin(np.pi*np.asarray(width)/n)


def joint_fit(curves,intercept=False):
    xs=[];ys=[];ws=[]
    for x,y in curves:
        xs.extend(x);ys.extend(y);ws.extend(np.full(len(x),1/len(x)))
    x,y,w=map(np.asarray,(xs,ys,ws))
    design=np.column_stack((x,np.ones_like(x))) if intercept else x[:,None]
    coef=np.linalg.lstsq(design*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]
    residual=y-design@coef
    # Conventional centered weighted R^2, never interpreted as a sampling error.
    total=float(np.sum(w*(y-np.average(y,weights=w))**2))
    return dict(slope=float(coef[0]),intercept=float(coef[1]) if intercept else 0.,
                weighted_r_squared=(1-float(np.sum(w*residual**2))/total) if total>0 else None,
                weighted_rms_residual=float(np.sqrt(np.average(residual**2,weights=w))),
                max_absolute_residual=float(abs(residual).max()),points=len(x),sizes=len(curves))


def fit_observables():
    results={};missing=[]
    curves=[]
    for ny in b.CORR_SIZES:
        data,_=b.load_case(20,ny,1)
        corr=data['correlation']; r=np.arange(2,ny//2+1)
        # Resolution threshold is recorded; the denominator is never replaced.
        if corr[-1]<=1e-24:
            missing.append(ny);continue
        selected=(r>=8)&(corr[r]>0)
        curves.append((np.log(np.sin(np.pi*r[selected]/ny)),np.log(corr[r[selected]]/corr[-1])))
    if curves:
        results['correlations']=joint_fit(curves)
        results['correlations']['beta']=-results['correlations']['slope']
    results['undefined_antipodal_sizes']=missing
    for key,sizes,prefactor in [('entropy',b.ENTROPY_SIZES,3.),('variance',b.ENTROPY_SIZES,np.pi**2),
                                ('wall_left',b.WALL_SIZES,3.),('wall_right',b.WALL_SIZES,3.)]:
        curves=[]
        for ny in sizes:
            data,_=b.load_case(20,ny,1)
            widths=data['widths']; select=widths>=8
            x=np.log(chord(ny,widths)/chord(ny,ny//2))
            y=data[key]-data[key][-1]
            curves.append((x[select],y[select]))
        results[key]=joint_fit(curves)
        results[key]['converted_coefficient']=prefactor*results[key]['slope']
    data,_=b.load_case(20,32,1)
    w=data['widths']; select=w>=5
    results['mode_count']=joint_fit([(np.log(chord(32,w[select])),data['mode_count'][select])],intercept=True)
    b.json_write(b.HERE/'data/fits.json',results)
    return results


def modular_profile(delta,cutoff):
    ny,q,_=delta.shape; width=ny//2;nx=q//2
    p=b.dense_from_delta(delta,width)
    nu,v=eigh(p,driver='evd')
    centered=np.clip(2*nu-1,-1+cutoff,1-cutoff)
    epsilon=-2*np.arctanh(centered)
    times=np.linspace(0,1,101)
    phase=np.exp(-1j*times[:,None]*epsilon)
    sources=(5,15);packets=[];drifts=[];weights=[];norm_error=zero_error=0.
    for x in sources:
        cells=width//2*q+2*x+np.arange(2)
        coefficients=v[cells,:].conj().T
        evolved=np.einsum('ij,tj,ja->tia',v,phase,coefficients,optimize=True)
        norm_error=max(norm_error,float(abs(np.sum(abs(evolved)**2,axis=1)-1).max()))
        initial=np.zeros((width*q,2),complex);initial[cells,np.arange(2)]=1
        zero_error=max(zero_error,float(abs(evolved[0]-initial).max()))
        density=np.sum(abs(evolved)**2,axis=2).reshape(len(times),width,nx,2).sum(3)
        window=density[:,:,x-2:x+3]
        retained=window.sum((1,2));assert retained.min()>0
        drift=np.sum(window*(np.arange(width)-width//2)[None,:,None],axis=(1,2))/retained
        packets.append(density[[0,10,20]])
        drifts.append(drift);weights.append(retained)
    return dict(times=times,snapshot_times=np.array([0.,.1,.2]),density=np.array(packets),
                displacement=np.array(drifts),retained_weight=np.array(weights)),dict(
                norm_residual=norm_error,initial_profile_residual=zero_error,
                zero_displacement_residual=float(abs(np.array(drifts)[:,0]).max()))


def modular():
    arrays={};diagnostics={}
    for alpha in (1,3):
        data,_=b.load_case(20,32,alpha)
        for cutoff in (1e-8,1e-10,1e-12):
            result,check=modular_profile(data['projector_delta'],cutoff)
            key=f'a{alpha}_eps{int(-np.log10(cutoff))}'
            for name,value in result.items():arrays[f'{key}_{name}']=value
            diagnostics[key]=check
        main=arrays[f'a{alpha}_eps10_displacement']
        diagnostics[f'alpha{alpha}_cutoff_max_displacement_change']={
            str(c):float(abs(arrays[f'a{alpha}_eps{c}_displacement']-main).max()) for c in (8,12)}
        for c in (8,10,12):
            d=arrays[f'a{alpha}_eps{c}_displacement']
            diagnostics[f'a{alpha}_eps{c}']['displacement_extrema']=[[float(v.min()),float(v.max())] for v in d]
    b.npz_write(b.HERE/'data/modular.npz',**arrays)
    b.json_write(b.HERE/'data/modular_checks.json',diagnostics)
    return diagnostics


def histograms():
    arrays={};rows=[]
    edges=np.linspace(0,1,101);ee=np.linspace(-np.log(199),np.log(199),102)
    for a in (1,3):
        data,_=b.load_case(20,32,a)
        nu=data['half_nu'];mask=abs(2*nu-1)<.99
        energy=np.log((1-nu[mask])/nu[mask])
        hc=np.histogram(nu,edges)[0];he=np.histogram(energy,ee)[0]
        # All 32 translations have identical spectra. Counts are per cut; pooling
        # translations multiplies numerator and normalization by the same factor.
        rho=hc/(len(nu)*np.diff(edges));rho_e=he/(mask.sum()*np.diff(ee))
        assert hc.sum()==len(nu) and he.sum()==mask.sum()
        assert abs(np.dot(rho,np.diff(edges))-1)<1e-12
        assert abs(np.dot(rho_e,np.diff(ee))-1)<1e-12
        arrays.update({f'occupation_density_a{a}':rho,f'energy_density_a{a}':rho_e,
                       f'occupation_counts_a{a}':hc,f'energy_counts_a{a}':he})
        rows.append(dict(alpha_1=a,occupation_count_per_cut=len(nu),retained_count_per_cut=int(mask.sum()),
                         translated_cuts=32,occupation_pooled_count=32*len(nu),retained_pooled_count=32*int(mask.sum())))
    arrays.update(occupation_edges=edges,energy_edges=ee)
    b.npz_write(b.HERE/'data/histograms.npz',**arrays)
    b.csv_write(b.HERE/'data/histogram_counts.csv',rows)


def tables():
    gap=[];mi=[];curves=[]
    for ny in b.GAP_SIZES:
        data,diag=b.load_case(20,ny,1)
        gap.append(dict(Ny=ny,minimum_absolute_energy=diag['minimum_absolute_energy'],
                        half_filling_gap=diag['half_filling_gap'],resolution=diag['degeneracy_tolerance'],
                        resolved=diag['minimum_absolute_energy']>diag['degeneracy_tolerance']))
    for ny in (20,24,28):
        for alpha in b.ALPHAS:
            _,diag=b.load_case(20,ny,alpha)
            mi.append(dict(Ny=ny,alpha_1=alpha,width=ny//4,mutual_information=diag['mutual_information']))
    for nx,ny,a in b.task_table():
        data,_=b.load_case(nx,ny,a)
        if 'entropy' in data:
            for j,w in enumerate(data['widths']):
                curves.append(dict(Nx=nx,Ny=ny,alpha_1=a,Ay=int(w),entropy=float(data['entropy'][j]),
                                   charge_variance=float(data['variance'][j]),mode_count=int(data['mode_count'][j])))
    b.csv_write(b.HERE/'data/gaps.csv',gap)
    b.csv_write(b.HERE/'data/mutual_information.csv',mi)
    b.csv_write(b.HERE/'data/strip_curves.csv',curves)


def reconciliation():
    """Common-input comparison with the documented historical occupation regulator."""
    periodic,pdiag=b.load_case(20,30,1)
    blocks,_=b.build_parent(20,30,1,occupation_regulator=1e-7)
    regulated,rdiag=b.diagonalize(blocks)
    diff=float(abs(regulated['projector_delta']-periodic['projector_delta']).max())
    s0=float(b.entropy_values(b.restricted(periodic['projector_delta'],15)[0]).sum())
    s1=float(b.entropy_values(b.restricted(regulated['projector_delta'],15)[0]).sum())
    result=dict(classification='protocol or estimator mismatch',
                prior_reference='00_WORKSPACE/CURRENT/final_production_new_designs/24_flattened_imaginary_time_density',
                common_input=b.configuration(20,30,1),periodic_gap=pdiag['half_filling_gap'],
                legacy_occupation_regulator=1e-7,regulated_gap=rdiag['half_filling_gap'],
                max_projector_delta_difference=diff,periodic_half_strip_entropy=s0,regulated_half_strip_entropy=s1,
                explanation='The primary calculations use strict periodic boundaries. The legacy convention evaluates representative OW columns at k+phi/Ny before periodic reconstruction. It resolves the edge crossing differently without changing the clean underlying parameters. No historical result is overwritten.')
    b.json_write(b.HERE/'data/prior_protocol_comparison.json',result)

    # Reconcile the apparent failure of the circuit's antipodal normalization
    # with the existing strictly periodic equilibrium calculation on common data.
    prior=b.REPO/'00_WORKSPACE/CURRENT/final_production_new_designs/06_domain_wall_flattened_ground_state_reference'
    output=prior/'analysis_outputs/wall_projected_ground_state_nx20_ny60_v1'
    current,_=b.load_case(20,60,1)
    with np.load(output/'wall_projected_ground_state_data.npz',allow_pickle=False) as z:
        old={k:z[k] for k in z.files}
    r=np.arange(1,30)
    amplitude=np.sqrt(2*current['column_correlation'][5,r])
    mask=(r>=5)&(r<=25)
    coordinate=abs(1/np.tan(np.pi*r/60))
    fit=joint_fit([(np.log(coordinate[mask]),np.log(amplitude[mask]))],intercept=True)
    result=dict(classification='protocol or estimator mismatch',
                explanation='The periodic parent reproduces the earlier clean equilibrium result. The earlier wall amplitude uses |cot(pi*r/Ny)|, while the manuscript trajectory estimator fits antipodally normalized squared correlations to a chord power. Squaring doubles the fitted exponent; the cotangent suppression makes the antipodal normalization poorly conditioned as a scaling comparison.',
                prior_files={str(p.relative_to(b.REPO)):b.sha(p) for p in (
                    output/'wall_projected_ground_state_data.npz',output/'summary.json',
                    prior/'analyze_wall_projected_ground_state.py',prior/'src/classA_U1FGTN.py')},
                common_geometry=b.configuration(20,60,1),
                projector_delta_max_difference=float(abs(old['projector_delta']-current['projector_delta']).max()),
                wall_amplitude_max_difference=float(abs(old['correlation_left']-amplitude).max()),
                half_strip_entropy_old=float(old['entropy_full'][-1]),
                half_strip_entropy_new=float(current['entropy'][-1]),
                fit_window=[5,25],wall_amplitude_cotangent_fit=fit,
                squared_wall_correlation_exponent=2*fit['slope'],
                entropy_difference_note='The historical contour clips all occupations to [1e-12,1-1e-12]; the new entropy uses exact 0log0 limits after roundoff clipping only.')
    b.json_write(b.HERE/'data/prior_periodic_comparison.json',result)


def run(threads):
    with threadpool_limits(limits=threads):
        topology(); fit_observables(); histograms(); tables()
        print('Computing modular profiles and cutoff checks',flush=True)
        modular(); reconciliation()
    print('Derived arrays, fits, tables, and prior-protocol comparison completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=4)
    run(parser.parse_args().threads)
