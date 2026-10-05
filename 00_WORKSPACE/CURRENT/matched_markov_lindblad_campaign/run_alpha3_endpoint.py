"""Matched endpoints with full-system dissipation or optional frozen exterior."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from tqdm.auto import tqdm
from matched_model import build_model, REPO
from campaign_schema import _matched_seed, sha256_file
from observables import exact_y_twirl, ky_spectrum_and_x_weights
from src.fgtn.diagnostics.mean_lindblad import PerfectCorrectionLindblad

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('markov_channel','lindblad'),required=True)
    p.add_argument('--wall',choices=('hard','soft'),required=True)
    p.add_argument('--alpha-1',type=float,choices=(1.,3.),default=3.)
    p.add_argument('--sequence',choices=('random','raster_y'),default='random')
    p.add_argument('--hard-exterior',choices=('evolve','frozen'),default='evolve')
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--cpu',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    out=args.run_root.resolve()
    out.mkdir(parents=True,exist_ok=False)
    nx,ny,walls=(8,4,[2,5]) if args.smoke else (20,64,[5,15])
    cycles=2*ny; hard=args.wall=='hard'
    frozen=hard and args.hard_exterior=='frozen'
    seed=_matched_seed(20260814,f'N{nx}x{ny}:random-site-word')
    config=dict(schema='raster_y_channel_endpoints_v1' if args.sequence=='raster_y' else 'alpha3_endpoints_v2',Nx=nx,Ny=ny,
        wall_locations=walls,alpha_1=args.alpha_1,alpha_2=30.,nshell=1,
        wall=args.wall,dw_truncation=hard,meas_slab_only=frozen,
        hard_exterior=args.hard_exterior,evolution_domain='slab_only' if frozen else 'full_system',
        active_initialization='maxmix',exterior_preparation='Born conditioned from maxmix' if frozen else 'none',
        exterior_seed=2026091403 if frozen else None,perfect_correction=True,dephasing=True,dtype='complex128',
        family=args.family,cycles=cycles,dt=.05,site_schedule=args.sequence,schedule_seed=seed,
        channel_order=['Ap','Am','Bp','Bm'],
        late_average_cycles=[ny+1,cycles],smoke=args.smoke,
        canonical_entry_point='classA_U1FGTN.run_'+('markov_channel' if args.family=='markov_channel' else 'lindblad_evolution'),
        hard_schedule_note=('Fixed raster_y: y increases at fixed x, then x increases; same word every cycle'
            if args.sequence=='raster_y' else 'Uniform random permutations of slab centers; not paired site words with full-system soft runs'
            if frozen else 'All physical centers; same random site words for hard and soft channel runs'),
        conditional_ensemble='Active measurement outcomes averaged analytically, conditional on one saved exterior preparation' if frozen else 'Measurement outcomes averaged analytically')
    sources={str(path.relative_to(REPO)):sha256_file(path) for path in [Path(__file__),
        HERE/'matched_model.py',HERE/'observables.py',HERE/'campaign_schema.py',
        REPO/'src/fgtn/classA_U1FGTN.py',REPO/'src/fgtn/occupied_frame.py',
        REPO/'src/fgtn/diagnostics/mean_lindblad.py']}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps({'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'cpu':args.cpu,'sources':sources},indent=2)+'\n')
    print(json.dumps(config,indent=2),flush=True)
    started=time.monotonic()
    try:
        parent=build_model({'model':dict(Nx=nx,Ny=ny,domain_wall=True,
            wall_locations=walls,alpha_run_in=args.alpha_1,alpha_run_out=30.,trial_orbitals='X',
            nshell=1,dw_truncation=hard)})
        if frozen:
            model,active=parent.restrict_ow_dynamics_to_slab()
        else:
            model=parent;active=np.arange(parent.Nlayer)
        outside=np.setdiff1d(np.arange(parent.Nlayer),active)
        exterior=np.zeros(parent.Nlayer)
        if frozen:
            # Maxmix is a product of independent Bernoulli(1/2) occupations.
            # Match canonical preparation order (x,y,orbital), with no feedback.
            rng=np.random.default_rng(config['exterior_seed'])
            for x,y in parent._exterior_site_coordinates():
                for mu in (0,1): exterior[2*(y*nx+x)+mu]=rng.random()<.5
        print(f'[evolve] {args.family} {args.wall}: active modes={len(active)}, frozen exterior modes={len(outside)}',flush=True)
        dim=len(active); identity=np.eye(dim,dtype=complex)
        checkpoints={0,ny//2,ny,3*ny//2,cycles}
        eigenvalues=[];spectral_times=[];charge=[];distance=[];words=[]
        late=np.zeros((dim,dim),complex);previous=None
        def observe(cycle,g):
            nonlocal previous,late
            charge.append(float(np.trace(g).real+exterior.sum()))
            distance.append(0. if previous is None else float(np.linalg.norm(g-previous)/np.sqrt(dim)))
            previous=g.copy()
            if cycle>ny: late+=g/ny
            if cycle in checkpoints:
                eig=np.linalg.eigvalsh(g)
                if eig.min() < -1e-10 or eig.max() > 1+1e-10:
                    raise ValueError(f'Unphysical occupation range at {cycle}: {eig.min()}, {eig.max()}')
                eigenvalues.append(eig);spectral_times.append(cycle)
        if args.family=='markov_channel':
            with tqdm(total=cycles,desc=f'{args.wall} channel cycles',unit='cycle') as bar:
                def observer(**payload):
                    cycle=int(payload['cycle']); g=(payload['G']+identity)/2
                    observe(cycle,g)
                    if cycle:
                        ids=np.asarray(payload['ordered_site_ids'])
                        assert len(ids)==model.Nx*ny and len(np.unique(ids))==len(ids)
                        if args.sequence=='raster_y':
                            expected=np.asarray([x+model.Nx*y for x in range(model.Nx) for y in range(ny)])
                            np.testing.assert_array_equal(ids,expected)
                        words.append(ids);bar.update(1)
                result=model.run_markov_channel(cycles=cycles,init_mode='maxmix',
                    sequence=args.sequence,schedule_seed=seed,perfect_correction=True,decoh=True,
                    G_history=False,save=False,progress=False,cycle_observer=observer)
                final=(result['G_final']+identity)/2
                assert result['run_config']['channel_order']==config['channel_order']
        else:
            result=model.run_lindblad_evolution(cycles=cycles,dt=.05,init_mode='maxmix',
                include_number_dephasing=True,observation_times=np.arange(cycles+1),
                representation='q0',progress=True)
            generator=PerfectCorrectionLindblad.from_canonical_model(model)
            for cycle,blocks in enumerate(tqdm(result['correlation_history'],desc='Observe Lindblad cycles',unit='cycle')):
                g=generator.q_sector_to_dense(blocks,q_index=0)
                observe(cycle,g)
            final=g
        assert len(charge)==cycles+1
        def embed(g):
            full=np.diag(exterior).astype(complex)
            full[np.ix_(active,active)]=g
            if frozen:
                np.testing.assert_array_equal(full[outside][:,outside],np.diag(exterior[outside]))
                np.testing.assert_array_equal(full[np.ix_(outside,active)],0)
            return full
        full_final=embed(final);full_late=embed(late)
        print('[save] Building endpoint spectra and writing covariance products',flush=True)
        # The random frozen exterior breaks translation symmetry. Keep its raw
        # full covariance AND label the momentum spectrum explicitly as twirled.
        twirl=exact_y_twirl(full_final,nx,ny)
        ky,occ,weights=ky_spectrum_and_x_weights(twirl,nx,ny)
        payload=dict(G_final=full_final[None],G_late_cycle_average=full_late[None],
            G_final_twirl=twirl[None],active_G_final=final,active_indices=active,
            frozen_exterior_indices=outside,frozen_exterior_occupations=exterior[outside],
            cycles=np.arange(cycles+1),global_charge=np.asarray(charge)[None],
            successive_state_distance=np.asarray(distance)[None],
            spectral_cycles=np.asarray(spectral_times),active_occupations=np.stack(eigenvalues),
            schedule_words=np.asarray(words,dtype=np.int64),ky=ky,twirled_ky_occupations=occ,
            twirled_ky_x_weights=weights,config_json=np.asarray(json.dumps(config)))
        temporary=out/'observables.tmp.npz';dest=out/'observables.npz'
        np.savez_compressed(temporary,**payload)
        digest=sha256_file(temporary)
        with np.load(temporary) as z:
            np.testing.assert_array_equal(z['G_final'][0],full_final)
        temporary.replace(dest)
        receipt=dict(status='complete',config=config,sources=sources,result_filename=dest.name,
            result_bytes=dest.stat().st_size,result_sha256=digest,elapsed_seconds=time.monotonic()-started,
            completed_utc=datetime.now(timezone.utc).isoformat())
        temp=out/'completion.tmp.json';temp.write_text(json.dumps(receipt,indent=2)+'\n')
        temp.replace(out/'completion.json')
        (out/'status.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(f'[complete] {dest} elapsed={receipt["elapsed_seconds"]:.1f}s',flush=True)
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps({'status':'failed','error':repr(exc)},indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
