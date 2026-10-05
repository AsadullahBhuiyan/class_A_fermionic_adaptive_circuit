"""Resumable local offline analysis. Run with --phase all inside tmux."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import study_core as C

ROOT,REPO = C.ROOT,C.REPO
PARENT = REPO/'00_WORKSPACE/CURRENT/final_production_new_designs'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def publish(path,arrays,identity,extra=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp)/path.name
        np.savez_compressed(staged,**arrays)
        check = digest(staged)
        with np.load(staged,allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
        os.replace(staged,path)
    assert digest(path) == check
    write_json(path.with_suffix('.json'),dict(identity=identity,file=check,extra=extra or {}))


def valid(path,identity):
    try:
        receipt = json.loads(path.with_suffix('.json').read_text())
        return receipt['identity'] == identity and receipt['file'] == digest(path)
    except (OSError,ValueError,KeyError):
        return False


def load_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def campaign_modules(config):
    result = {}
    for cohort,name in config['cohorts'].items():
        folder = PARENT/name
        load_module('random_center_observer',folder/'random_center_observer.py')
        runner = load_module('campaign_'+cohort,folder/'run_campaign.py')
        data = folder/'data'/runner.REVISION
        plan = json.loads((data/'execution_plan.json').read_text())
        runner.validate_plan(plan,runner.default_config(),runner.identity(runner.default_config()))
        cfg = plan['config']
        assert cfg['DW'] and cfg['dw_truncation'] and cfg['meas_slab_only']
        assert cfg['perfect_correction'] and not cfg['postselect'] and cfg['nshell'] == 1
        assert cfg['alpha_1'] == 1 and cfg['alpha_2'] == 30 and cfg['dtype'] == 'complex128'
        assert cfg['sequence'] == 'raster_y' and cfg['init_mode'] == 'default'
        result[cohort] = runner,data,plan
    return result


def audit_inputs(modules,out):
    rows = []
    for cohort,(runner,data,plan) in modules.items():
        samples = {}
        for task in tqdm(plan['tasks'],desc=f'Hash {cohort} inputs',unit='batch'):
            path = data/(task['id']+'.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            actual = digest(path)
            assert receipt['file'] == actual and receipt['result'] == path.name
            assert receipt['task'] == task and receipt['identity'] == plan['identity']
            key = (task.get('nx',plan['config'].get('nx')),task['ny'])
            samples.setdefault(key,[]).extend(task['sample_ids'])
            rows.append(dict(cohort=cohort,task=task,path=str(path),file=actual,
                             receipt=digest(path.with_suffix('.json')),identity=plan['identity']))
        for ids in samples.values(): np.testing.assert_array_equal(sorted(ids),np.arange(100))
        assert len(samples) == 3
    assert len(rows) == 38
    write_json(out/'input_inventory.json',rows)
    return rows


def read_input(runner,data,plan,task):
    path = data/(task['id']+'.npz')
    with np.load(path,allow_pickle=False) as z:
        arrays = {name:z[name] for name in z.files}
    assert json.loads(str(arrays['metadata_json'])) == dict(task=task,identity=plan['identity'],
        config=plan['config'],entry_point='classA_U1FGTN_gpu.run_markov_circuit')
    runner.validate_arrays(arrays,task,plan['config'])
    return arrays


def reference_set(nx,ny,config,out,identity):
    path = out/'references'/f'nx{nx}_ny{ny}.npz'
    if valid(path,identity):
        with np.load(path,allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata_json']))
            refs = {m['key']:{k:z[m['key']+'__'+k] for k in ('projector','chern','corr_y','corr_x')} for m in metadata}
        return refs,metadata
    print(f'[references] {nx}x{ny}: canonical CPU OW parents, nsh1 and dense',flush=True)
    refs,metadata = C.build_references(nx,ny,config)
    controls = C.reference_controls(refs,metadata,nx,ny)
    arrays = {'metadata_json':np.array(json.dumps(metadata))}
    for key,value in refs.items():
        if isinstance(value,dict):
            for name,a in value.items(): arrays[key+'__'+name] = a
        else: arrays[key] = value
    publish(path,arrays,identity,dict(controls=controls,metadata=metadata))
    return refs,metadata


def analyze_one(arrays,j,task,refs,metadata,config):
    nx,ny = task.get('nx',20),task['ny']
    rank = int(arrays['final_ranks'][j])
    frame = arrays['final_frame'][j,:,:rank]
    result,p = C.endpoint(frame,nx,ny,refs,metadata,config['projector_tolerance'])
    radius = .2*nx
    radii = C.geometry(nx,ny)[4]
    i = list(radii).index(radius)
    ys = arrays['centers_y'][j,-1]
    np.testing.assert_allclose(result['chern_radius'][i,ys],arrays['real_space_chern'][j,-1],atol=2e-9,rtol=2e-9)
    factorized = C.frame_disk_check(frame,nx,ny,radius,int(ys[0]))
    np.testing.assert_allclose(factorized,result['chern_radius'][i,ys[0]],atol=2e-9,rtol=2e-9)
    result['saved_chern_check_error'] = np.array(float(np.max(abs(result['chern_radius'][i,ys]-arrays['real_space_chern'][j,-1]))))
    result['frame_chern_check_error'] = np.array(float(abs(factorized-result['chern_radius'][i,ys[0]])))
    del p
    return result


def benchmark(modules,config,out,identity):
    records = []
    for cohort,(runner,data,plan) in modules.items():
        seen = set()
        for task in plan['tasks']:
            geom = task.get('nx',plan['config'].get('nx')),task['ny']
            if geom in seen: continue
            seen.add(geom)
            refs,metadata = reference_set(*geom,config,out,identity)
            start = time.perf_counter()
            arrays = read_input(runner,data,plan,task)
            load_seconds = time.perf_counter()-start
            start = time.perf_counter()
            result = analyze_one(arrays,0,task,refs,metadata,config)
            seconds = time.perf_counter()-start
            records.append(dict(cohort=cohort,nx=geom[0],ny=geom[1],sample_id=int(arrays['sample_ids'][0]),
                batch_load_validation_seconds=load_seconds,endpoint_seconds=seconds,
                forecast_100_endpoints_seconds=100*seconds,
                maximum_residual=float(np.max(result['diagnostics']))))
            del arrays,refs,result
            print('[benchmark] '+json.dumps(records[-1]),flush=True)
    forecast = sum(r['forecast_100_endpoints_seconds'] for r in records)
    write_json(out/'benchmark.json',dict(identity=identity,records=records,endpoint_forecast_seconds=forecast,
        note='Six measured endpoints; forecast excludes reference construction, repeated batch I/O, and final plotting.'))
    print(f'[forecast] endpoint work {forecast/60:.1f} minutes; plus batch I/O and aggregation',flush=True)


def run_batches(modules,inventory,config,out,identity):
    lookup = {(r['cohort'],r['task']['id']):r for r in inventory}
    done = 0
    with tqdm(total=38,desc='Study batches',unit='batch') as outer:
        for cohort,(runner,data,plan) in modules.items():
            previous_geom,refs,metadata = None,None,None
            for task in plan['tasks']:
                geom = task.get('nx',plan['config'].get('nx')),task['ny']
                if geom != previous_geom:
                    refs,metadata = reference_set(*geom,config,out,identity)
                    previous_geom = geom
                batch_identity = dict(study=identity,input=lookup[cohort,task['id']],
                                      reference=digest(out/'references'/f'nx{geom[0]}_ny{geom[1]}.npz'))
                path = out/'batches'/cohort/(task['id']+'.npz')
                if valid(path,batch_identity):
                    done += 1;outer.update();outer.set_postfix(skipped_or_completed=done);continue
                arrays = read_input(runner,data,plan,task)
                start = time.perf_counter()
                products = []
                for j in tqdm(range(len(task['sample_ids'])),desc=f'{cohort} {geom}',unit='endpoint',leave=False):
                    products.append(analyze_one(arrays,j,task,refs,metadata,config))
                payload = {key:np.stack([r[key] for r in products]) for key in products[0]}
                for key in ('cycles','sample_ids','center_average','centers_y','real_space_chern','global_charge'):
                    payload[key] = arrays[key]
                payload['radii'] = C.geometry(*geom)[4]
                payload['reference_names'] = np.array([r['key'] for r in metadata])
                payload['q_slab'] = payload['global_charge']-payload['q_exterior'][:,None]
                ns = len(C.geometry(*geom)[2])
                assert np.all((payload['q_slab'] >= 0) & (payload['q_slab'] <= ns))
                publish(path,payload,batch_identity,dict(elapsed_seconds=time.perf_counter()-start,
                    endpoint_cycle=int(arrays['cycles'][-1]),task=task,cohort=cohort))
                done += 1;outer.update();outer.set_postfix(skipped_or_completed=done)
                print(f'[saved] {cohort}/{task["id"]}: {done}/38 verified analysis batches',flush=True)
                del arrays,payload,products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('benchmark','run','report','all'),default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'results/v1')
    args = parser.parse_args()
    config = json.loads((ROOT/'config.json').read_text())
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    identity = dict(config=config,sources={str(p.relative_to(REPO)):digest(p) for p in
        (ROOT/'study_core.py',ROOT/'run_study.py',REPO/'src/fgtn/classA_U1FGTN.py',REPO/'src/fgtn/occupied_frame.py')})
    existing = out/'study_identity.json'
    if existing.exists() and json.loads(existing.read_text()) != identity:
        raise RuntimeError('Study source/config changed; use a new output directory to preserve results')
    write_json(existing,identity)
    with threadpool_limits(limits=config['threads']):
        if args.phase in ('all','benchmark','run'):
            modules = campaign_modules(config)
            inventory = audit_inputs(modules,out)
            if args.phase in ('all','benchmark'):
                benchmark(modules,config,out,identity)
            if args.phase in ('all','run'):
                run_batches(modules,inventory,config,out,identity)
        if args.phase in ('all','report'):
            from report_study import report
            report(out)
    print(f'[complete] phase={args.phase}; {out}',flush=True)


if __name__ == '__main__':
    main()
