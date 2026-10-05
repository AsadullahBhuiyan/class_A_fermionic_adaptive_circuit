"""Checksum-resumable runner for redesigned G4."""

from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, tempfile, threading, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from g4_observables import CycleProbabilityAccumulator, G4Observer, observation_cycles

BUNDLE = "04_maxmix_operator_cft"
REVISION = "maxmix_operator_cft_v1"
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
HEARTBEAT_SECONDS = 60.0

@contextmanager
def _shard_heartbeat(case_id, shard_index, *, stream=None):
    """Emit newline-delimited liveness updates while one shard is running."""
    interval=float(HEARTBEAT_SECONDS)
    if interval<=0: raise ValueError("HEARTBEAT_SECONDS must be positive")
    stream=sys.stderr if stream is None else stream
    stopped=threading.Event(); started=time.monotonic()
    def emit():
        while not stopped.wait(interval):
            elapsed=time.monotonic()-started
            tqdm.write(f"[{BUNDLE}] heartbeat case={case_id} shard={int(shard_index):02d} elapsed={elapsed:.0f}s",file=stream)
    thread=threading.Thread(target=emit,name=f"{BUNDLE}-heartbeat",daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set(); thread.join()

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def seed(root,*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str,(root,*parts))).encode()).digest()[:8],"little")%(2**63-1)

def source_hashes(bundle_root):
    names=("classA_U1FGTN_gpu.py","occupied_frame_gpu.py","g4_observables.py","g4_runner.py","g4_analysis.py")
    return {name:sha256(Path(bundle_root)/"src"/name) for name in names}

def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False,suffix=".partial") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); tmp=Path(f.name)
    os.replace(tmp,path)

def save_npz(path,arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w+b",dir=path.parent,delete=False,suffix=".partial") as f: np.savez_compressed(f,**arrays); tmp=Path(f.name)
    digest=sha256(tmp); os.replace(tmp,path)
    if sha256(path)!=digest: raise IOError("post-write checksum mismatch")
    return digest

def load_config(root):
    config=json.loads((Path(root)/"production_config.json").read_text())
    if config.get("bundle")!=BUNDLE or config.get("campaign_revision")!=REVISION: raise ValueError("G4 identity changed")
    c=config["contract"]
    required={"Nx":20,"Ny_values":[20,30,40,50,60],"samples_per_case":25,"samples_per_shard":5,"cycles_rule":"2*Ny","initialization":"maxmix","canonical_entry_point":ENTRY_POINT,"tangent_tracked":False,"trajectory_record_saved":False}
    for k,v in required.items():
        if c.get(k)!=v: raise ValueError(f"G4 contract changed: {k}")
    return config

def cases(config,pilot=False):
    nys=config["pilot"]["Ny_values"] if pilot else config["contract"]["Ny_values"]
    samples=config["pilot"]["samples_per_case"] if pilot else config["contract"]["samples_per_case"]
    result=[]
    for ny in nys:
        for arm in ("wall","matched_trivial"):
            p=config["protocols"][arm]
            result.append({"case_id":f"G4_N20x{ny}_{arm}","case_ordinal":len(result),"arm":arm,"samples":samples,"model":{"Nx":20,"Ny":ny,"nshell":1,"filling_frac":0.5,"trial_orbitals":"X","dtype":"complex128","backend":"local",**p},"run":{"cycles":2*ny,"sequence":"random","perfect_correction":True,"postselect":False,"postselect_probability":0.0,"n_a":0.5,"checkpoints":observation_cycles(ny),"strip_cycles":[ny,3*ny//2,2*ny]}})
    return result

def run_shard(bundle_root,config,case,shard_index,output_root,sample_count=None,allow_cpu=False):
    if not torch.cuda.is_available() and not allow_cpu: raise RuntimeError("G4 production requires an A100 GPU")
    if torch.cuda.is_available() and "A100" not in torch.cuda.get_device_name(0).upper() and not allow_cpu: raise RuntimeError("G4 production requires an A100 GPU")
    shard_size=config["contract"]["samples_per_shard"]; start=shard_index*shard_size
    count=min(shard_size,case["samples"]-start) if sample_count is None else int(sample_count)
    if count<=0 or start+count>case["samples"]: raise ValueError("invalid shard")
    directory=Path(output_root)/case["case_id"]; data=directory/f"shard_{shard_index:02d}.npz"; manifest=directory/f"shard_{shard_index:02d}.manifest.json"
    stream_seed=seed(config["root_seed"],case["case_id"],shard_index,"born")
    global_ids=list(range(case["case_ordinal"]*25+start,case["case_ordinal"]*25+start+count))
    identity={"bundle":BUNDLE,"campaign_revision":REVISION,"case":case,"shard_index":shard_index,"sample_start":start,"sample_count":count,"global_sample_ids":global_ids,"trajectory_stream_seed":stream_seed,"root_seed":config["root_seed"],"production_config_sha256":sha256(Path(bundle_root)/"production_config.json"),"source_sha256":source_hashes(bundle_root)}
    if data.exists()!=manifest.exists(): raise RuntimeError("orphan immutable G4 artifact")
    if data.exists():
        old=json.loads(manifest.read_text())
        if any(old.get(k)!=v for k,v in identity.items()) or old.get("output_sha256")!=sha256(data): raise RuntimeError("immutable G4 identity/checksum mismatch")
        return {"status":"verified_existing","data_path":str(data)}
    model=classA_U1FGTN_gpu(device="cuda:0" if torch.cuda.is_available() else "cpu",**case["model"])
    observer=G4Observer(20,case["model"]["Ny"],count,tuple(case["run"]["checkpoints"]),tuple(case["run"]["strip_cycles"]))
    probabilities=CycleProbabilityAccumulator(count,case["run"]["cycles"])
    torch.manual_seed(stream_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(stream_seed)
    begun=time.time()
    metadata=model.run_markov_circuit(G_history=False,progress=False,cycles=case["run"]["cycles"],samples=count,save=False,save_init=False,return_data=False,sequence="random",perfect_correction=True,postselect=False,postselect_probability=0.0,n_a=0.5,meas_slab_only=False,init_mode="maxmix",state_representation="covariance",batch_size=min(5,count),record_observer=probabilities,cycle_observer=observer,cycle_observer_cycles=range(0,case["run"]["cycles"]+1),track_choi=False,choi_observer=None)
    prob=probabilities.arrays(); arrays=observer.arrays(prob["cumulative_log_probability"]); arrays.update(prob)
    arrays.update({"global_sample_ids":np.asarray(global_ids,dtype=np.int64),"trajectory_stream_seed":np.asarray(stream_seed,dtype=np.uint64)})
    if np.max(arrays["soft_eigenpair_residuals"])>config["observations"]["eigenpair_residual_tolerance"]: raise RuntimeError("natural eigenpair residual exceeds contract")
    if np.max(arrays["soft_eigenvector_gram_error"])>config["observations"]["eigenvector_gram_tolerance"]: raise RuntimeError("natural eigenvector Gram error exceeds contract")
    digest=save_npz(data,arrays)
    write_json(manifest,{**identity,"status":"complete_local","canonical_entry_point":ENTRY_POINT,"saved_products":sorted(arrays),"trajectory_record_saved":False,"tangent_tracked":False,"choi_tracked":False,"engine_metadata":str(metadata),"runtime_seconds":time.time()-begun,"host":platform.node(),"output_sha256":digest})
    return {"status":"completed","data_path":str(data),"manifest_path":str(manifest),"sha256":digest}

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("pilot","production","queue","preflight")); p.add_argument("--case-id"); p.add_argument("--shard-index",type=int); p.add_argument("--output-root",type=Path,default=Path("gpu_data")); p.add_argument("--allow-cpu",action="store_true"); a=p.parse_args(argv)
    root=Path(__file__).resolve().parents[1]; config=load_config(root)
    if a.mode=="preflight":
        if not torch.cuda.is_available() and not a.allow_cpu: raise RuntimeError("G4 preflight requires an A100 GPU")
        name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and "A100" not in name.upper() and not a.allow_cpu: raise RuntimeError(f"G4 preflight requires an A100 GPU; detected {name!r}")
        print(json.dumps({"device":name,"allow_cpu":bool(a.allow_cpu)},sort_keys=True)); return 0
    selected=cases(config,pilot=a.mode=="pilot")
    if a.case_id: selected=[c for c in selected if c["case_id"]==a.case_id]
    if not selected: raise KeyError(a.case_id)
    if a.mode=="queue":
        queue=[{"queue_index":i,"case_id":c["case_id"],"shard_index":s} for i,(c,s) in enumerate((c,s) for c in selected for s in range(5))]
        write_json(root/"production_queue.json",{"bundle":BUNDLE,"revision":REVISION,"shard_count":len(queue),"queue":queue}); return 0
    jobs=[]
    for c in selected:
        shard_indices=[0] if a.mode=="pilot" else ([a.shard_index] if a.shard_index is not None else range(5))
        jobs.extend((c,int(s),2 if a.mode=="pilot" else None) for s in shard_indices)
    bar=tqdm(total=len(jobs),desc=f"{BUNDLE} {a.mode}",unit="shard",dynamic_ncols=True,leave=True,file=sys.stderr)
    try:
        for c,s,sample_count in jobs:
            bar.set_postfix_str(f"case={c['case_id']} shard={s:02d}",refresh=True)
            with _shard_heartbeat(c["case_id"],s):
                result=run_shard(root,config,c,s,a.output_root,sample_count=sample_count,allow_cpu=a.allow_cpu)
            if result.get("status") not in {"completed","verified_existing"}: raise RuntimeError(f"G4 shard did not reach a durable state: {result!r}")
            print(json.dumps(result,sort_keys=True),flush=True)
            bar.update(1)
    finally:
        bar.close()
    return 0
