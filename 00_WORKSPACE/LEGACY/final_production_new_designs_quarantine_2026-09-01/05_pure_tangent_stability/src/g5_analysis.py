"""Independent pure-trajectory tangent-stability analysis for G5."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fit(x,y,power=1):
    X=np.column_stack((np.ones(len(x)),np.asarray(x,float)**(-power))); b=np.linalg.lstsq(X,y,rcond=None)[0]; rss=max(float(np.sum((y-X@b)**2)),np.finfo(float).tiny); n=len(y); return {"intercept":float(b[0]),"slope":float(b[1]),"aicc":float(n*np.log(rss/n)+4+(12/(n-3) if n>3 else np.inf))}
def ci(values): return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]
def shift(value,reference): return float(abs(value-reference)/max(abs(reference),1e-15))
def analyze(root,config):
    groups={}; seen=set()
    for mp in sorted(Path(root).glob("G5_*/shard_*.manifest.json")):
        m=json.loads(mp.read_text()); p=mp.with_name(mp.name.replace(".manifest.json",".npz"))
        if m.get("status")!="complete_local" or m.get("output_sha256")!=sha(p): raise RuntimeError("invalid G5 shard")
        with np.load(p,allow_pickle=False) as z: row={k:z[k] for k in z.files}
        ids=row["global_sample_ids"].tolist()
        if seen.intersection(ids): raise RuntimeError("duplicate G5 IDs")
        seen.update(ids); groups.setdefault((m["case"]["arm"],m["case"]["model"]["Ny"]),[]).append(row)
    groups={k:{n:np.concatenate([r[n] for r in rows],axis=0) if rows[0][n].ndim else rows[0][n] for n in rows[0]} for k,rows in groups.items()}
    nys=np.asarray(config["contract"]["Ny_values"]); complete=all((a,int(n)) in groups and len(groups[(a,int(n))]["global_sample_ids"])==25 for a in ("wall","matched_trivial") for n in nys)
    if complete and {int(value) for row in groups.values() for value in row["global_sample_ids"]} != set(range(250)): raise RuntimeError("G5 has missing or unexpected global sample IDs")
    result={"schema":"g5_nested_analysis_v1","data_status":"complete" if complete else "incomplete","verdicts":{}}
    if not complete:return result
    wall_samples=[np.min(np.abs(groups[("wall",int(n))]["pair_rates"][:,-1]),axis=1) for n in nys]; control_samples=[np.min(np.abs(groups[("matched_trivial",int(n))]["pair_rates"][:,-1]),axis=1) for n in nys]
    wall=np.asarray(list(map(np.mean,wall_samples))); control=np.asarray(list(map(np.mean,control_samples))); wf=fit(nys,wall); cf=fit(nys,control); constant_rss=max(float(np.sum((wall-wall.mean())**2)),np.finfo(float).tiny); constant_aicc=float(len(nys)*np.log(constant_rss/len(nys))+2+4/(len(nys)-2)); daicc=constant_aicc-wf["aicc"]
    rng=np.random.default_rng(config["analysis"]["bootstrap_seed"]); wall_intercepts=[]; control_intercepts=[]
    for _ in range(config["analysis"]["bootstrap_replicates"]):
        wall_intercepts.append(fit(nys,np.asarray([np.mean(v[rng.integers(0,len(v),len(v))]) for v in wall_samples]))["intercept"]); control_intercepts.append(fit(nys,np.asarray([np.mean(v[rng.integers(0,len(v),len(v))]) for v in control_samples]))["intercept"])
    wall_ci=ci(wall_intercepts); control_ci=ci(control_intercepts)
    temporal_slopes=[]
    for checkpoint in range(4): temporal_slopes.append(fit(nys,np.asarray([np.mean(np.min(np.abs(groups[("wall",int(n))]["pair_rates"][:,checkpoint]),axis=1)) for n in nys]))["slope"])
    temporal_shift=max(shift(value,temporal_slopes[-1]) for value in temporal_slopes[:-1]); delete_shift=shift(fit(nys[1:],wall[1:])["slope"],wf["slope"])
    residual=max(float(np.max(groups[(a,int(n))]["selected_residuals"])) for a in ("wall","matched_trivial") for n in nys)
    weights=[]; balances=[]; masks=[]
    for center in (5,15):
        mask=np.zeros(20,bool); mask[[(center-1)%20,center,(center+1)%20]]=True; masks.append(mask)
    for n in nys:
        x=groups[("wall",int(n))]["pair_x_density"][:,-1].sum(axis=1); left=x[:,masks[0]].sum(1); right=x[:,masks[1]].sum(1); total=x.sum(1); weights.extend(((left+right)/total).tolist()); balances.extend((np.minimum(left,right)/(left+right)).tolist())
    means=[np.mean(np.asarray(weights)[rng.integers(0,len(weights),len(weights))]) for _ in range(config["analysis"]["bootstrap_replicates"])]; balance_means=[np.mean(np.asarray(balances)[rng.integers(0,len(balances),len(balances))]) for _ in range(config["analysis"]["bootstrap_replicates"])]; loc_ci=ci(means); balance_ci=ci(balance_means)
    passed=residual<=1e-8 and wall_ci[0]<=0<=wall_ci[1] and control_ci[0]>0 and daicc>=4 and loc_ci[0]>.7 and balance_ci[0]>=config["selection"]["two_wall_balance_minimum"] and temporal_shift<=.1 and delete_shift<=.1
    result["verdicts"]["G5_pure_tangent_stability"]={"pass":bool(passed),"wall_fit":wf,"wall_intercept_ci95":wall_ci,"matched_trivial_fit":cf,"matched_trivial_intercept_ci95":control_ci,"delta_aicc":daicc,"maximum_core_residual":residual,"interface_weight_ci95":loc_ci,"two_wall_balance_ci95":balance_ci,"temporal_shift":temporal_shift,"delete_smallest_shift":delete_shift}
    result["cross_gate_policy"]="independent ensembles; no recordwise or numerical equality test against G4"
    return result
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--archive-root",type=Path,required=True); p.add_argument("--bundle-root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--output",type=Path); a=p.parse_args(argv); result=analyze(a.archive_root,json.loads((a.bundle_root/"production_config.json").read_text())); out=a.output or a.archive_root/"g5_analysis_summary.json"; out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(out)
if __name__=="__main__": main()
