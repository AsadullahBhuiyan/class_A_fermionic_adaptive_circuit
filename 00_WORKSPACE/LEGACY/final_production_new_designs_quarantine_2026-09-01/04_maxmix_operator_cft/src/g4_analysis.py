"""Whole-trajectory aggregation and nested scientific verdicts for G4."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _fit(x,y,power):
    X=np.column_stack((np.ones(len(x)),np.asarray(x,dtype=float)**(-power))); beta=np.linalg.lstsq(X,y,rcond=None)[0]; residual=y-X@beta; rss=max(float(residual@residual),np.finfo(float).tiny); n=len(y); k=2; return {"intercept":float(beta[0]),"slope":float(beta[1]),"rss":rss,"aicc":float(n*np.log(rss/n)+2*k+(2*k*(k+1)/(n-k-1) if n>k+1 else np.inf))}
def bootstrap(values,replicates,seed,statistic=np.mean):
    values=np.asarray(values); rng=np.random.default_rng(seed); draws=np.asarray([statistic(values[rng.integers(0,len(values),len(values))]) for _ in range(replicates)]); return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]
def _ci(values): return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]
def _relative_shift(value, reference): return float(abs(value-reference)/max(abs(reference),1e-15))
def _bootstrap_size_fits(sample_sets,nys,power,reps,rng):
    fits=[]
    for _ in range(reps):
        means=[np.mean(v[rng.integers(0,len(v),len(v))]) for v in sample_sets]
        fits.append(_fit(nys,np.asarray(means),power))
    return fits
def _fit_l2_l4(nys,values):
    X=np.column_stack((np.ones(len(nys)),nys.astype(float)**-2,nys.astype(float)**-4)); b=np.linalg.lstsq(X,values,rcond=None)[0]
    return {"intercept":float(b[0]),"slope_l2":float(b[1]),"slope_l4":float(b[2])}
def load_archive(root):
    groups={}; seen=set()
    for manifest_path in sorted(Path(root).glob("G4_*/shard_*.manifest.json")):
        manifest=json.loads(manifest_path.read_text()); data=manifest_path.with_name(manifest_path.name.replace(".manifest.json",".npz"))
        if manifest.get("status")!="complete_local" or manifest.get("output_sha256")!=_sha256(data): raise RuntimeError(f"invalid G4 shard {data}")
        with np.load(data,allow_pickle=False) as archive: row={k:archive[k] for k in archive.files}
        ids=row["global_sample_ids"].tolist()
        if seen.intersection(ids): raise RuntimeError("duplicate G4 global sample IDs")
        seen.update(ids); key=(manifest["case"]["arm"],manifest["case"]["model"]["Ny"]); groups.setdefault(key,[]).append(row)
    return {key:{name:np.concatenate([r[name] for r in rows],axis=0) if rows[0][name].ndim else rows[0][name] for name in rows[0]} for key,rows in groups.items()}
def analyze(root,config):
    data=load_archive(root); nys=np.asarray(config["contract"]["Ny_values"]); reps=config["analysis"]["bootstrap_replicates"]; seed=config["analysis"]["bootstrap_seed"]
    complete=all((arm,int(ny)) in data and len(data[(arm,int(ny))]["global_sample_ids"])==25 for arm in ("wall","matched_trivial") for ny in nys)
    if complete and seen_ids(data) != set(range(250)):
        raise RuntimeError("G4 has missing or unexpected global sample IDs")
    result={"schema":"g4_nested_analysis_v1","data_status":"complete" if complete else "incomplete","verdicts":{}}
    if not complete: return result
    rng=np.random.default_rng(seed)
    wall_samples=[data[("wall",int(ny))]["neutral_gap"][:,-1]/(2*ny) for ny in nys]
    trivial_samples=[data[("matched_trivial",int(ny))]["neutral_gap"][:,-1]/(2*ny) for ny in nys]
    wall_gap=np.asarray(list(map(np.mean,wall_samples))); trivial_gap=np.asarray(list(map(np.mean,trivial_samples)))
    wall_fit=_fit(nys,wall_gap,1); trivial_fit=_fit(nys,trivial_gap,1); wall_const={"aicc":float(len(nys)*np.log(max(np.sum((wall_gap-wall_gap.mean())**2),np.finfo(float).tiny)/len(nys))+2)}
    wall_boot=_bootstrap_size_fits(wall_samples,nys,1,reps,rng); trivial_boot=_bootstrap_size_fits(trivial_samples,nys,1,reps,rng)
    wall_intercept_ci=_ci([v["intercept"] for v in wall_boot]); trivial_intercept_ci=_ci([v["intercept"] for v in trivial_boot])
    prefix_slopes=[]
    for fraction in (1.0,1.5,2.0):
        means=[]
        for ny in nys:
            row=data[("wall",int(ny))]; index=int(np.flatnonzero(row["checkpoint_cycles"]==int(fraction*ny))[0]); means.append(np.mean(row["neutral_gap"][:,index]/int(fraction*ny)))
        prefix_slopes.append(_fit(nys,np.asarray(means),1)["slope"])
    prefix_shift=max(_relative_shift(v,prefix_slopes[-1]) for v in prefix_slopes[:-1])
    delete_shift=_relative_shift(_fit(nys[1:],wall_gap[1:],1)["slope"],wall_fit["slope"])
    gap_pass=wall_intercept_ci[0]<=0<=wall_intercept_ci[1] and trivial_intercept_ci[0]>0 and wall_const["aicc"]-wall_fit["aicc"]>=config["analysis"]["aicc_preference_minimum"] and prefix_shift<=.1 and delete_shift<=.1
    # Interface localization uses a one-cell neighborhood of both exact domain-wall positions.
    weights=[]; wall_balance=[]; soft_fraction=[]
    wall_x=(5,15); wall_masks=[]
    for x in wall_x:
        mask=np.zeros(20,bool); mask[[(x-1)%20,x,(x+1)%20]]=True; wall_masks.append(mask)
    mask=wall_masks[0]|wall_masks[1]
    for ny in nys:
        row=data[("wall",int(ny))]; vectors=row["soft_eigenvectors"][:,-1]
        xw=(np.abs(vectors)**2).reshape(len(vectors),ny,20,2,16).sum(axis=(1,3)).transpose(0,2,1)
        subspace=xw.sum(axis=1); left=subspace[:,wall_masks[0]].sum(1); right=subspace[:,wall_masks[1]].sum(1); total=subspace.sum(1)
        weights.extend(((left+right)/total).tolist()); wall_balance.extend((np.minimum(left,right)/(left+right)).tolist())
        q=row["amplitude_cost"][:,-1]; contribution=.25/np.cosh(np.clip(q,0,350))**2; soft_fraction.extend((np.sort(contribution,axis=1)[:,-16:].sum(1)/contribution.sum(1)).tolist())
    loc_ci=bootstrap(weights,reps,seed); balance_ci=bootstrap(wall_balance,reps,seed+1); frac_ci=bootstrap(soft_fraction,reps,seed+2); mechanism=loc_ci[0]>.7 and balance_ci[0]>=config["analysis"]["two_wall_balance_minimum"] and frac_ci[0]>.7
    result["verdicts"]["G4a_purification_localization"]={"pass":bool(mechanism),"interface_weight_ci95":loc_ci,"two_wall_balance_ci95":balance_ci,"soft16_impurity_fraction_ci95":frac_ci}
    result["verdicts"]["G4b_operator_gap"]={"pass":bool(gap_pass),"wall_fit":wall_fit,"wall_intercept_ci95":wall_intercept_ci,"matched_trivial_fit":trivial_fit,"matched_trivial_intercept_ci95":trivial_intercept_ci,"delta_aicc":wall_const["aicc"]-wall_fit["aicc"],"prefix_shift":prefix_shift,"delete_smallest_shift":delete_shift}
    # Tower and Casimir are intentionally nested secondary verdicts; never overwrite G4a/gap.
    k_by_size=[]; k_samples=[]; v_by_size=[]; descendant_residuals=[]
    for ny in nys:
        levels=data[("wall",int(ny))]["charge_sector_levels"][:,-1]; neutral=np.mean(levels[:,3,1]); plus=np.mean(levels[:,4,0]); minus=np.mean(levels[:,2,0]); v_by_size.append(ny*neutral/(2*np.pi*2*ny)); per=levels[:,3,1]/(levels[:,4,0]+levels[:,2,0]); k_samples.append(per); k_by_size.append(np.mean(per))
        for sample in levels:
            spacing=sample[3,1]
            for sector in (2,3,4):
                finite=sample[sector,np.isfinite(sample[sector])][:8]; shifted=(finite-finite[0])/spacing; descendant_residuals.extend((shifted-np.rint(shifted)).tolist())
    k=float(np.mean(k_by_size)); q_asym=float(np.mean([abs(data[("wall",int(ny))]["charge_sector_levels"][:,-1,4,0].mean()-data[("wall",int(ny))]["charge_sector_levels"][:,-1,2,0].mean())/max(data[("wall",int(ny))]["charge_sector_levels"][:,-1,[2,4],0].mean(),1e-15) for ny in nys]))
    k_ci=bootstrap(np.concatenate(k_samples),reps,seed+3); descendant_rms=float(np.sqrt(np.mean(np.square(descendant_residuals))))
    k_prefix=[]
    for fraction in (1.0,1.5,2.0):
        values=[]
        for ny in nys:
            row=data[("wall",int(ny))]; index=int(np.flatnonzero(row["checkpoint_cycles"]==int(fraction*ny))[0]); levels=row["charge_sector_levels"][:,index]; values.extend((levels[:,3,1]/(levels[:,4,0]+levels[:,2,0])).tolist())
        k_prefix.append(float(np.mean(values)))
    k_prefix_shift=max(_relative_shift(value,k_prefix[-1]) for value in k_prefix[:-1]); k_delete=float(np.mean(k_by_size[1:])); kpass=k_ci[0]<=1<=k_ci[1] and abs(k-1)<=.1 and q_asym<=.1 and descendant_rms<=.15 and _relative_shift(k_delete,k)<=.1 and k_prefix_shift<=.1
    result["verdicts"]["G4c_operator_tower_k"]={"pass":bool(kpass),"k_op":k,"ci95":k_ci,"charge_asymmetry":q_asym,"descendant_rms_spacings":descendant_rms,"prefix_shift":k_prefix_shift,"delete_smallest_shift":_relative_shift(k_delete,k)}
    free=[]
    for ny in nys:
        w=data[("wall",int(ny))]["leading_log_sigma"][:,-1].mean()/(2*ny); t=data[("matched_trivial",int(ny))]["leading_log_sigma"][:,-1].mean()/(2*ny); free.append(-(w-t)/(2*ny))
    casimir=_fit(nys,np.asarray(free),2); v=float(np.mean(v_by_size)); c=float(-12*casimir["slope"]/(np.pi*v)) if v else np.nan
    c_draws=[]
    for _ in range(reps):
        free_draw=[]; v_draw=[]
        for ny in nys:
            wr=data[("wall",int(ny))]; tr=data[("matched_trivial",int(ny))]; wi=rng.integers(0,25,25); ti=rng.integers(0,25,25)
            free_draw.append(-(wr["leading_log_sigma"][wi,-1].mean()/(2*ny)-tr["leading_log_sigma"][ti,-1].mean()/(2*ny))/(2*ny)); v_draw.append(wr["neutral_gap"][wi,-1].mean()/(4*np.pi))
        fit_draw=_fit(nys,np.asarray(free_draw),2); vd=np.mean(v_draw); c_draws.append(-12*fit_draw["slope"]/(np.pi*vd) if vd else np.nan)
    c_ci=_ci(np.asarray(c_draws)[np.isfinite(c_draws)]); c_delete=-12*_fit(nys[1:],np.asarray(free)[1:],2)["slope"]/(np.pi*np.mean(v_by_size[1:])); c_l4=-12*_fit_l2_l4(nys,np.asarray(free))["slope_l2"]/(np.pi*v)
    cpass=np.isfinite(c) and c_ci[0]<=1<=c_ci[1] and not(c_ci[0]<=0<=c_ci[1]) and abs(c-1)<=.2 and _relative_shift(c_delete,c)<=.2 and _relative_shift(c_l4,c)<=.2
    result["verdicts"]["G4d_operator_casimir"]={"pass":bool(cpass),"c_eff_op":c,"ci95":c_ci,"fit":casimir,"v_op":v,"delete_smallest_shift":_relative_shift(c_delete,c),"l4_shift":_relative_shift(c_l4,c)}
    result["state_cft_comparison_policy"]="k_state,c_state are separate observables; equality to k_op,c_eff_op is not tested"
    return result
def seen_ids(data):
    return {int(value) for row in data.values() for value in row["global_sample_ids"]}
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--archive-root",type=Path,required=True); p.add_argument("--bundle-root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--output",type=Path); a=p.parse_args(argv); config=json.loads((a.bundle_root/"production_config.json").read_text()); result=analyze(a.archive_root,config); output=a.output or a.archive_root/"g4_analysis_summary.json"; output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(output)
if __name__=="__main__": main()
