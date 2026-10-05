#!/usr/bin/env python3
"""Four-panel purification figure: alpha3 control and one slowest mode/sample."""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from plot_purification_gap_density_summary import read_rows, sha, SOURCE
from plot_purification_gap_summary import ROOT, make_figure
from plot_endpoint_lyapunov_gap import write_csv

OUT = ROOT/'analysis_outputs/purification_control_minmode_4x1_v2_single_column'
CONTROL = ROOT.parent/'20_hard_wall_alpha3_purification/gpu_data/hard_wall_alpha3_maxmix_nx20_ny40_s100_4ny_v1'
MODES = ROOT.parents[1]/'experiment_review/purification_slow_mode_profiles'


def alpha_control():
    manifest=json.loads((CONTROL/'DOWNLOAD_MANIFEST.json').read_text())
    config=manifest['resolved_configuration']
    assert config['alpha_1']==3 and config['Ny_values']==[40]
    assert config['perfect_correction'] and not config['postselect']
    files={row['path']:row for row in manifest['files']}
    ids, parts, inputs = [], [], []
    for relative in tqdm(sorted(files),desc='Verify control originals',unit='file'):
        path=CONTROL/relative; row=files[relative]
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        inputs.append(dict(path=str(path),sha256=row['sha256']))
        if path.suffix!='.npz': continue
        receipt=json.loads(path.with_suffix('.complete.json').read_text())
        assert receipt['result_sha256']==row['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert str(z['configuration_hash'])==manifest['configuration_hash']==receipt['configuration_hash']
            assert float(z['alpha_1'])==3 and int(z['Ny'])==40
            np.testing.assert_array_equal(z['cycles'],np.arange(161))
            ids.extend(z['sample_indices'].tolist()); parts.append(z['total_entropy'])
    order=np.argsort(ids); samples=np.concatenate(parts)[order]
    np.testing.assert_array_equal(np.array(ids)[order],np.arange(100))
    assert samples.shape==(100,161) and np.isfinite(samples).all() and (samples>0).all()
    normalized=samples/40
    mean=normalized.mean(axis=0); sem=normalized.std(axis=0,ddof=1)/10
    rows=[dict(alpha_1=1,**r) for r in read_rows('total_entropy_curves.csv') if r['Ny']==40]
    rows += [dict(alpha_1=3,bundle=20,Ny=40,cycle=t,normalized_cycle=t/40,samples=100,
                  mean_entropy_over_Ny=float(mean[t]),sem=float(sem[t])) for t in range(161)]
    assert len(rows)==322
    return rows, samples, inputs


def select_minimum_mode(occupations, rates, vectors, full_indices, ny=30):
    """Select the smallest magnitude, not the most-negative signed exponent."""
    calculated=(np.log1p(-occupations)-np.log(occupations))/(2*4*ny)
    np.testing.assert_allclose(calculated,rates,rtol=1e-12,atol=1e-14)
    j=int(np.argmin(abs(rates)))
    # No degenerate minimum in this dataset; refuse an arbitrary tie choice.
    assert np.count_nonzero(np.isclose(abs(rates),abs(rates[j]),rtol=1e-10,atol=1e-12))==1
    probability=np.zeros(2*20*ny)
    probability[full_indices]=abs(vectors[:,j])**2
    density=probability.reshape(ny,20,2).sum(axis=2)
    np.testing.assert_allclose(density.sum(),1,atol=1e-12)
    return j,density


def minimum_mode_density(ny=30):
    manifest=json.loads((MODES/'analysis_manifest.json').read_text())
    expected={r['result_file']:r['result_sha256'] for r in manifest['outputs']}
    densities, records, inputs = [], [], []
    for sample in tqdm(range(100),desc='Select min |lambda| mode',unit='sample'):
        path=MODES/f'endpoint_modes/hard/Ny{ny:03d}/sample_{sample:03d}.npz'
        digest=sha(path); assert digest==expected[str(path.relative_to(MODES))]
        with np.load(path,allow_pickle=False) as z:
            assert int(z['sample_index'])==sample and int(z['Ny'])==ny and int(z['cycles'])==4*ny
            assert str(z['construction'])=='hard' and float(z['alpha_1'])==1
            nu=z['finite_occupations']; rates=z['finite_signed_rates']; full=z['finite_full_spectrum_indices']
            j,density=select_minimum_mode(nu,rates,z['finite_eigenvectors'],z['active_indices'],ny)
            assert z['finite_mode_individually_separated'][j]
            # Check that the selected mode minimizes over the entire saved spectrum.
            all_nu=z['occupations']; mask=(all_nu>1e-9)&(all_nu<1-1e-9)
            all_costs=abs(np.log1p(-all_nu[mask])-np.log(all_nu[mask]))/(8*ny)
            np.testing.assert_allclose(abs(rates[j]),all_costs.min(),atol=1e-14)
            np.testing.assert_allclose(density.sum(axis=0),z['finite_mode_x_profiles'][j],atol=1e-12)
            records.append(dict(sample_index=sample,full_spectrum_index=int(full[j]),
                occupation=float(nu[j]),signed_lambda=float(rates[j]),absolute_lambda=float(abs(rates[j])),
                eigenvector_residual=float(z['finite_eigensolver_residuals'][j])))
            densities.append(density)
        inputs.append(dict(path=str(path),sha256=digest))
    samples=np.stack(densities); mean=samples.mean(axis=0)
    assert not np.any(mean[:,:5]) and not np.any(mean[:,16:])
    np.testing.assert_allclose(mean.sum(),1,atol=1e-12)
    return mean,samples,records,inputs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,control,inputs=alpha_control()
    mean,densities,modes,mode_inputs=minimum_mode_density()
    summary=json.loads((SOURCE/'analysis_summary.json').read_text())
    fit=summary['weighted_log_space_power_law_fit']
    make_figure(rows,read_rows('spatial_entropy_curves.csv'),read_rows('gap_summary.csv'),fit,
        density=mean,density_vmax=float(mean.max()),output=OUT,alpha_comparison=True,smallest_mode_density=True)
    write_csv(OUT/'total_entropy_alpha_comparison.csv',rows)
    write_csv(OUT/'selected_minimum_modes.csv',modes)
    np.savez_compressed(OUT/'Ny030_minimum_mode_density.npz',sample_densities=densities,
        mean_density=mean,sem_density=densities.std(axis=0,ddof=1)/10,sample_indices=np.arange(100))
    np.savez_compressed(OUT/'alpha3_sample_entropy.npz',total_entropy=control,sample_indices=np.arange(100),cycles=np.arange(161))
    inputs += mode_inputs + [dict(path=str(SOURCE/name),sha256=sha(SOURCE/name)) for name in
                            ['total_entropy_curves.csv','spatial_entropy_curves.csv','gap_summary.csv','analysis_summary.json']]
    record=dict(figure_size_inches=[3.375,6.8],alpha_comparison_Ny=40,samples_per_alpha=100,
        uncertainty='sample-wise SEM, no bootstrap',panel_b_c='Unchanged data and fit from previous figure',
        panel_d_Ny=30,panel_d_estimator='One argmin_j |lambda_j| eigenmode per trajectory; sum orbital probabilities, then equal mean of 100 trajectories',
        selected_modes_per_sample=1,degenerate_minima=0,probability_sum=float(mean.sum()),
        density_vmax=float(mean.max()),positive_selected_rates=sum(r['signed_lambda']>0 for r in modes),
        negative_selected_rates=sum(r['signed_lambda']<0 for r in modes),fit=fit,inputs=inputs,
        alpha3_entropy_floor_note='Observer clips occupations to [1e-12,1-1e-12] for entropy. Its near-zero tail is an estimator floor, not residual mixedness.',
        script_sha256=sha(Path(__file__)))
    (OUT/'analysis_summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(OUT)
    print('Selected one nondegenerate minimum-magnitude mode for each of 100 samples.')


if __name__=='__main__': main()
