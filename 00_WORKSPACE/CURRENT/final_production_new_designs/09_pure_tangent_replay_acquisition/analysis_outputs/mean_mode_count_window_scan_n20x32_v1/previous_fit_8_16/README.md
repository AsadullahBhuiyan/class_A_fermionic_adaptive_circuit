# Mean mode count: spectral-window scan

{
  "Nx": 20,
  "Ny": 32,
  "samples": 100,
  "origins": 32,
  "cycle": 64,
  "alpha_1": 1,
  "alpha_2": 30,
  "nshell": 1,
  "construction": "hard",
  "initialization": "pure; exterior product frame",
  "sequence": "raster_y",
  "windows": [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999
  ],
  "fit_widths": [
    8,
    16
  ],
  "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit (saved inputs only)",
  "estimator": "mean over origins inside each trajectory, then mean over trajectories; no normalization by mode count",
  "spectrum_cache": "/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v2/subsystem_spectra.npz",
  "spectrum_cache_sha256": "8cd3aa2eee6df5a92d958f9df2025e530d9a3d62c9e33efb66c7fd129e71d260",
  "output": "/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/mean_mode_count_window_scan_n20x32_v1"
}

Fit model: mean N_L = a(L) + b(L) log[(32/pi) sin(pi Ay/32)]. Coefficient uncertainties are SEMs of fits to the 100 trajectory-level origin averages. All fits are descriptive, with correlated widths and windows.

      L  intercept  intercept_sem    slope  slope_sem  R_squared  residual_rms  max_abs_residual  fit_min_width  fit_max_width
0.10000   1.168112       0.052458 0.012662   0.024360   0.049902      0.006370          0.012961              8             16
0.20000   2.223546       0.068098 0.089883   0.031022   0.783904      0.005441          0.013701              8             16
0.30000   3.371655       0.061568 0.142769   0.027968   0.820814      0.007690          0.014578              8             16
0.40000   4.619438       0.044370 0.168220   0.020008   0.968093      0.003521          0.006571              8             16
0.50000   5.930971       0.062960 0.226408   0.028779   0.942987      0.006418          0.009803              8             16
0.60000   7.392242       0.043870 0.291995   0.019613   0.957980      0.007050          0.012087              8             16
0.70000   9.129806       0.054502 0.367604   0.024431   0.992896      0.003585          0.008190              8             16
0.80000  11.433373       0.051509 0.450984   0.023443   0.988479      0.005613          0.010991              8             16
0.90000  15.139003       0.048757 0.604116   0.021731   0.993260      0.005737          0.009806              8             16
0.95000  18.891010       0.049910 0.723902   0.021833   0.996209      0.005149          0.007545              8             16
0.99000  27.115527       0.058452 1.164643   0.022992   0.998834      0.004588          0.006794              8             16
0.99900  36.955059       0.064173 1.655054   0.027507   0.999183      0.005457          0.010146              8             16
0.99990  50.721530       0.067581 2.212548   0.028146   0.999560      0.005351          0.009170              8             16
0.99999  59.256257       0.077857 2.822734   0.034004   0.999259      0.008864          0.012814              8             16

Outputs: mean_mode_counts.csv; window_fit_summary.csv; window_scan_statistics.npz (counts, trajectory averages, coefficients and joint covariance); input_provenance.json; window_scan_diagnostics.json; five PDF/PNG figure pairs, including the standalone L=0.99 fit and the three-window near-endpoint overlay; the earlier L=0.9 figure is also preserved. The validated spectrum cache stays in the adjacent source directory.
