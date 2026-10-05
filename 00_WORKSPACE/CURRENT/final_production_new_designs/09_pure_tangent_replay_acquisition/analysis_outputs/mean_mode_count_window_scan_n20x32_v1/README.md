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
    5,
    16
  ],
  "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit (saved inputs only)",
  "estimator": "mean over origins inside each trajectory, then mean over trajectories; no normalization by mode count",
  "spectrum_cache": "/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v2/subsystem_spectra.npz",
  "spectrum_cache_sha256": "8cd3aa2eee6df5a92d958f9df2025e530d9a3d62c9e33efb66c7fd129e71d260",
  "output": "/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/analysis_outputs/mean_mode_count_window_scan_n20x32_v1",
  "percentage_summary": {
    "Ay": 16,
    "full_system_modes": 1280,
    "subsystem_modes": 640,
    "definition": "100 times mean central count at Ay=16 divided by 1280 full-system modes (or 640 subsystem modes)"
  }
}

Fit model: mean N_L = a(L) + b(L) log[(32/pi) sin(pi Ay/32)]. Coefficient uncertainties are SEMs of fits to the 100 trajectory-level origin averages. All fits are descriptive, with correlated widths and windows.

      L  intercept  intercept_sem    slope  slope_sem  R_squared  residual_rms  max_abs_residual  fit_min_width  fit_max_width  percentage_reference_Ay  mean_modes_at_reference_Ay  mean_modes_sem_at_reference_Ay  percent_of_full_system_modes  percent_of_full_system_modes_sem  percent_of_subsystem_modes  percent_of_subsystem_modes_sem
0.10000   1.111501       0.020514 0.038174   0.010132   0.588983      0.007608          0.015068              5             16                       16                    1.196250                        0.012246                      0.093457                          0.000957                    0.186914                        0.001913
0.20000   2.243548       0.025896 0.081062   0.012411   0.931211      0.005256          0.014857              5             16                       16                    2.430625                        0.012338                      0.189893                          0.000964                    0.379785                        0.001928
0.30000   3.357642       0.023793 0.148971   0.010939   0.964696      0.006799          0.015339              5             16                       16                    3.707500                        0.012013                      0.289648                          0.000939                    0.579297                        0.001877
0.40000   4.627064       0.020727 0.164639   0.010043   0.990452      0.003857          0.007079              5             16                       16                    5.009375                        0.010600                      0.391357                          0.000828                    0.782715                        0.001656
0.50000   5.923572       0.024044 0.229814   0.011763   0.988941      0.005798          0.009361              5             16                       16                    6.453750                        0.013979                      0.504199                          0.001092                    1.008398                        0.002184
0.60000   7.357675       0.020496 0.307496   0.009036   0.992577      0.006344          0.014668              5             16                       16                    8.067500                        0.011808                      0.630273                          0.000922                    1.260547                        0.001845
0.70000   9.104471       0.024901 0.378885   0.011002   0.998261      0.003773          0.010247              5             16                       16                    9.981875                        0.014305                      0.779834                          0.001118                    1.559668                        0.002235
0.80000  11.386852       0.020022 0.471908   0.008018   0.997686      0.005422          0.009866              5             16                       16                   12.476250                        0.015861                      0.974707                          0.001239                    1.949414                        0.002478
0.90000  15.087735       0.024036 0.627313   0.009381   0.998285      0.006202          0.010606              5             16                       16                   16.533125                        0.020823                      1.291650                          0.001627                    2.583301                        0.003254
0.95000  18.768175       0.029439 0.779166   0.010755   0.998054      0.008208          0.013148              5             16                       16                   20.564375                        0.026224                      1.606592                          0.002049                    3.213184                        0.004098
0.99000  27.064999       0.032880 1.187414   0.013417   0.999696      0.004936          0.009119              5             16                       16                   29.811875                        0.034042                      2.329053                          0.002660                    4.658105                        0.005319
0.99900  36.756321       0.037311 1.744830   0.016041   0.998904      0.013791          0.027935              5             16                       16                   40.793125                        0.045198                      3.186963                          0.003531                    6.373926                        0.007062
0.99990  50.467882       0.037760 2.327072   0.018193   0.999108      0.016592          0.032382              5             16                       16                   55.860625                        0.038444                      4.364111                          0.003003                    8.728223                        0.006007
0.99999  58.785782       0.049868 3.035396   0.022359   0.997862      0.033521          0.074731              5             16                       16                   65.816875                        0.059221                      5.141943                          0.004627                   10.283887                        0.009253

Outputs: all_windows_summary.csv (all windows with half-system central counts and full/subsystem percentages); mean_mode_counts.csv; window_fit_summary.csv; window_scan_statistics.npz (counts, trajectory averages, coefficients and joint covariance); input_provenance.json; window_scan_diagnostics.json; five PDF/PNG figure pairs, including the standalone L=0.99 fit and the three-window near-endpoint overlay; the earlier L=0.9 figure is also preserved. The validated spectrum cache stays in the adjacent source directory.
