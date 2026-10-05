# Local gain–loss click sequence pilot

This pilot records the four local controller clicks generated from the two
eigenvectors of the selected Pauli trial operator and the upper/lower flattened
Hamiltonian bands. The Pauli basis is fixed to trial_orbitals="X": A is the
+1 eigenvector of Pauli X, while B is the -1 eigenvector. The Ap/Bp channels
are target-empty loss channels and Am/Bm are target-filled gain channels.

The production geometry is fixed to N20x24, nshell=1, DW=True,
dw_truncation=True, meas_slab_only=True, alpha_1=1, and alpha_2=30. Both
schedules use perfect correction, ten trajectories, 48 cycles, a 24-cycle
burn-in, and root seed 20260814.

## Validation smoke run

Run the two-schedule, two-sample, two-cycle schema check with small resampling
counts:

    taskset -c 56-57 python topological_frustration_diagnostics/launch_click_sequence_pilot.py \
      --smoke \
      --output-dir /tmp/local_click_sequence_smoke \
      --cpu-budget 2 --workers 2 \
      --activity-bootstraps 8 \
      --sequence-permutations 8 \
      --sequence-bootstraps 8 \
      --sequence-min-support 1

The launcher runs raster_y first and random second. A failed raster stage
prevents the random stage from starting.

## Production tmux deployment

Do not interrupt the existing campaigns. The launcher below remains idle until
the tfd_nx20_perfect_20260813_121340 session exits and only then creates its
output directory. If the CFT campaign remains active on CPUs 0-55, the pilot
runs on the disjoint affinity 56-111:

    tmux new-session -d -s clicks_N20x24_S10_C48_20260814 \
      "taskset -c 56-111 python topological_frustration_diagnostics/launch_click_sequence_pilot.py \
      --output-dir topological_frustration_diagnostics/results/local_gain_loss_clicks/20260814_N20x24_S10_C48_production \
      --wait-for-tmux-session tfd_nx20_perfect_20260813_121340 \
      --wait-poll-seconds 60"

Monitor it with:

    tmux capture-pane -p -t clicks_N20x24_S10_C48_20260814 -S -80

Every schedule directory contains a stage log and success/failure marker. The
campaign root receives campaign_complete.txt only after both schedules, the
cross-schedule seed check, and automatic execution of the output notebook pass.
The default parent for timestamped runs is
topological_frustration_diagnostics/results/local_gain_loss_clicks/.

## Analysis

The launcher executes an output copy automatically. To regenerate it manually,
set the completed campaign path and run:

    CLICK_CAMPAIGN_ROOT=/absolute/path/to/campaign \
    CLICK_CPU_RANGE=56-65 \
    jupyter nbconvert --to notebook --execute \
      topological_frustration_diagnostics/notebooks/analyze_local_gain_loss_click_sequences.ipynb \
      --output analyze_local_gain_loss_click_sequences.executed.ipynb \
      --output-dir /absolute/path/to/campaign \
      --ExecutePreprocessor.timeout=600

The compressed activity_raw.npz file is the source of truth. The Parquet files
provide channel-level and unit-cell-level tidy views; the sequence NPZ and
candidate CSV contain the permutation-null and trajectory-bootstrap results.
