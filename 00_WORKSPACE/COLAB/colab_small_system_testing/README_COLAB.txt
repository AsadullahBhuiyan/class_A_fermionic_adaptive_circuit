Colab pure-state GPU snapshot bundle

Expected layout:
colab_small_system_testing/
  notebooks/
    data_generation/
      run_pure_state_markov_snapshots.ipynb
      run_purification_maxmix_entropy_contours.ipynb
    characterization/
      run_pure_state_strip_entropy_contours.ipynb
      analyze_purification_maxmix_entropy_contours_cpu.ipynb
  src/
    __init__.py
    classA_U1FGTN_gpu.py
    strip_entropy_contours_gpu.py
  gpu_data/
    index.json
    pure_state_covariance_snapshots/
    pure_state_strip_entropy_contours/
    purification_entropy_contours_maxmix/
  analysis_outputs/
    purification_entropy_contours_maxmix_cpu/

Open a notebook in Colab, enable a GPU runtime, and run from the top.

Data generation notebooks:

notebooks/data_generation/run_pure_state_markov_snapshots.ipynb runs four
canonical GPU Markov-circuit configurations:
- Nx = 16
- Ny = 30, 40
- nshell = 1, 2
- alpha_1 = 1
- alpha_2 = 30
- dw_truncation = True
- protocol = perfect_correction
- init_mode = default
- dtype = complex128
- samples = 10
- cycles = 50
- saved covariance snapshots = 5, 10, 20, 50

Outputs are written under:
colab_small_system_testing/gpu_data/pure_state_covariance_snapshots/runs/N16x{Ny}_nsh{nshell}_perfect_correction/

Each run writes sharded snapshot .npy files under a compact run_<run_id> folder
plus a manifest.json from classA_U1FGTN_gpu.run_markov_circuit(...). The full
cache key and run configuration remain in the manifest. The notebook also writes
per-run summary JSON files and a campaign_manifest.json.

notebooks/data_generation/run_purification_maxmix_entropy_contours.ipynb runs
purification dynamics from the maximally mixed state:
- Nx = 16
- Ny = 32
- nshell = 1, 2
- samples = 10 for perfect_correction
- samples = 1 for postselect, because postselect=True is deterministic in the
  GPU driver
- cycles = 100
- protocols = perfect_correction, postselect
- dtype = complex128
- dw_truncation = False
- init_mode = maxmix

It does not save covariance histories. It calls the canonical GPU driver with a
streaming cycle observer, computes the sample-averaged whole-system spatial
entropy contour for cycles 0 through 100, then saves the contour observables
under the active campaign:
colab_small_system_testing/gpu_data/purification_entropy_contours_maxmix/campaigns/N16x32_C100_dwtrunc0/runs/nsh{nshell}_{protocol}/

Purification data is grouped by campaign:
colab_small_system_testing/gpu_data/purification_entropy_contours_maxmix/
  latest_campaign.json
  campaigns/
    N16x30_C50_dwtrunc1/
    N16x32_C100_dwtrunc0/
    N16x32_C100_dwtrunc1/

latest_campaign.json points to N16x32_C100_dwtrunc0 by default. That is the
current intended Nx=16, Ny=32, cycles=100, dw_truncation=False campaign. The
N16x32_C100_dwtrunc1 campaign is kept as a comparison dataset, and
N16x30_C50_dwtrunc1 preserves the older Ny=30, cycles=50,
dw_truncation=True data.

Characterization notebooks:

notebooks/characterization/run_pure_state_strip_entropy_contours.ipynb is a
derived-observable Colab notebook. It does not run new circuit trajectories. It
loads the pure-state covariance snapshots, uses CUDA on an A100 to diagonalize
strip-restricted covariance matrices, and saves sample/y0-averaged
entanglement-contour products for strip widths Ay = 0 through Ny//2 under:
colab_small_system_testing/gpu_data/pure_state_strip_entropy_contours/runs/N16x{Ny}_nsh{nshell}_perfect_correction/

Before computing each nonzero Ay product, the notebook benchmarks candidate y0
chunk sizes on the real first snapshot and uses the fastest safe chunk for the
full run. It also validates that the source campaign contains exactly the four
expected pure-state perfect-correction snapshot configs before any CUDA
computation starts.

Each Ay product is written as:
Ay_{Ay:03d}/strip_entropy_contour_Ay{Ay:03d}.npz

The helper module src/strip_entropy_contours_gpu.py contains only observable
post-processing utilities. It does not simulate circuits.

The top-level gpu_data/index.json lists the campaign manifests:
- pure_state_covariance_snapshots/campaign_manifest.json
- pure_state_strip_entropy_contours/campaign_manifest.json
- purification_entropy_contours_maxmix/latest_campaign.json

notebooks/characterization/analyze_purification_maxmix_entropy_contours_cpu.ipynb
is local-only. It reads purification_entropy_contours_maxmix/latest_campaign.json
unless CAMPAIGN_MANIFEST_OVERRIDE is set to a specific campaign manifest. CPU
analysis tables and figures are namespaced by campaign under:
colab_small_system_testing/analysis_outputs/purification_entropy_contours_maxmix_cpu/N16x32_C100_dwtrunc0/

It does not import Colab, CUDA, torch, classA_U1FGTN_gpu, or run new Markov
dynamics.

Pure-state covariance snapshots and pure-state strip entropy outputs stay in
their existing layouts. Only purification entropy contour data is split into
campaign folders.

The final cell of each Colab notebook disconnects the Colab runtime when the
campaign finishes.
