Large N20 Entanglement Scaling Colab Bundle

This bundle contains Colab-ready GPU notebooks for pure-state Markov runs and
strip-entanglement log-chord slope characterization. The notebooks assume an
A100 GPU with 40 GB GPU RAM and end by disconnecting the Colab runtime.

Repository policy notes:
- Circuit dynamics are run only through classA_U1FGTN_gpu.run_markov_circuit(...).
- The bundled src/classA_U1FGTN_gpu.py is copied from the canonical
  src/fgtn/classA_U1FGTN_gpu.py and should stay synchronized.
- The notebooks do not save covariance histories. They stream covariance
  batches through cycle observers, compute entanglement contour observables,
  and save only contour batches, entropy curves, fit rows, figures, and
  metadata.

Notebooks:
- notebooks/run_slope_vs_system_size.ipynb
  Runs Nx=20, nshell=1, dw_truncation=True, samples=100, cycles=50 for
  Ny in [30, 40, 50, 60, 80, 100, 120]. It computes sample- and y0-averaged
  strip entanglement contours at cycle 50 and extracts log-chord slopes with
  fit window Ay=8..Ny//2.

- notebooks/run_slope_vs_cycle.ipynb
  Runs Nx=20, Ny=40, nshell=1, dw_truncation=True, samples=100, cycles=100.
  It computes sample- and y0-averaged strip entanglement contours for every
  cycle 0..100 and extracts the log-chord slope at each cycle with fit window
  Ay=8..Ny//2.

Domain-wall geometry:
For Nx=20, the class uses half=Nx//2=10 and w=floor(0.2*Nx)=4, so
DW_loc=[6, 14]. The alpha_1 topological slab is x=6..14 inclusive; the
alpha_2 trivial regions are x=0..5 and x=15..19.

Output roots:
- gpu_data/pure_state_entanglement_slope_vs_system_size/
- gpu_data/pure_state_entanglement_slope_vs_cycle/

Contour data are saved in Ay and cycle batches under each run directory.
Checkpoint files allow partial progress inspection and resumable downstream
post-processing.
