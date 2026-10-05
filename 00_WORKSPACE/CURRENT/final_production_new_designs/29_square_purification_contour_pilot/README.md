# Square purification contour pilot

One self-contained Colab notebook, two independent tasks: **30×30 and 40×40**,
one Born trajectory each, **60 cycles**. Hard/support-truncated walls,
alpha1=1, alpha2=30, nshell=1, trial orbital X, perfect correction, raster-y,
global maximally mixed initialization, complex128, **meas_slab_only=False**.
No exterior preparation, postselection or covariance projection/clipping.
Canonical walls are x=8,22 for L30 and x=10,30 for L40 (the engine uses
L//2 ± L//4, not a floating-point quarter of L).

## Run

Place this whole folder under MyDrive/final_production_new_designs/ and open
run_square_purification_contour.ipynb in an A100 40-GB-class Colab runtime.
Mount once, inspect the complete configuration cell, and run. Code is staged
to /content. REPORT_ONLY prints the inventory; MAX_NEW_TASKS=1 runs the first
unfinished size. Run again to resume. Do not run concurrent writers.
The final cell disconnects the runtime. Upload is not part of local creation.

Output: MyDrive/classA_final_production_outputs/
square_full_measurement_purification_l30-40_s1_t60_v1/.
Each Lxxx_sample000 folder contains result.npz and result.json.
Root seed 2026092929; size-derived independent seeds are invariant to task order.
Changing science requires a new revision/output directory; old data are not imported.

## Saved data

Every cycle t=0..60, including the unprepared global maxmix state at zero:

- total_entropy: sum of h(nu) over full-system occupations, natural logs;
- entropy_contour: float64 (61,L,L), cell-resolved, summing two orbitals;
- occupation_spectrum: float64 (61,2L²), sorted ascending, raw eigensolver values;
- global_charge, modular_gap, lyapunov_gap, finite_mode_count;
- Hermiticity, occupation-bound and contour-closure diagnostics.

Here C=(G+I)/2, h(nu)=-nu log(nu)-(1-nu)log(1-nu), and
s_i=sum_j |U_ij|² h(nu_j). This is **full-system purification entropy**,
not subsystem entanglement. The contour integrates to the saved total entropy.
Entropy uses the existing 1e-12 occupation endpoint regulator, producing a tiny
numerical floor for pure states. No clipping is applied to the evolving state.
Occupation violations beyond 1e-9 are independently rechecked on CPU; a
confirmed violation stops the run and retains the previous durable checkpoint.

Gap diagnostics reuse the same eigenvalues: minimum absolute modular energy
over 1e-9<nu<1-1e-9, divided by 2t for the finite-time Lyapunov half gap.
The rate is NaN at t=0; both gaps are NaN if every mode is capped.
These are one-sample finite-time diagnostics, not precision scaling estimates.
No full eigenvectors, covariance history or subsystem-width sweep is saved.

## Resume and cost

Every five cycles publish a rolling checkpoint.npz/checkpoint.json containing
the full centered covariance, NumPy/Torch CPU/all CUDA RNG states, completed
cycle and all partial observables. Restore RNG immediately before the engine
continuation, skip its repeated local zero, and preserve global cycle numbering.
Partial/mismatched pairs rerun deterministically; verified results are skipped.
Results/checkpoints use local staging, DriveFS temporary copy, checksum readback,
atomic replacement and receipt last. This is mounted-filesystem readback, not
independent Drive API verification. Never delete the final checkpoint until
the complete result passes verification. A lost runtime repeats at most five
physical cycles since the last successful checkpoint.

Final arrays are about 4 MB total uncompressed for both sizes. A covariance
checkpoint is about 52 MB for L30 or 164 MB for L40, plus temporary copies.
Allow at least 2 GiB scratch and Drive free space. Runtime will be printed per
five-cycle segment; no one-trajectory A100 benchmark has been measured here.
Small sample count does not imply linear speedup over an efficiently batched
100-sample run. Local tests exercise small CPU-backed canonical GPU-class cases;
they are not production A100 timing/acceptance runs.

Scientific entry point: classA_U1FGTN_gpu.run_markov_circuit.
Bundled engine/helper are byte-identical to canonical repository sources.
