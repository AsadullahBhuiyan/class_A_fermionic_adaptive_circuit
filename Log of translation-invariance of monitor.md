08/10/25

Log of translation-invariance of monitored circuit traj-avged DW steady-state vs cycles

Parameters
- Nshell = None
- Nx = Ny = 16
- Samples = 100
- filling frac = 1/2

Test procedure for translation invariance
- Each experiement with DW=on (breaking x-translation invariance) produces S=100 trajectories, each corresponding to an evolution by T cycles. A multi-dim array of shape (S, T, 4NxNy, 4NxNy) representing the complex covariance is produced. The data is further processed such that the trajectories are averaged over S=100 samples, and only the last snapshot T=-1 is kept, the 'top' layer is extracted and reshaped into a 6-dim tensor of shape (Nx, Ny, 2, Nx, Ny, 2)

To verify y-translation invariance, we perform two main checks:

1) Shift Invariances
    We perform a single site shift G{y,y'} -> G'{y,y'} = G{y+1,y'+1} and compute the maximum elements of |G-G'|, where |...| should be understood elementwise

2) Fourier Basis Off Diagonal Weight
    We fourier transform along the y-direction such that G{y,y'} -> G(ky,ky'). Then a square Frobenius norm is computed along all other dimensions via np.sum(np.abs(Gk)**2, axis=(0, 2, 3, 5)). This leads to a (Ny, Ny) object. A sum over all elements, one including and and one excluding the diagonal entries is taken. The ratio is then inspected.

Below are the results for difference cycles

Control Run (DW=off, S=1, T=20)
{'shift_max_abs_diff': 2.4924506902834764e-13, 'offdiag_ratio': 8.886616796731308e-27}

DW Run 1 (DW = on, S=100, T=20)
{'shift_max_abs_diff': 0.15546235836942923,'offdiag_ratio': 0.001517960655899712}

DW Run 2 (DW = on, S=100, T=40)
{'shift_max_abs_diff': 0.12997568019964356, 'offdiag_ratio': 0.0016191323171695787}

DW Run 3 (DW = on, S=100, T=80)
{'shift_max_abs_diff': 0.1203674000786985, 'offdiag_ratio': 0.0015114453560686678}

Lindbladian Run (DW=on, T=20, Decoh=on) 
{'shift_max_abs_diff': 9.992007221626409e-16, 'offdiag_ratio': 1.2849692308093654e-31}

14/10/25

We tried to determine 