import numpy as np
import os
import math
import time
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
import joblib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from contextlib import contextmanager, nullcontext
from scipy.optimize import curve_fit
from contextlib import contextmanager


class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    """Update tqdm whenever a joblib batch finishes."""
    def __init__(self, tqdm_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object
    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager linking joblib's callback to a tqdm progress bar."""
    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: _TqdmBatchCompletionCallback(tqdm_object, *args, **kwargs)
    try:
        with tqdm_object as pbar:
            yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback

class classA_U1FGTN:

    def __init__(
        self,
        Nx,
        Ny,
        DW=True,
        nshell=None,
        filling_frac=1 / 2,
        G0=None,
        alpha_triv = 3
    ):
        '''Initialize lattice dimensions, domain-wall option, and starting covariance.'''

        self.time_init = time.time()
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.Ntot = 4 * self.Nx * self.Ny
        self.filling_frac = filling_frac
        self.nshell = nshell
        self.DW = bool(DW)
        self.alpha_triv = alpha_triv
        if self.DW:
            self.create_domain_wall(alpha_triv=self.alpha_triv)

        self.G0 = None if G0 is None else np.array(G0, dtype=np.complex128, copy=True)
        self.G = None
        self.G_history_samples = None

        print("------------------------- classA_U1FGTN Initialized -------------------------")

    def _build_initial_covariance(self, G0=None):
        if G0 is not None:
            return np.asarray(G0, dtype=np.complex128)

        Ntot = self.Ntot
        Nlayer = self.Ntot//2
        Nfill = int(round(self.filling_frac * Ntot))
        Nfill = max(0, min(Ntot, Nfill))
        diag = np.concatenate(
            [
                np.ones(Nfill, dtype=np.complex128),
                -np.ones(Ntot - Nfill, dtype=np.complex128),
            ]
        )
        rng = np.random.default_rng()
        rng.shuffle(diag)
        D = np.diag(diag)
        U_top = self.random_unitary(Nlayer)
        I_bot = np.eye(Nlayer, dtype=np.complex128)
        U_tot = self._block_diag2(U_top, I_bot)
        return U_tot.conj().T @ D @ U_tot

    # ------------------------------ Utilities ------------------------------
    def _joblib_tqdm_ctx(self, total, desc):
        """
        Single outer tqdm bar for joblib.Parallel.
        Use as: with self._joblib_tqdm_ctx(samples, "samples"): Parallel(...).
        """
        return tqdm_joblib(tqdm(total=total, desc=desc, unit="task")) if total and total > 1 else nullcontext()
    
    def format_interval(self, seconds):
        """Convert seconds to H:MM:SS (or D:HH:MM:SS if >1 day)."""
        seconds = int(round(seconds))
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        if days > 0:
            return f"{days}d {hours:02}:{minutes:02}:{seconds:02}"
        else:
            return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path
    
    def _solve_regularized(self, K, B, eps=1e-9):
        """
        Solve K X = B with a small Tikhonov ridge if needed; fall back to pinv.
        """
        try:
            return np.linalg.solve(K, B)
        except np.linalg.LinAlgError:
            pass
        n = K.shape[0]
        K_reg = K + eps * np.eye(n, dtype=K.dtype)
        try:
            return np.linalg.solve(K_reg, B)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(K_reg) @ B

    def _block_diag2(self, A, B):
        """
        Minimal block_diag(A,B) without scipy. Returns [[A,0],[0,B]].
        """
        n, m = A.shape[0], B.shape[0]
        Z1 = np.zeros((n, m), dtype=A.dtype)
        Z2 = np.zeros((m, n), dtype=B.dtype)
        return np.block([[A, Z1],
                         [Z2, B]])

    # ------------------ Overcomplete Wannier (OW) projectors ------------------

    def create_domain_wall(self, alpha_triv=None):
        '''Construct and store the default domain-wall mass profile for the lattice.'''
        Nx, Ny = self.Nx, self.Ny
        if alpha_triv is None:
            alpha = np.full((Nx, Ny), 3, dtype=np.complex128)  # trivial
        else:
            alpha = np.full((Nx, Ny), float(alpha_triv), dtype=np.complex128)
        half = Nx // 2
        w = max(1, int(np.floor(0.2 * Nx)))
        x0 = max(0, half - w)
        x1 = min(Nx, half + w + 1)  # inclusive slab -> slice end-exclusive
        alpha[x0:x1, :] = 1         # topological region
        print(f"DWs at x=({int(x0-1)}, {int(x1-1)})")
        self.DW_loc = [int(x0-1), int(x1-1)]
        self.alpha = alpha

    def construct_OW_projectors(self, nshell, DW):
        '''Build overcomplete Wannier projectors used for adaptive measurements.'''
        Nx, Ny = self.Nx, self.Ny

        # Mass profile alpha(x) for the Dirac-CI model
        if not DW:
            self.alpha = np.ones((Nx, Ny), dtype=np.complex128)
        
        alpha = self.alpha

        # k-grid (FFT order)
        kx = 2*np.pi * np.fft.fftfreq(Nx)
        ky = 2*np.pi * np.fft.fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # (Nx, Ny)

        # unit vector n(k)
        nx = np.sin(KX)[:, :, None, None]
        ny = np.sin(KY)[:, :, None, None]
        nz = alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None]
        nmag = np.sqrt(nx**2 + ny**2 + nz**2)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        # Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)

        # h(k) = n̂ · σ
        hk = (nx[..., None, None] * sx +
              ny[..., None, None] * sy +
              nz[..., None, None] * sz) / nmag[..., None, None]  # (Nx,Ny,Rx,Ry,2,2)

        # band projectors in k-space
        self.Pminus = 0.5 * (I2 - hk)
        self.Pplus  = 0.5 * (I2 + hk)

        # local 2-spinors
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]],  dtype=np.complex128)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=np.complex128)

        # phases for centers R=(Rx,Ry)
        Rx_grid = np.arange(Nx)
        Ry_grid = np.arange(Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Rx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ry)
        phase   = phase_x * phase_y

        def k2_to_r2(Ak):
            # FFT over k-axes (0,1) for all centers
            return np.fft.fft2(Ak, axes=(0, 1))

        # Build normalized W for each center from tau^† P(k)
        def make_W(Pband, tau, phase):
            tau_dag = tau[:, 0].conj()
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband, optimize=True)  # (...,2)

            F0 = phase * psi_k[..., 0]
            F1 = phase * psi_k[..., 1]
            W0 = k2_to_r2(F0)  # (Nx,Ny,Rx,Ry)
            W1 = k2_to_r2(F1)  # (Nx,Ny,Rx,Ry)

            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)  # (Nx,Ny,2,Rx,Ry)

            if nshell is not None:
                x = np.arange(Nx)[:, None, None, None]
                y = np.arange(Ny)[None, :, None, None]
                Rx = np.arange(Nx)[None, None, :, None]
                Ry = np.arange(Ny)[None, None, None, :]
                dxw = ((x - Rx + Nx//2) % Nx) - Nx//2
                dyw = ((y - Ry + Ny//2) % Ny) - Ny//2
                mask = ((np.abs(dxw) <= nshell) & (np.abs(dyw) <= nshell))[:, :, None, :, :]
                W = W * mask
            # Normalize per center (Rx,Ry) over (x,y,μ)
            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
            return W / denom

        # Wannier spinors (Nx,Ny,2,Rx,Ry)
        W_Ap = make_W(self.Pplus,  tauA, phase)
        W_Bp = make_W(self.Pplus,  tauB, phase)
        W_Am = make_W(self.Pminus, tauA, phase)
        W_Bm = make_W(self.Pminus, tauB, phase)

        def flatten_centers(W):
            # (Nx,Ny,2,Rx,Ry) -> (2*Nx*Ny, Rx, Ry) with i=μ+2x+2Nx y (Fortran on μ,x,y)
            W_mu_xy = np.transpose(W, (2, 0, 1, 3, 4))                 # (2,Nx,Ny,Rx,Ry)
            return W_mu_xy.reshape(2 * Nx * Ny, Nx, Ny, order='F')

        # Spinors flattened
        self.WF_Ap = flatten_centers(W_Ap)
        self.WF_Bp = flatten_centers(W_Bp)
        self.WF_Am = flatten_centers(W_Am)
        self.WF_Bm = flatten_centers(W_Bm)

    def _proj_from_WF(self, WF, Rx, Ry):
        """Build rank-1 projector χχ† on-the-fly from a stored Wannier spinor WF[:, Rx, Ry]."""
        chi = np.asarray(WF[:, Rx, Ry], dtype=np.complex128)
        # make writable (avoid read-only view issues under loky/shared arrays)
        chi = np.array(chi, copy=True)
        return np.outer(chi, chi.conj())

    def _bulk_band_gap(self, alpha=None):
        """
        Estimate the direct gap 2*min_k |n(k)| for the 2-band CI Hamiltonian on the Brillouin grid.

        Follows the same Bloch-vector construction used in construct_OW_projectors, but reduces any
        spatially-varying mass profile to its spatial average before evaluating the k-space spectrum.
        """
        Nx, Ny = self.Nx, self.Ny
        if alpha is None:
            alpha_eff = 1.0
        else:
            alpha_eff = float(alpha)

        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        nx = np.sin(KX)
        ny = np.sin(KY)
        nz = alpha_eff - np.cos(KX) - np.cos(KY)

        nmag = np.sqrt(nx * nx + ny * ny + nz * nz)
        gap = 2.0 * float(np.min(nmag))
        if not np.isfinite(gap) or gap <= 0.0:
            raise RuntimeError("Unable to determine a positive bulk gap from the k-space spectrum.")
        return gap
    

     # ------------------------------ Exact CI state ------------------------------

    
    def G_CI(self, alpha=1.0, k_is_centered=False, norm='backward'):
        '''Build the CI lower-band covariance G = 2 P_-^* - I for the top layer.

        Flattening index: i = μ + 2*x + 2*Nx*y (μ fastest; Fortran order).
        '''
        Nx, Ny = self.Nx, self.Ny

        kx = 2*np.pi * np.fft.fftfreq(Nx, d=1.0)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        nx = np.sin(KX)
        ny = np.sin(KY)
        nz = float(alpha) - np.cos(KX) - np.cos(KY)

        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        n_mag = np.where(n_mag == 0, 1e-15, n_mag)

        def _k_to_r_rel(nk, k_centered=False, norm='backward'):
            arr = np.fft.ifftshift(nk) if k_centered else nk
            nR = np.fft.ifft2(arr, norm=norm)
            nR = np.real_if_close(nR, tol=1e3)
            x = np.arange(Nx); y = np.arange(Ny)
            dX = (x[:, None, None, None] - x[None, None, :, None]) % Nx
            dY = (y[None, :, None, None] - y[None, None, None, :]) % Ny
            return nR[dX, dY]

        nx_real = _k_to_r_rel(nx / n_mag, k_centered=k_is_centered, norm=norm)
        ny_real = _k_to_r_rel(ny / n_mag, k_centered=k_is_centered, norm=norm)
        nz_real = _k_to_r_rel(nz / n_mag, k_centered=k_is_centered, norm=norm)

        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = 1j * np.array([[0, -1], [1, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        h_real = (nx_real[..., None, None] * sx +
                  ny_real[..., None, None] * sy +
                  nz_real[..., None, None] * sz)
        h_real = np.moveaxis(h_real, 4, 2)  # (Nx,Ny,2, Nx,Ny,2)

        dims = (Nx, Ny, 2)
        I6 = np.eye(np.prod(dims), dtype=np.complex128).reshape(*dims, *dims, order='F')
        Pminus = 0.5 * (I6 - h_real)

        P6 = np.transpose(Pminus, (2, 0, 1, 5, 3, 4))  # (2,Nx,Ny, 2,Nx,Ny)
        Pminus_flat = P6.reshape(2*Nx*Ny, 2*Nx*Ny, order='F')
        return 2 * Pminus_flat.conj() - np.eye(2*Nx*Ny, dtype=np.complex128)
    
    def _domain_wall_hamiltonian(self, periodic=True, alpha=None):
        """
        Build the real-space Dirac/Chern Hamiltonian with spatially varying mass alpha(x,y).
        """
        if alpha is None:
            if not hasattr(self, "alpha"):
                self.create_domain_wall()
            alpha = self.alpha

        Nx, Ny = self.Nx, self.Ny
        N = 2 * Nx * Ny
        H = np.zeros((N, N), dtype=np.complex128)

        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        def idx(mu, x, y):
            return mu + 2 * x + 2 * Nx * y

        def add_block(x0, y0, x1, y1, block):
            i0 = [idx(mu, x0, y0) for mu in (0, 1)]
            i1 = [idx(mu, x1, y1) for mu in (0, 1)]
            H[np.ix_(i0, i1)] += block

        # onsite mass terms
        for x in range(Nx):
            for y in range(Ny):
                add_block(x, y, x, y, alpha[x, y] * sigma_z)

        # nearest-neighbour hoppings
        hop_x = -0.5 * sigma_z - 0.5j * sigma_x
        hop_y = -0.5 * sigma_z - 0.5j * sigma_y  # sign matches sin(k_y)= (e^{ik_y}-e^{-ik_y})/(2i)

        for x in range(Nx):
            for y in range(Ny):
                xp = x + 1
                if xp < Nx:
                    add_block(x, y, xp, y, hop_x)
                    add_block(xp, y, x, y, hop_x.conj().T)
                elif periodic:
                    xp = 0
                    add_block(x, y, xp, y, hop_x)
                    add_block(xp, y, x, y, hop_x.conj().T)

                yp = y + 1
                if yp < Ny:
                    add_block(x, y, x, yp, hop_y)
                    add_block(x, yp, x, y, hop_y.conj().T)
                elif periodic:
                    yp = 0
                    add_block(x, y, x, yp, hop_y)
                    add_block(x, yp, x, y, hop_y.conj().T)
        return H

    def G_CI_domain_wall(self, periodic=True, alpha=None, tol=1e-9):
        """
        Build the complex covariance by diagonalizing the real-space DW Hamiltonian.
        """
        H = self._domain_wall_hamiltonian(periodic=periodic, alpha=alpha)
        evals, evecs = np.linalg.eigh(H)
        occ = evals < -tol
        if not np.any(occ):
            half = H.shape[0] // 2
            order = np.argsort(evals)
            occ = np.zeros_like(evals, dtype=bool)
            occ[order[:half]] = True
        P_minus = evecs[:, occ] @ evecs[:, occ].conj().T
        return 2 * P_minus.conj() - np.eye(H.shape[0], dtype=np.complex128)
    
    #============================== Circuit Operations ====================================
    
    def random_unitary(self, N, rng=None):
        '''Generate a random unitary by diagonalizing a random Hermitian matrix.'''
        rng = np.random.default_rng() if rng is None else rng
        M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H = 0.5 * (M + M.conj().T)
        w, V = np.linalg.eigh(H)
        U = V @ np.diag(np.exp(1j * w)) @ V.conj().T
        return U

    def random_complex_fermion_covariance(self, N, rng=None):
        '''Build G = U^† D U with ±1 occupations at the requested filling fraction.'''
        assert N % 2 == 0, "Total dimension N must be even."
        rng = np.random.default_rng() if rng is None else rng
        filling_frac = getattr(self, "filling_frac", 0.5)
        Nfill = int(round(filling_frac * N))
        Nfill = max(0, min(N, Nfill))
        diag = np.concatenate([np.ones(Nfill), -np.ones(N - Nfill)])
        D = np.diag(diag).astype(np.complex128)

        U = self.random_unitary(N, rng=rng)
        return U.conj().T @ D @ U
    
    def build_maxmix_Gtop(self, top=False):
        """
        Build an initial G0 where the top layer is maximally mixed (zero block),
        and the bottom layer is diagonal +/-1 to meet filling_frac.
        Returns the new (4N,4N) G0; does NOT mutate self.G0.
        """
        Ntot   = self.Ntot
        Nlayer = Ntot // 2
    
        # bottom fill +/-1 by filling_frac
        Nelec = int(round(getattr(self, "filling_frac", 0.5) * Nlayer))
        occ   = np.array([1] * Nelec + [-1] * (Nlayer - Nelec), dtype=np.float64)
        np.random.shuffle(occ)
        Gbb = np.diag(occ).astype(np.complex128)
    
        # top maximally mixed => zero block
        Gtt = np.zeros((Nlayer, Nlayer), dtype=np.complex128)
    
        # block diag [[Gtt,0],[0,Gbb]]
        if not top:
            return self._block_diag2(Gtt, Gbb)
        else:
            return Gtt
       

    # --------------------------- Measurement updates ---------------------------

    def measure_bottom_layer(self, G, P, particle=True, symmetrize=True):
        '''Apply a charge-conserving measurement update on the bottom layer.'''
        Ntot = self.Ntot
        Nlayer = Ntot // 2

        G = np.asarray(G, dtype=np.complex128)
        P = np.asarray(P, dtype=np.complex128)

        Il = np.eye(Nlayer, dtype=np.complex128)

        Gtt = G[:Nlayer, :Nlayer]
        Gbb = G[Nlayer:, Nlayer:]
        Gtb = G[:Nlayer, Nlayer:]

        if particle:
            H11, H21, H22 = -P, (Il - P), P
        else:
            H11, H21, H22 =  P, (Il - P), -P


        M = self._block_diag2(Gtt, H22)
        K = self._block_diag2(Gtb, H21)

        # apply sherman-morrison formula to block inverse
        #c = 1 - np.trace(Gbb @ H11)
        #c = np.real_if_close(c, tol=1e-12)
        #c = float(np.real(c))
        #normG = np.linalg.norm(G)
        #eps = 1e-8
        #if not np.isfinite(c) or abs(c) < eps:
        #    c = np.copysign(eps, c if c != 0.0 else 1.0)
            
        #blockinv11 = H11 / c
        #blockinv12 = -Il + H11 @ Gbb / c
        #blockinv22 = - Gbb + Gbb @ H11 @ Gbb / c
        #blockinvfull = np.block([[blockinv11, blockinv12], [blockinv12.conj().T, blockinv22]])

        K = np.block([[Gbb,    -Il],
                      [-Il,     H11]])
        L = self._block_diag2(Gtb, H21)
        invK_Ldag = self._solve_regularized(K, L.conj().T, eps=1e-9)
        M = self._block_diag2(Gtt, H22)
        Gp = M - L @ invK_Ldag

        #Gp = M - K @ blockinvfull @ K.conj().T

        if symmetrize:
            Gp = 0.5 * (Gp + Gp.conj().T)
        return Gp

    def measure_top_layer(self, G, P, particle=True, symmetrize=True):
        '''Apply a charge-conserving measurement update on the top layer.'''
        Ntot = self.Ntot
        Nlayer = Ntot // 2

        G = np.asarray(G, dtype=np.complex128)
        P = np.asarray(P, dtype=np.complex128)

        Il = np.eye(Nlayer, dtype=np.complex128)

        Gtt = G[:Nlayer, :Nlayer]
        Gbb = G[Nlayer:, Nlayer:]
        Gbt = G[Nlayer:, :Nlayer]

        if particle:
            H11, H21, H22 = -P, (Il - P), P
        else:
            H11, H21, H22 =  P, (Il - P), -P

        M = self._block_diag2(H22, Gbb)
        K = self._block_diag2(H21, Gbt)

        # apply sherman-morrison formula to block inverse
        # Numerical guard: c should be real and positive, but round-off can make it tiny/complex.
        #c = 1 - np.trace(Gtt @ H11)
        #c = np.real_if_close(c, tol=1e-12)
        #c = float(np.real(c))
        #normG = np.linalg.norm(G)
        #eps = 1e-8
        #if not np.isfinite(c) or abs(c) < eps:
        #    c = np.copysign(eps, c if c != 0.0 else 1.0)
        #blockinv11 = -Gtt + Gtt @ H11 @ Gtt / c
        #blockinv12 = -Il + Gtt @ H11 / c
        #blockinv22 = H11/c
        #blockinvfull = np.block([[blockinv11, blockinv12], [blockinv12.conj().T, blockinv22]])

        K = np.block([[H11,  -Il],
                      [-Il,   Gtt]])
        L = self._block_diag2(H21, Gbt)
        invK_Ldag = self._solve_regularized(K, L.conj().T, eps=1e-9)
        M = self._block_diag2(H22, Gbb)
        Gp = M - L @ invK_Ldag

        #Gp = M - K @ blockinvfull @ K.conj().T

        if symmetrize:
            Gp = 0.5 * (Gp + Gp.conj().T)
        return Gp

    def fSWAP(self, chi_top, chi_bottom):
        Ntot = self.Ntot
        Nlayer = Ntot // 2

        ct = np.array(chi_top,   dtype=np.complex128, copy=True).reshape(-1)
        cb = np.array(chi_bottom, dtype=np.complex128, copy=True).reshape(-1)

        nt = np.linalg.norm(ct) + 1e-15
        nb = np.linalg.norm(cb) + 1e-15

        ct = ct/nt
        cb = cb/nb

        psi_t = np.zeros(Ntot, dtype=np.complex128); psi_t[:Nlayer]  = ct
        psi_b = np.zeros(Ntot, dtype=np.complex128); psi_b[Nlayer:]  = cb

        Pt = np.outer(psi_t, psi_t.conj())
        Pb = np.outer(psi_b, psi_b.conj())
        Xtb = np.outer(psi_t, psi_b.conj())
        Xbt = np.outer(psi_b, psi_t.conj())
        U = (np.eye(Ntot, dtype=np.complex128) - Pt - Pb) + (Xtb + Xbt)
        return U
    
        # ---------------------- Local feedback / post-selection ----------------------

    def markov_meas_feedback(self, G, Rx, Ry, n_a=0.5):
        '''Perform Markovian measurement plus feedback at site (Rx, Ry) on top-layer covariance.'''
        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        if G.shape != (Nlayer, Nlayer):
            raise ValueError(f"markov_meas_feedback expects top-layer covariance of shape ({Nlayer},{Nlayer}); got {G.shape}")

        Il = np.eye(Nlayer, dtype=np.complex128)

        def _measure(G, P, particle):
            if particle:
                H11, H21, H22 = -P, (Il - P), P
            else:
                H11, H21, H22 = P, (Il - P), -P

            A = G @ H11 - Il
            B = G @ H21.conj().T
            eps_scale = np.linalg.norm(A, ord=np.inf)
            if not np.isfinite(eps_scale) or eps_scale < 1.0:
                eps_scale = 1.0
            X = self._solve_regularized(A, B, eps=1e-9 * eps_scale)

            G_upd = H22 - H21 @ X

            #if not np.all(np.isfinite(G)):
            #    raise FloatingPointError("Non-finite entries encountered in top-layer covariance before measurement.")

            # Numerical guard: c should be real and positive, but round-off can make it tiny/complex.
            #c = 1 - np.trace(G @ H11)
            #c = np.real_if_close(c, tol=1e-12)
            #c = float(np.real(c))
            #normG = np.linalg.norm(G)
            #eps = 1e-8
            #if not np.isfinite(c) or abs(c) < eps:
            #    c = np.copysign(eps, c if c != 0.0 else 1.0)
#
            #inv_c = 1.0 / c
            #GH = G @ H11 @ G
            #numer = GH - c * G
            #G_upd = H22 - H21 @ (numer * inv_c) @ H21.conj().T
#
            #if not np.all(np.isfinite(G_upd)):
            #    raise FloatingPointError(
            #        f"Non-finite top-layer update (particle={particle}, c={c:.3e}, ||G||={normG:.3e})"
            #    )

            G_upd = 0.5 * (G_upd + G_upd.conj().T)
            return G_upd

        def _ancilla_swap_top(Gtop, P, n_a):
            Q = Il - P
            if np.random.rand() < n_a:
                ancilla_cov = 1
            else:
                ancilla_cov = -1
            Gnew = Q @ Gtop @ Q + ancilla_cov * P
            Gnew = 0.5 * (Gnew + Gnew.conj().T)
            if not np.all(np.isfinite(Gnew)):
                raise FloatingPointError(f"Non-finite ancilla swap result (n_a={n_a}, occ={ancilla_cov})")
            return Gnew

        P_Ap = self._proj_from_WF(self.WF_Ap, Rx, Ry)
        P_Bp = self._proj_from_WF(self.WF_Bp, Rx, Ry)
        P_Am = self._proj_from_WF(self.WF_Am, Rx, Ry)
        P_Bm = self._proj_from_WF(self.WF_Bm, Rx, Ry)

        def _clamp_prob(val):
            return np.clip(float(np.real(val)), 0.0, 1.0)

        Gtt_2pt = 0.5 * (G + Il)

        # Upper band A: expect UNOCCUPIED
        p_occ = _clamp_prob(np.trace(Gtt_2pt @ P_Ap))
        occ_event = np.random.rand() < p_occ
        G = _measure(G, P_Ap, particle=occ_event)
        if occ_event: 
            G = _ancilla_swap_top(G, P_Ap, n_a) # pump out
        Gtt_2pt = 0.5 * (G + Il)

        # Upper band B: expect UNOCCUPIED
        p_occ = _clamp_prob(np.trace(Gtt_2pt @ P_Bp))
        occ_event = np.random.rand() < p_occ
        G = _measure(G, P_Bp, particle=occ_event)
        if occ_event:
            G = _ancilla_swap_top(G, P_Bp, n_a) # pump out
        Gtt_2pt = 0.5 * (G + Il)

        # Lower band A: expect OCCUPIED
        p_occ = _clamp_prob(np.trace(Gtt_2pt @ P_Am))
        occ_event = np.random.rand() < p_occ
        G = _measure(G, P_Am, particle=occ_event)
        if not occ_event:
            G = _ancilla_swap_top(G, P_Am, n_a) # pump in
        Gtt_2pt = 0.5 * (G + Il)

        # Lower band B: expect OCCUPIED
        p_occ = _clamp_prob(np.trace(Gtt_2pt @ P_Bm))
        occ_event = np.random.rand() < p_occ
        G = _measure(G, P_Bm, particle=occ_event)
        if not occ_event:
            G = _ancilla_swap_top(G, P_Bm, n_a) # pump in

        return G
        

    def top_layer_meas_feedback(self, G, Rx, Ry):
        '''Perform adaptive measurement plus feedback at site (Rx, Ry).'''
        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        Il = np.eye(Nlayer, dtype=np.complex128)
        Gtt_2pt = 0.5 * (G[:Nlayer, :Nlayer] + Il)

        # On-the-fly projectors at (Rx, Ry)
        P_Ap = self._proj_from_WF(self.WF_Ap, Rx, Ry)
        P_Bp = self._proj_from_WF(self.WF_Bp, Rx, Ry)
        P_Am = self._proj_from_WF(self.WF_Am, Rx, Ry)
        P_Bm = self._proj_from_WF(self.WF_Bm, Rx, Ry)

        # Chi spinors (for fSWAP) — make writable copies
        chi_Ap = np.array(self.WF_Ap[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Bp = np.array(self.WF_Bp[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Am = np.array(self.WF_Am[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Bm = np.array(self.WF_Bm[:, Rx, Ry], dtype=np.complex128, copy=True)

        Nx = self.Nx
        eA_b = np.eye(Nlayer, dtype=np.complex128)[0 + 2*Rx + 2*Nx*Ry]
        eB_b = np.eye(Nlayer, dtype=np.complex128)[1 + 2*Rx + 2*Nx*Ry]

        def do_fswap(Gmat, chi_top, chi_bot):
            U = self.fSWAP(chi_top, chi_bot)
            return U.conj().T @ Gmat @ U

        # Upper band A: want UNOCCUPIED
        p_occ = float(np.real(np.trace(Gtt_2pt @ P_Ap))); p_occ = np.clip(p_occ, 0.0, 1.0)
        if np.random.rand() < p_occ:
            G = self.measure_top_layer(G, P_Ap, particle=True)
            G = do_fswap(G, chi_Ap, eA_b)  # swap OUT
        else:
            G = self.measure_top_layer(G, P_Ap, particle=False)

        # Upper band B: want UNOCCUPIED
        p_occ = float(np.real(np.trace(Gtt_2pt @ P_Bp))); p_occ = np.clip(p_occ, 0.0, 1.0)
        if np.random.rand() < p_occ:
            G = self.measure_top_layer(G, P_Bp, particle=True)
            G = do_fswap(G, chi_Bp, eB_b)
        else:
            G = self.measure_top_layer(G, P_Bp, particle=False)

        # Lower band A: want OCCUPIED
        p_occ = float(np.real(np.trace(Gtt_2pt @ P_Am))); p_occ = np.clip(p_occ, 0.0, 1.0)
        if np.random.rand() < p_occ:
            G = self.measure_top_layer(G, P_Am, particle=True)
        else:
            G = self.measure_top_layer(G, P_Am, particle=False)
            G = do_fswap(G, chi_Am, eA_b)  # swap IN

        # Lower band B: want OCCUPIED
        p_occ = float(np.real(np.trace(Gtt_2pt @ P_Bm))); p_occ = np.clip(p_occ, 0.0, 1.0)
        if np.random.rand() < p_occ:
            G = self.measure_top_layer(G, P_Bm, particle=True)
        else:
            G = self.measure_top_layer(G, P_Bm, particle=False)
            G = do_fswap(G, chi_Bm, eB_b)

        return G

    def post_selection_top_layer(self, G, Rx, Ry):
        '''Project the top layer at (Rx, Ry) onto the desired four outcomes.'''
        # On-the-fly projectors
        P_Ap = self._proj_from_WF(self.WF_Ap, Rx, Ry)
        P_Bp = self._proj_from_WF(self.WF_Bp, Rx, Ry)
        P_Am = self._proj_from_WF(self.WF_Am, Rx, Ry)
        P_Bm = self._proj_from_WF(self.WF_Bm, Rx, Ry)

        G = self.measure_top_layer(G, P_Ap, particle=False)  # Ap unocc
        G = self.measure_top_layer(G, P_Bp, particle=False)  # Bp unocc
        G = self.measure_top_layer(G, P_Am, particle=True)   # Am occ
        G = self.measure_top_layer(G, P_Bm, particle=True)   # Bm occ
        return G

    def measure_all_bottom_modes(self, G):
        '''Measure every bottom-layer mode once, sampling the proper Bernoulli distribution.'''
        #start = time.time()

        G = np.asarray(G, dtype=np.complex128)
        Ntot = self.Ntot
        Nlayer = Ntot // 2
        Il = np.eye(Nlayer, dtype=np.complex128)

        # iterate all bottom single-site projectors in canonical basis
        for idx in range(Nlayer):
            Gbb_2pt = 0.5 * (G[Nlayer:, Nlayer:] + Il)  # refresh per step
            chi = Il[idx]
            P = np.outer(chi, chi.conj())
            p_occ = float(np.real(np.trace(Gbb_2pt @ P)))
            p_occ = np.clip(p_occ, 0.0, 1.0)
            G = self.measure_bottom_layer(
                G, P, particle=(np.random.rand() < p_occ), symmetrize=True
            )

        #self.bottom_layer_mode_meas_time = time.time() - start
        #if not getattr(self, "_suppress_bottom_measure_prints", False):
            #print()
            #print(f"\nAll bottom layer modes measured | Time elapsed: {self.bottom_layer_mode_meas_time:.3f} s", flush=True)

        return G

    def randomize_bottom_layer(self, G):
        '''Apply an independently random unitary to the bottom layer.'''
        G = np.asarray(G, dtype=np.complex128)
        Ntot = self.Ntot
        Nlayer = Ntot // 2
        Il = np.eye(Nlayer, dtype=np.complex128)

        U_bott = self.random_unitary(Nlayer)
        U_tot = self._block_diag2(Il, U_bott)
        return U_tot.conj().T @ G @ U_tot


    # ==================== run_adaptive_circuit with samples loop ====================

    def run_adaptive_circuit(
        self,
        G_history=True,
        tol=1e-8,
        progress=True,
        cycles=None,
        postselect=False,
        samples=None,
        n_jobs=None,
        backend="loky",
        parallelize_samples=False,
        store="none",           # "none", "top", "full"
        init_mode="default",    # "default" or "maxmix"
        G_init=None,
        remember_init=True,
        save=True,
        save_suffix=None,
    ):
        """Execute the adaptive circuit with optional history collection."""
        have_ow = all(hasattr(self, a) for a in ("WF_Ap","WF_Bp","WF_Am","WF_Bm"))
        if not have_ow:
            self.construct_OW_projectors(nshell=self.nshell, DW=self.DW)

        cycles = 5 if cycles is None else int(cycles)

        def _cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_mode, store_mode):
            nsh = "None" if nshell is None else str(nshell)
            return (f"N{int(Nx)}x{int(Ny)}"
                    f"_C{int(cycles)}"
                    f"_S{int(samples)}"
                    f"_nsh{nsh}"
                    f"_DW{int(bool(DW))}"
                    f"_init-{init_mode}"
                    f"_store-{store_mode}")

        def _save_histories(array, samples_count):
            if not (save and store != "none" and array is not None):
                return None
            outdir = self._ensure_outdir("cache/G_history_samples")
            key = _cache_key(Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                            nshell=self.nshell, DW=self.DW, init_mode=init_mode, store_mode=store)
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            np.savez_compressed(path, G_hist=np.asarray(array, dtype=np.complex128))
            return path

        def _expected_save_path(samples_count):
            if not (save and store != "none"):
                return None
            outdir = os.path.join("cache", "G_history_samples")
            key = _cache_key(
                Nx=self.Nx,
                Ny=self.Ny,
                cycles=cycles,
                samples=samples_count,
                nshell=self.nshell,
                DW=self.DW,
                init_mode=init_mode,
                store_mode=store,
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            return os.path.join(outdir, filename)

        def _emit_save_notice(samples_count):
            path = _expected_save_path(samples_count)
            if path is None:
                return
            print(f"[info] Adaptive circuit will save history to {path}")
            time.sleep(2)

        # --------- choose/validate initial G0 ----------
        validated_G_init = None
        if G_init is not None:
            arr = np.asarray(G_init, dtype=np.complex128)
            if arr.shape != (self.Ntot, self.Ntot):
                raise ValueError(f"G_init must have shape ({self.Ntot},{self.Ntot}); got {arr.shape}")
            validated_G_init = np.array(arr, copy=True)
            self.G0 = np.array(validated_G_init, copy=True)
        else:
            if init_mode == "default":
                if self.G0 is None:
                    self.G0 = self._build_initial_covariance(None)
            elif init_mode == "maxmix":
                pass
            else:
                raise ValueError("init_mode must be 'default' or 'maxmix'.")

        # ---------------- single trajectory ----------------
        if not parallelize_samples or (samples is None or int(samples) <= 1):
            # initial state
            if validated_G_init is not None:
                self.G = np.array(validated_G_init, copy=True)
                self.G0 = np.array(self.G, copy=True)
            elif init_mode == "default":
                self.G = np.array(self.G0, copy=True)
            elif init_mode == "maxmix":
                maxmix_G = np.array(self.build_maxmix_Gtop(), copy=True)
                self.G = maxmix_G
                self.G0 = maxmix_G.copy()
            else:
                raise ValueError("init_mode must be 'default' or 'maxmix'.")

            if G_history:
                self.G_list = []
                if remember_init:
                    self.G_list.append(self.G.copy())
            self.g2_flags = []

            Nx, Ny = int(self.Nx), int(self.Ny)
            D = int(self.Ntot)
            I = np.eye(D, dtype=np.complex128)
            total_sites = cycles * Nx * Ny
            pbar = tqdm(total=total_sites, desc="RAC (sites)", unit="site", leave=True) if progress else None

            _emit_save_notice(1)

            for _c in range(cycles):
                for Rx in range(Nx):
                    for Ry in range(Ny):
                        self.G = (self.top_layer_meas_feedback(self.G, Rx, Ry)
                                if not postselect else
                                self.post_selection_top_layer(self.G, Rx, Ry))
                        self.g2_flags.append(int(np.allclose(self.G @ self.G, I, atol=tol)))
                        if pbar is not None:
                            pbar.update(1)
                if not postselect:
                    self.G = self.randomize_bottom_layer(self.G)
                    self.G = self.measure_all_bottom_modes(self.G)
                if G_history:
                    self.G_list.append(self.G.copy())

            if pbar is not None:
                pbar.close()

            # return/serialize like before
            if store == "none":
                return None
            Ntot = self.Ntot
            Nlayer = Ntot // 2
            per_cycle = self.G_list if (G_history and len(self.G_list) > 0) else [self.G]
            if store == "top":
                hist = [np.asarray(Gk)[:Nlayer, :Nlayer] for Gk in per_cycle]
            elif store == "full":
                hist = [np.asarray(Gk) for Gk in per_cycle]
            else:
                raise ValueError("store must be 'none', 'top', or 'full'")
            G_hist = np.expand_dims(np.stack(hist, axis=0), axis=0)  # (1,T,dim,dim)
            if store == "full":
                self.G_history_samples = G_hist
            saved_path = _save_histories(G_hist, samples_count=1)
            return {
                "G_hist": G_hist,
                "G_hist_avg": np.mean(G_hist, axis=0),
                "samples": 1,
                "T": G_hist.shape[1],
                "save_path": saved_path,
            }

        # ---------------- parallel multi-sample ----------------
        samples = 1 if samples is None else int(samples)
        if samples <= 0:
            raise ValueError("samples must be a positive integer")
        S = samples

        n_jobs_eff = n_jobs

        if backend not in ("loky", "threading"):
            raise ValueError("backend must be 'loky' or 'threading'.")

        Ntot = self.Ntot
        Nlayer = Ntot // 2
        _emit_save_notice(S)
        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()

        def _make_G0():
            if validated_G_init is not None:
                return np.array(validated_G_init, copy=True)
            if init_mode == "default":
                if self.G0 is None:
                    self.G0 = self._build_initial_covariance(None)
                return np.array(self.G0, copy=True)
            if init_mode == "maxmix":
                return np.array(self.build_maxmix_Gtop(), copy=True)
            raise ValueError("init_mode must be 'default' or 'maxmix'.")

        def _worker(seed_u32):
            # keep BLAS single-threaded inside each worker
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                child.G0 = _make_G0()
                child.run_adaptive_circuit(
                    G_history=True, tol=tol, progress=False, cycles=cycles, postselect=postselect,
                    parallelize_samples=False, store="none", init_mode="default",
                    remember_init=remember_init
                )
                full_hist = [np.asarray(Gk) for Gk in child.G_list]
                if store == "full":
                    return np.stack(full_hist, axis=0)  # (T,Ntot,Ntot)
                elif store == "top":
                    top_hist = [Gk[:Nlayer, :Nlayer] for Gk in full_hist]
                    return np.stack(top_hist, axis=0)   # (T,Nlayer,Nlayer)
                else:
                    raise ValueError("When parallelizing samples, set store='top' or 'full'.")

        with self._joblib_tqdm_ctx(S, "samples"):
            if backend == "loky":
                # Process backend; also tell joblib to use 1 thread per worker
                with parallel_backend("loky", n_jobs=n_jobs_eff, inner_max_num_threads=1):
                    with threadpool_limits(limits=1):
                        G_hist_list = Parallel(n_jobs=n_jobs_eff)(
                            delayed(_worker)(seeds[i]) for i in range(S)
                        )
            else:
                # Threading backend; ensure BLAS stays at 1 thread
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
                with threadpool_limits(limits=1):
                    G_hist_list = Parallel(n_jobs=n_jobs_eff, backend="threading")(
                        delayed(_worker)(seeds[i]) for i in range(S)
                    )

        G_hist = np.stack(G_hist_list, axis=0)       # (S,T,dim,dim)
        G_hist_avg = np.mean(G_hist, axis=0)
        self.G_history_samples = G_hist if store == "full" else None
        saved_path = _save_histories(G_hist, samples_count=G_hist.shape[0])
        return {
            "G_hist": G_hist,
            "G_hist_avg": G_hist_avg,
            "samples": S,
            "T": G_hist.shape[1],
            "save_path": saved_path,
        }
    def run_markov_circuit(
        self,
        G_history=True,
        progress=True,
        cycles=None,
        samples=None,
        n_jobs=None,
        backend="loky",
        parallelize_samples=False,
        init_mode="default",
        G_init=None,
        remember_init=True,
        save_final_G_only=False,
        save=True,
        save_suffix=None,
        n_a=0.5,
    ):
        """Execute the Markovian adaptive circuit with ancilla feedback."""
        have_ow = all(hasattr(self, attr) for attr in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm"))
        if not have_ow:
            self.construct_OW_projectors(nshell=self.nshell, DW=self.DW)

        cycles = 5 if cycles is None else int(cycles)

        def _cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_mode, n_a):
            nsh = "None" if nshell is None else str(nshell)
            return (
                f"N{int(Nx)}x{int(Ny)}"
                f"_C{int(cycles)}"
                f"_S{int(samples)}"
                f"_nsh{nsh}"
                f"_DW{int(bool(DW))}"
                f"_init-{init_mode}"
                f"_n_a{n_a}"
                "_markov_circuit"
            )

        def _save_histories(array, samples_count):
            if not (save and G_history and array is not None):
                return None
            outdir = self._ensure_outdir("cache/G_history_samples")
            key = _cache_key(
                Nx=self.Nx,
                Ny=self.Ny,
                cycles=cycles,
                samples=samples_count,
                nshell=self.nshell,
                DW=self.DW,
                init_mode=init_mode,
                n_a=n_a,
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            np.savez_compressed(path, G_hist=np.asarray(array, dtype=np.complex128))
            return path

        def _expected_save_path(samples_count):
            if not (save and G_history):
                return None
            outdir = os.path.join("cache", "G_history_samples")
            key = _cache_key(
                Nx=self.Nx,
                Ny=self.Ny,
                cycles=cycles,
                samples=samples_count,
                nshell=self.nshell,
                DW=self.DW,
                init_mode=init_mode,
                n_a=n_a,
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            return os.path.join(outdir, filename)

        def _emit_save_notice(samples_count):
            path = _expected_save_path(samples_count)
            if path is None:
                return
            print(f"[info] Markov circuit will save history to {path}")
            time.sleep(2)

        Nlayer = self.Ntot // 2
        filling = getattr(self, "filling_frac", 0.5)

        def _prepare_initial_top(instance):
            if G_init is not None:
                arr = np.asarray(G_init, dtype=np.complex128)
                if arr.shape == (instance.Ntot, instance.Ntot):
                    arr = arr[:Nlayer, :Nlayer]
                elif arr.shape != (Nlayer, Nlayer):
                    raise ValueError(
                        f"G_init must have shape ({instance.Ntot},{instance.Ntot}) or ({Nlayer},{Nlayer}); got {arr.shape}"
                    )
                return 0.5 * (arr + arr.conj().T)
            if init_mode == "default":
                return instance.random_complex_fermion_covariance(Nlayer, filling_frac=filling)
            if init_mode == "maxmix":
                return np.zeros((Nlayer, Nlayer), dtype=np.complex128)
            raise ValueError("init_mode must be 'default' or 'maxmix'.")

        store_full_history = G_history and not save_final_G_only

        def _run_single(instance, enable_progress, return_eta=False, sample_idx=None, total_samples=None):
            G_top = np.array(_prepare_initial_top(instance), copy=True)
            history = [] if store_full_history else None
            if store_full_history and remember_init:
                history.append(G_top.copy())

            Nx, Ny = int(instance.Nx), int(instance.Ny)
            total_sites = cycles * Nx * Ny
            desc = "Markov RAC (sites)"
            if sample_idx is not None:
                suffix = f"{sample_idx}/{total_samples}" if total_samples and total_samples > 1 else f"{sample_idx}"
                desc = f"Sample {suffix} | {desc}"
            pbar = tqdm(total=total_sites, desc=desc, unit="site", leave=True) if (enable_progress and progress) else None
            last_eta = None

            def _capture_eta():
                nonlocal last_eta
                if pbar is None:
                    return
                remaining = pbar.format_dict.get("remaining", None)
                if remaining is None:
                    return
                try:
                    remaining = float(remaining)
                except (TypeError, ValueError):
                    return
                if remaining > 0 and np.isfinite(remaining):
                    last_eta = remaining

            for _ in range(cycles):
                for Rx in range(Nx):
                    for Ry in range(Ny):
                        G_top = instance.markov_meas_feedback(G_top, Rx, Ry, n_a=n_a)
                        if pbar is not None:
                            pbar.update(1)
                            _capture_eta()
                if store_full_history:
                    history.append(G_top.copy())

            if pbar is not None:
                _capture_eta()
                pbar.close()

            result = None
            if G_history:
                if save_final_G_only:
                    result = np.expand_dims(G_top.copy(), axis=0)
                else:
                    result = np.stack(history, axis=0) if history else np.empty((0, Nlayer, Nlayer), dtype=np.complex128)
            else:
                result = G_top

            if return_eta:
                return result, (last_eta if last_eta is not None else None)
            return result

        samples = 1 if samples is None else int(samples)
        if samples <= 0:
            raise ValueError("samples must be a positive integer")

        # sequential path
        if not parallelize_samples:
            histories = [] if G_history else None
            total_samples = samples
            _emit_save_notice(total_samples)
            for idx in range(total_samples):
                if progress:
                    single_result, eta = _run_single(
                        self, enable_progress=True, return_eta=True, sample_idx=idx + 1, total_samples=total_samples
                    )
                else:
                    single_result = _run_single(
                        self, enable_progress=False, return_eta=False, sample_idx=idx + 1, total_samples=total_samples
                    )
                    eta = None

                if histories is not None:
                    histories.append(single_result)

                if progress and total_samples > 1:
                    completed = idx + 1
                    remaining_samples = total_samples - completed
                    if eta is not None and np.isfinite(eta) and remaining_samples > 0:
                        print(
                            f"Samples {completed}/{total_samples} completed. "
                            f"Est remaining time: {self.format_interval(eta * remaining_samples)}",
                            flush=True,
                        )
                    else:
                        print(f"Samples {completed}/{total_samples} completed.", flush=True)

            if not G_history:
                return None

            G_hist = np.stack(histories, axis=0)
            saved_path = _save_histories(G_hist, samples_count=G_hist.shape[0])
            return {
                "G_hist": G_hist,
                "G_hist_avg": np.mean(G_hist, axis=0),
                "samples": total_samples,
                "T": G_hist.shape[1],
                "save_path": saved_path,
            }

        # -------- parallel path over samples --------
        S = samples
        n_jobs_eff = n_jobs

        _emit_save_notice(S)

        if backend not in ("loky", "threading"):
            raise ValueError("backend must be 'loky' or 'threading'.")

        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()

        def _worker(seed_u32):
            # Make BLAS single-threaded in each worker as well
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                return _run_single(child, enable_progress=False)

        with self._joblib_tqdm_ctx(S, "samples"):
            if backend == "loky":
                # Processes + one BLAS thread per process
                with parallel_backend("loky", n_jobs=n_jobs_eff, inner_max_num_threads=1):
                    with threadpool_limits(limits=1):
                        G_hist_list = Parallel(n_jobs=n_jobs_eff)(
                            delayed(_worker)(seeds[i]) for i in range(S)
                        )
            else:  # threading
                # Keep BLAS single-threaded for threads too
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", "1")
                os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
                with threadpool_limits(limits=1):
                    G_hist_list = Parallel(n_jobs=n_jobs_eff, backend="threading")(
                        delayed(_worker)(seeds[i]) for i in range(S)
                    )

        if not G_history:
            return None

        G_hist = np.stack(G_hist_list, axis=0)
        G_hist_avg = np.mean(G_hist, axis=0)
        saved_path = _save_histories(G_hist, samples_count=G_hist.shape[0])
        return {
            "G_hist": G_hist,
            "G_hist_avg": G_hist_avg,
            "samples": S,
            "T": G_hist.shape[1],
            "save_path": saved_path,
        }

    
    # ---------------------------- Chern observables ----------------------------

    def real_space_chern_number(self, G):
        '''Compute the disk-partition real-space Chern number from a covariance.'''
        Nx, Ny = self.Nx, self.Ny
        Nlayer = 2 * Nx * Ny

        # Tri-partition masks inside a disk
        R = 0.4 * min(Nx, Ny)
        xref, yref = Nx // 2, Ny // 2
        inside = np.zeros((Nx, Ny), dtype=bool)
        A_mask = np.zeros_like(inside)
        B_mask = np.zeros_like(inside)
        C_mask = np.zeros_like(inside)
        rr = R * R
        ymax = int(math.floor(R))
        a2 = 2*np.pi/3
        a4 = 4*np.pi/3

        for dy in range(-ymax, ymax + 1):
            y = yref + dy
            if y < 0 or y >= Ny:
                continue
            max_dx = int(math.floor(math.sqrt(rr - dy*dy)))
            x0 = max(0, xref - max_dx)
            x1 = min(Nx - 1, xref + max_dx)
            if x0 > x1:
                continue
            inside[x0:x1+1, y] = True
            dxs = np.arange(x0, x1+1) - xref
            dys = np.full_like(dxs, dy)
            theta = np.mod(np.arctan2(dys, dxs), 2*np.pi)
            A_mask[x0:x1+1, y] = (theta >= 0)  & (theta < a2)
            B_mask[x0:x1+1, y] = (theta >= a2) & (theta < a4)
            C_mask[x0:x1+1, y] = (theta >= a4) & (theta < 2*np.pi)

        G = np.asarray(G, dtype=np.complex128)
        if G.shape != (self.Ntot, self.Ntot):
            raise ValueError(f"Expected full covariance of shape ({self.Ntot},{self.Ntot}); got {G.shape}")
        Gtt = G[:Nlayer, :Nlayer]
        P = (0.5 * (np.eye(Nlayer, dtype=np.complex128) + Gtt)).conj()

        def idx_from_mask(mask):
            xs, ys = np.nonzero(mask)
            idx0 = 0 + 2*xs + 2*Nx*ys
            idx1 = 1 + 2*xs + 2*Nx*ys
            return np.sort(np.concatenate([idx0, idx1]))

        iA = idx_from_mask(A_mask)
        iB = idx_from_mask(B_mask)
        iC = idx_from_mask(C_mask)

        P_CA = P[np.ix_(iC, iA)]; P_AB = P[np.ix_(iA, iB)]; P_BC = P[np.ix_(iB, iC)]
        P_AC = P[np.ix_(iA, iC)]; P_CB = P[np.ix_(iC, iB)]; P_BA = P[np.ix_(iB, iA)]

        t1 = np.trace(P_CA @ P_AB @ P_BC)
        t2 = np.trace(P_AC @ P_CB @ P_BA)
        Y = 12 * np.pi * 1j * (t1 - t2)
        return np.real_if_close(Y, tol=1e-6)

    def local_chern_marker_flat(self, G, mask_outside=False, inside_mask=None):
        '''Evaluate the flattened local Chern marker from a top-layer covariance.'''
        Nx, Ny = self.Nx, self.Ny
        Nlayer = 2 * Nx * Ny

        G = np.asarray(G, dtype=np.complex128)
        if G.shape == (self.Ntot, self.Ntot):
            Gflat = G[:Nlayer, :Nlayer]
        elif G.shape == (Nlayer, Nlayer):
            Gflat = G
        else:
            raise ValueError(f"Expected top-layer shape ({Nlayer},{Nlayer}) or full ({self.Ntot},{self.Ntot}); got {G.shape}")

        G2 = 0.5 * (Gflat + np.eye(Nlayer, dtype=np.complex128))
        G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
        G6 = np.transpose(G6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,2, Nx,Ny,2)

        P = G6.conj()

        X = np.arange(1, Nx + 1, dtype=float)
        Y = np.arange(1, Ny + 1, dtype=float)
        Xr = X[None, None, None, :, None, None]
        Yr = Y[None, None, None, None, :, None]

        def right_X(A): return A * Xr
        def right_Y(A): return A * Yr
        mm = lambda A, B: np.einsum('ijslmn,lmnopr->ijsopr', A, B, optimize=True)

        T = mm(right_Y(mm(right_X(P), P)), P)  # P X P Y P
        U = mm(right_X(mm(right_Y(P), P)), P)  # P Y P X P
        M = (2.0 * np.pi * 1j) * (T - U)

        ix = np.arange(Nx)[:, None, None]
        iy = np.arange(Ny)[None, :, None]
        ispin = np.arange(2)[None, None, :]
        diag_vals = M[ix, iy, ispin, ix, iy, ispin]  # (Nx,Ny,2)
        C = np.sum(diag_vals, axis=2)

        # numerical round-off can leave a tiny imaginary part; drop it before plotting
        C = np.real_if_close(C, tol=1e-6)
        C = np.tanh(np.real(C)).astype(np.float64, copy=False)
        if mask_outside and inside_mask is not None:
            C = np.where(inside_mask, C, 0.0)
        return C

    # ------------------------- Plotting Methods -------------------------

    def plot_real_space_chern_history(self, G_histories, filename=None, traj_avg=False):
        r'''
        Plot the real-space Chern number across time for supplied histories.

        Parameters
        ----------
        G_histories : ndarray
            Array of shape (S, T, Ntot, Ntot) containing full-layer covariances.
        save_suffix : str, optional
            When provided, this string is appended (before the extension) to every file
            path that this routine writes or returns.

        If traj_avg is False (default):
            - uses a single trajectory (first sample).
        If traj_avg is True:
            - LEFT  subplot: traj-resolved average  \overline{C_G}(t) = (1/S) sum_s C(G^{(s)}(t))
            - RIGHT subplot: traj-averaged curve   C_{Ḡ}(t) = C( (1/S) sum_s G^{(s)}(t) )

        Figures are saved if filename is provided.
        '''
        histories = np.asarray(G_histories, dtype=np.complex128)
        if histories.ndim != 4:
            raise ValueError("G_histories must have shape (S, T, Ntot, Ntot)")
        S, T, dim1, dim2 = histories.shape
        if dim1 != self.Ntot or dim2 != self.Ntot:
            raise ValueError(f"Expected final dimensions ({self.Ntot},{self.Ntot}); got ({dim1},{dim2})")
        if S == 0 or T == 0:
            raise RuntimeError("No histories available.")
        cycles = T

        # Compute Chern as requested
        x = np.arange(1, T + 1)
        if not traj_avg:
            cherns = np.empty(T, dtype=float)
            for t in range(T):
                cherns[t] = float(np.real(self.real_space_chern_number(histories[0, t])))
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            ax.plot(x, cherns, marker="o", lw=1.25)
            ax.set_xlabel("Cycles")
            ax.set_ylabel("Real-space Chern Number")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            if filename is not None:
                outdir = self._ensure_outdir(os.path.dirname(filename) or "figs/chern_history")
                fig.savefig(os.path.join(outdir, os.path.basename(filename)), bbox_inches="tight")
            return fig, ax, cherns

        # traj_avg=True -> two subplots
        C_traj_res = np.zeros(T, dtype=float)
        C_traj_avg = np.zeros(T, dtype=float)
        for t in range(T):
            Cs = [float(np.real(self.real_space_chern_number(histories[s, t]))) for s in range(S)]
            C_traj_res[t] = float(np.mean(Cs))

            Gbar_t = np.mean(histories[:, t], axis=0)
            C_traj_avg[t] = float(np.real(self.real_space_chern_number(Gbar_t)))

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.0), constrained_layout=True)
        axL.plot(x, C_traj_res, marker="o", lw=1.25)
        axL.set_title(r"$\overline{C_{G}}(t)$ (traj-resolved)")
        axL.set_xlabel("Cycles"); axL.set_ylabel("Chern"); axL.grid(True, alpha=0.3)

        axR.plot(x, C_traj_avg, marker="o", lw=1.25)
        axR.set_title(r"$C_{\overline{G}}(t)$ (traj-averaged)")
        axR.set_xlabel("Cycles"); axR.set_ylabel("Chern"); axR.grid(True, alpha=0.3)

        fig.suptitle(f"Real-space Chern history (cycles={T}, samples={S})")
        if filename is not None:
            outdir = self._ensure_outdir(os.path.dirname(filename) or "figs/chern_history")
            fig.savefig(os.path.join(outdir, os.path.basename(filename)), bbox_inches="tight")
        return fig, (axL, axR), (C_traj_res, C_traj_avg)

    def chern_marker_dynamics(
        self,
        G_histories,
        outbasename=None,
        vmin=-1.0,
        vmax=1.0,
        cmap='RdBu_r',
        traj_avg=False,
    ):
        r'''
        Animate the local Chern marker over explicit multi-sample histories.

        Parameters
        ----------
        G_histories : ndarray
            Array of shape (S, T, Ntot, Ntot) containing full-layer covariances.

        traj_avg = False:
            - single panel using first trajectory.
        traj_avg = True:
            - two-panel animation:
                LEFT  = average over samples of marker maps: \overline{tanh C(G)} (avg of f(G))
                RIGHT = marker of per-cycle averaged G: tanh C(\overline{G})      (f of avg G)

        Saves one GIF and a final PNG frame (with cycles in the title).
        '''
        Nx, Ny = self.Nx, self.Ny
        outdir = self._ensure_outdir('figs/chern_marker')
        histories = np.asarray(G_histories, dtype=np.complex128)
        if histories.ndim != 4:
            raise ValueError("G_histories must have shape (S, T, Ntot, Ntot)")
        S, T, dim1, dim2 = histories.shape
        if dim1 != self.Ntot or dim2 != self.Ntot:
            raise ValueError(f"Expected final dimensions ({self.Ntot},{self.Ntot}); got ({dim1},{dim2})")
        if S == 0 or T == 0:
            raise RuntimeError("No histories available.")
        Nlayer = self.Ntot // 2
        top_histories = histories[:, :, :Nlayer, :Nlayer]

        if outbasename is None:
            nshell_str = getattr(self, "nshell", None)
            nshell_str = "None" if nshell_str is None else str(nshell_str)
            outbasename = f"chern_marker_dynamics_N={Nx}_nshell={nshell_str}_cycles={T}_DWis{int(bool(self.DW))}"
        gif_path   = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.png")

        if not traj_avg:
            frames = []
            for t in range(T):
                Cmap = self.local_chern_marker_flat(top_histories[0, t])
                frames.append(Cmap)

            fig = plt.figure(figsize=(3.6, 4.0))
            ax  = fig.add_subplot(111)
            im  = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
            for sp in ax.spines.values():
                sp.set_linewidth(1.5); sp.set_color('black')
            ax.set_xlabel("y"); ax.set_ylabel("x")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Local Chern marker (cycles={T})")

            def _upd(i):
                im.set_data(frames[i])
                ax.set_title(f"Local Chern marker (t={i+1}/{T})")
                return [im]

            ani = animation.FuncAnimation(fig, _upd, frames=T, interval=500, blit=True)
            ani.save(gif_path, writer="pillow", dpi=120)
            final = frames[-1]
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(3.6, 4.0))
            im2 = ax2.imshow(final, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
            fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            ax2.set_xlabel("y"); ax2.set_ylabel("x")
            ax2.set_title(f"Local Chern marker — final (cycles={T})")
            fig2.savefig(final_path, bbox_inches='tight', dpi=140); plt.close(fig2)
            return gif_path, final_path, final, top_histories[0, -1]

        # traj_avg=True -> two-panel animation
        frames_L, frames_R = [], []
        for t in range(T):
            maps = [self.local_chern_marker_flat(top_histories[s, t]) for s in range(S)]
            frames_L.append(np.mean(maps, axis=0))
            Gbar_t = np.mean(top_histories[:, t], axis=0)
            frames_R.append(self.local_chern_marker_flat(Gbar_t))

        vmax_auto = max(np.max(np.abs(frames_L)), np.max(np.abs(frames_R)))
        vmax_auto = max(vmax_auto, 1.0)
        vmin_use, vmax_use = -vmax_auto, vmax_auto

        fig = plt.figure(figsize=(7.6, 4.2))
        axL = fig.add_subplot(1, 2, 1); axR = fig.add_subplot(1, 2, 2)
        imL = axL.imshow(frames_L[0], cmap=cmap, vmin=vmin_use, vmax=vmax_use, origin='upper', aspect='equal')
        imR = axR.imshow(frames_R[0], cmap=cmap, vmin=vmin_use, vmax=vmax_use, origin='upper', aspect='equal')
        axL.set_title(r"$\overline{\tanh\mathcal{C}}(\mathbf{r},t)$"); axR.set_title(r"$\tanh\mathcal{C}_{\overline{G}}(\mathbf{r},t)$")
        for ax in (axL, axR):
            ax.set_xlabel("y"); ax.set_ylabel("x")
        fig.colorbar(imL, ax=axL, fraction=0.046, pad=0.04)
        fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)
        fig.suptitle(f"Local Chern marker (cycles={T}, samples={S})")

        def _upd(i):
            imL.set_data(frames_L[i]); imR.set_data(frames_R[i])
            axL.set_title(rf"$\overline{{\tanh\mathcal{{C}}}}$ (t={i+1}/{T})")
            axR.set_title(rf"$\tanh\mathcal{{C}}(\overline{{G}})$ (t={i+1}/{T})")
            return [imL, imR]

        ani = animation.FuncAnimation(fig, _upd, frames=T, interval=500, blit=True)
        ani.save(gif_path, writer="pillow", dpi=120)
        plt.close(fig)

        fig2 = plt.figure(figsize=(7.6, 4.2))
        axL2 = fig2.add_subplot(1, 2, 1); axR2 = fig2.add_subplot(1, 2, 2)
        imL2 = axL2.imshow(frames_L[-1], cmap=cmap, vmin=vmin_use, vmax=vmax_use, origin='upper', aspect='equal')
        imR2 = axR2.imshow(frames_R[-1], cmap=cmap, vmin=vmin_use, vmax=vmax_use, origin='upper', aspect='equal')
        for ax in (axL2, axR2):
            ax.set_xlabel("y"); ax.set_ylabel("x")
        fig2.colorbar(imL2, ax=axL2, fraction=0.046, pad=0.04)
        fig2.colorbar(imR2, ax=axR2, fraction=0.046, pad=0.04)
        fig2.suptitle(f"Local Chern marker — final (cycles={T}, samples={S})")
        fig2.savefig(final_path, bbox_inches='tight', dpi=140); plt.close(fig2)
        return gif_path, final_path, (frames_L[-1], frames_R[-1]), (top_histories[0, -1],)

    # ---------------------- Parallel-safe spawner for v2 ----------------------

    def _spawn_for_parallel(self):
        """Create a lightweight worker instance for process-based parallelism.
        Shares large read-only arrays by reference; initializes per-worker state.
        """
        child = object.__new__(self.__class__)
    
        # ---- copy simple scalars / metadata ----
        child.Nx = self.Nx
        child.Ny = self.Ny
        child.Ntot = self.Ntot
        child.nshell = self.nshell
        child.DW = self.DW
        child.time_init = self.time_init
        child.DW_loc = getattr(self, "DW_loc", None)
    
        # ---- share big, read-only arrays (do NOT mutate these in workers) ----
        # NOTE: we intentionally DO NOT copy; workers should treat these as read-only.
        child.Pminus = self.Pminus
        child.Pplus  = self.Pplus
        child.WF_Ap  = self.WF_Ap
        child.WF_Bp  = self.WF_Bp
        child.WF_Am  = self.WF_Am
        child.WF_Bm  = self.WF_Bm
        child.alpha  = self.alpha
    
        # ---- per-worker mutable state ----
        # fresh starting state for each worker (top-layer randomized in your __init__)
        child.G0 = None if self.G0 is None else np.array(self.G0, dtype=np.complex128, copy=True)
        child.G = None
        child.G_list = []
        child.g2_flags = []
    
        # ---- suppress noisy worker output / ETA ----
        child._eta_step_baseline = None
        child._suppress_bottom_measure_prints = True
    
        # Any attributes used by RAC's ETA logic should default to benign values
        child._eta_parallel_factor = 1
    
        return child

    # ---------------------- Parallelized corr y-profiles (v2) ----------------------

    def plot_corr_y_profiles(self, G_histories, filename=None, ry_max=None, save=True, save_suffix=None, x_positions=None, spec=True, chern=True):
        '''
        Plot correlation profiles along y for supplied histories.

        Parameters
        ----------
        G_histories : ndarray
            Array of shape (S, T, Ntot, Ntot) containing full-layer covariances.
        filename : str or None
            Output filename (PDF) for the static panel. Defaults to a descriptive name.
        ry_max : int or None
            Maximum separation r_y to include; defaults to Ny//2.
        save : bool
            If True, write the static panel to disk.
        x_positions : iterable or None
            Optional list of x-indices (or (index,label) pairs) to plot.
        '''
        Nx, Ny = self.Nx, self.Ny
        Nlayer = self.Ntot // 2

        histories = np.asarray(G_histories, dtype=np.complex128)
        if histories.ndim != 4:
            raise ValueError("G_histories must have shape (S, T, ... , ...)")
        S, T, dim1, dim2 = histories.shape
        if dim1 != dim2:
            raise ValueError(f"Non-square history blocks: got shape ({dim1},{dim2})")
        if S == 0 or T == 0:
            raise RuntimeError("No histories available.")
        if dim1 == self.Ntot:
            top_histories = histories[:, :, :Nlayer, :Nlayer]
        elif dim1 == Nlayer:
            top_histories = histories
        else:
            raise ValueError(
                f"expects last dims {self.Ntot} or {Nlayer}; got {dim1}"
            )
        cycles = T

        # Defaults for plotting
        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # pick x positions
        def _pick_x_positions():
            if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
                xs = [
                    (xL // 2) % Nx,
                    (xL - 1) % Nx, xL % Nx, (xL + 1) % Nx,
                    ((xL + xR) // 2) % Nx,
                    (xR - 1) % Nx, xR % Nx, (xR + 1) % Nx,
                    (xR + (Nx // 2)) % Nx,
                ]
                seen, uniq = set(), []
                for x in xs:
                    if x not in seen:
                        uniq.append(int(x)); seen.add(int(x))
                return [(x, f"{x}") for x in uniq]
            else:
                xs = np.linspace(0, Nx-1, 9, dtype=int)
                return [(int(x), f"{int(x)}") for x in xs]

        if x_positions is None:
            x_positions = _pick_x_positions()
        else:
            parsed = []
            seen = set()
            for item in x_positions:
                if isinstance(item, (list, tuple)):
                    if not item:
                        continue
                    x_idx = int(item[0]) % Nx
                    label = str(item[1]) if len(item) > 1 else f"{x_idx}"
                else:
                    x_idx = int(item) % Nx
                    label = f"{x_idx}"
                if x_idx in seen:
                    continue
                seen.add(x_idx)
                parsed.append((x_idx, label))
            if not parsed:
                parsed = _pick_x_positions()
            x_positions = parsed

        # -------------- helpers --------------
        def _two_point_kernel_top(G_in):
            Gin = np.asarray(G_in, dtype=np.complex128)
            if Gin.ndim == 6:
                return Gin
            if Gin.shape != (Nlayer, Nlayer):
                raise ValueError(f"Expected top-layer covariance shape ({Nlayer},{Nlayer}); got {Gin.shape}")
            G2 = 0.5 * (Gin + np.eye(Nlayer, dtype=np.complex128))
            G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
            return np.transpose(G6, (1, 2, 0, 4, 5, 3))

        def _C_xslice_from_kernel(Gker, x0, ry_vals_arr):
            x0 = int(x0) % Nx
            ry_arr = np.atleast_1d(ry_vals_arr).astype(int)
            Ny_loc = Gker.shape[1]
            Gx = Gker[x0, :, :, x0, :, :]                # (Ny,2,Ny,2)
            Y  = np.arange(Ny_loc, dtype=np.intp)[:, None]
            Yp = (Y + ry_arr[None, :]) % Ny_loc
            Gx_re   = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny_loc*Ny_loc, 2, 2)
            flat_ix = (Y * Ny_loc + Yp).reshape(-1)
            blocks  = Gx_re[flat_ix].reshape(Ny_loc, ry_arr.size, 2, 2)
            return np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny_loc)

        def _chern_from_Gtop(G_top):
            return self.local_chern_marker_flat(G_top)

        # Final-step aggregates for x-positions
        C_accum = {x0: np.zeros_like(ry_vals, dtype=float) for x0, _ in x_positions}
        avg_enabled = S > 1
        if avg_enabled:
            Gsum_fin = np.zeros((Nlayer, Nlayer), dtype=np.complex128)
        else:
            Gsum_fin = None
        last_Gtt = None

        for s in range(S):
            Gtt_final = np.asarray(top_histories[s][-1], dtype=np.complex128)
            last_Gtt = Gtt_final
            Gker_fin = _two_point_kernel_top(Gtt_final)
            for x0, _ in x_positions:
                C_accum[x0] += _C_xslice_from_kernel(Gker_fin, x0, ry_vals).real
            if avg_enabled:
                Gsum_fin += Gtt_final

        C_resolved = {x0: C_accum[x0] / S for x0, _ in x_positions}
        if avg_enabled:
            Gavg_fin = Gsum_fin / S
            C_avg = {
                x0: _C_xslice_from_kernel(_two_point_kernel_top(Gavg_fin), x0, ry_vals).real
                for x0, _ in x_positions
            }
        else:
            Gavg_fin = None
            C_avg = None

        # spectra & Chern (final step)
        if spec:
            evals_last = np.linalg.eigvalsh(last_Gtt)
            if avg_enabled:
                evals_avg = np.linalg.eigvalsh(Gavg_fin)
            else:
                evals_avg = None
        else:
            evals_last = evals_avg = None

        if chern:
            Chern_last = _chern_from_Gtop(last_Gtt)
            if avg_enabled:
                Chern_avg = _chern_from_Gtop(Gavg_fin)
            else:
                Chern_avg = None
        else:
            Chern_last = Chern_avg = None

        # ---------------- plotting: 3 x 2 ----------------
        suffix = "" if save_suffix is None else str(save_suffix)
        outdir = self._ensure_outdir('figs/corr_y_profiles')
        if filename is None:
            xdesc = "-".join(f"{x}" for x, _ in x_positions)
            filename = f"corr2_y_profiles_v2_N{Nx}_xs_{xdesc}_S{S}.pdf"
        if suffix:
            root, ext = os.path.splitext(filename)
            filename = f"{root}{suffix}{ext}"
        fullpath = os.path.join(outdir, filename)

        mpl.rcParams['text.usetex'] = False

        if avg_enabled:
            row_count = 1 + (1 if spec else 0) + (1 if chern else 0)
            height_ratios = [1.0]
            if spec:
                height_ratios.append(0.9)
            if chern:
                height_ratios.append(1.05)
            fig_height = 4.0 * row_count
            fig = plt.figure(figsize=(12.5, fig_height), constrained_layout=True)
            gs  = fig.add_gridspec(nrows=row_count, ncols=2, height_ratios=height_ratios)

            current_row = 0
            axC1 = fig.add_subplot(gs[current_row, 0])
            axC2 = fig.add_subplot(gs[current_row, 1])

            current_row += 1
            if spec:
                axE1 = fig.add_subplot(gs[current_row, 0])
                axE2 = fig.add_subplot(gs[current_row, 1])
                current_row += 1
            else:
                axE1 = axE2 = None

            if chern:
                axM1 = fig.add_subplot(gs[current_row, 0])
                axM2 = fig.add_subplot(gs[current_row, 1])
            else:
                axM1 = axM2 = None

            # Top row: y-profiles (resolved vs averaged)
            for ax, Cdict, ylab in (
                (axC1, C_resolved, r"$\overline{C}_G(x_0,r_y)$"),
                (axC2, C_avg,      r"$C_{\overline{G}}(x_0,r_y)$" if C_avg is not None else r"$C_{\overline{G}}(x_0,r_y)$"),
            ):
                if Cdict is None:
                    ax.axis("off")
                    continue
                for x0, lbl in x_positions:
                    C_vec = Cdict[x0]
                    line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=fr"$x_0 = {lbl}$")
                    finite = np.isfinite(C_vec)
                    if np.any(finite):
                        y_right = C_vec[finite][-1]
                    else:
                        y_right = C_vec[-1]
                    if ry_vals[-1] > 0:
                        x_right = ry_vals[-1] * 1.02
                    else:
                        x_right = ry_vals[-1] + 0.5
                    ax.annotate(lbl, xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                                textcoords='data', ha='left', va='center', fontsize=8,
                                color=line.get_color())
                ax.set_xlabel(r"$r_y$")
                ax.set_ylabel(ylab)
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=7)
                if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                    ax.text(0.02, 0.96, fr"DWs at $x_0 = {int(self.DW_loc[0])}, \ {int(self.DW_loc[1])}$",
                            transform=ax.transAxes, ha='left', va='top', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", alpha=0.6))

            # sync y-axis limits
            ymin = axC1.get_ylim()[0]
            ymax = axC1.get_ylim()[1]
            if C_avg is not None:
                ymin = min(ymin, axC2.get_ylim()[0])
                ymax = max(ymax, axC2.get_ylim()[1])
                axC2.set_ylim(ymin, ymax)
            axC1.set_ylim(ymin, ymax)

            # Eigenvalue spectra row (optional)
            if spec and axE1 is not None and axE2 is not None:
                axE1.plot(np.arange(len(evals_last)), np.sort(evals_last), '.', ms=3)
                axE1.set_title(r"eigvals($G_{\mathrm{final}}$)")
                axE1.set_xlabel("index"); axE1.set_ylabel("eigenvalue"); axE1.grid(True, alpha=0.3)

                axE2.plot(np.arange(len(evals_avg)),  np.sort(evals_avg),  '.', ms=3)
                axE2.set_title(r"eigvals($\overline{G}_{\mathrm{final}}$)")
                axE2.set_xlabel("index"); axE2.set_ylabel("eigenvalue"); axE2.grid(True, alpha=0.3)

            # Chern maps row (optional)
            if chern and axM1 is not None and axM2 is not None:
                Chern_last = np.real_if_close(Chern_last, tol=1e-9)
                Chern_avg = np.real_if_close(Chern_avg, tol=1e-9)

                im1 = axM1.imshow(Chern_last, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
                axM1.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for final $G$")
                axM1.set_xlabel("y"); axM1.set_ylabel("x"); axM1.grid(False)
                fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)

                im2 = axM2.imshow(Chern_avg,  cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
                axM2.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for $\overline{G}$")
                axM2.set_xlabel("y"); axM2.set_ylabel("x"); axM2.grid(False)
                fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04)

            fig.suptitle(f"Correlation profiles (cycles={cycles}, samples={S})")
        else:
            row_count = 1 + (1 if spec else 0) + (1 if chern else 0)
            height_ratios = [1.1]
            if spec:
                height_ratios.append(0.9)
            if chern:
                height_ratios.append(1.05)
            fig_height = 3.7 * row_count
            fig = plt.figure(figsize=(7.0, fig_height), constrained_layout=True)
            gs = fig.add_gridspec(nrows=row_count, ncols=1, height_ratios=height_ratios)

            current_row = 0
            axC = fig.add_subplot(gs[current_row, 0])
            current_row += 1
            if spec:
                axE = fig.add_subplot(gs[current_row, 0])
                current_row += 1
            else:
                axE = None
            if chern:
                axM = fig.add_subplot(gs[current_row, 0])
            else:
                axM = None

            for x0, lbl in x_positions:
                C_vec = C_resolved[x0]
                line, = axC.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=fr"$x_0 = {lbl}$")
                finite = np.isfinite(C_vec)
                if np.any(finite):
                    y_right = C_vec[finite][-1]
                else:
                    y_right = C_vec[-1]
                if ry_vals[-1] > 0:
                    x_right = ry_vals[-1] * 1.02
                else:
                    x_right = ry_vals[-1] + 0.5
                axC.annotate(lbl, xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                             textcoords='data', ha='left', va='center', fontsize=8,
                             color=line.get_color())
            axC.set_xlabel(r"$r_y$")
            axC.set_ylabel(r"$C_G(x_0,r_y)$")
            axC.set_xscale('log'); axC.set_yscale('log')
            axC.grid(True, alpha=0.3)
            axC.legend(loc='best', fontsize=7)
            if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                axC.text(0.02, 0.96, fr"DWs at $x_0 = {int(self.DW_loc[0])}, \ {int(self.DW_loc[1])}$",
                         transform=axC.transAxes, ha='left', va='top', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", alpha=0.6))

            if spec and axE is not None:
                axE.plot(np.arange(len(evals_last)), np.sort(evals_last), '.', ms=3)
                axE.set_title(r"eigvals($G_{\mathrm{final}}$)")
                axE.set_xlabel("index"); axE.set_ylabel("eigenvalue"); axE.grid(True, alpha=0.3)

            if chern and axM is not None:
                Chern_last = np.real_if_close(Chern_last, tol=1e-9)
                im = axM.imshow(Chern_last, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
                axM.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for final $G$")
                axM.set_xlabel("y"); axM.set_ylabel("x"); axM.grid(False)
                fig.colorbar(im, ax=axM, fraction=0.046, pad=0.04)

            fig.suptitle(f"Correlation profiles (cycles={cycles}, single trajectory)")

        plt.show()
        if save:
            fig.savefig(fullpath, bbox_inches='tight'); plt.close(fig)

        return fullpath

    def plot_corr_y_scaling(self, G_histories, ry_max=None, x0_list=None, save=False, filename=None, power_fit=True, exp_fit=True):
        """
        Fit y-direction correlation profiles from the final snapshot to exponential and power-law forms.

        Parameters
        ----------
        G_histories : ndarray
            Array of shape (S, T, Ntot, Ntot) or (S, T, Nlayer, Nlayer) with covariance histories.
        ry_max : int or None
            Maximum r_y separation to include (defaults to Ny//2).
        x0_list : iterable or None
            Lattice x-indices where correlations are evaluated. Defaults to DW-centered triplet.
        save : bool
            If True, write the figure under figs/corr_y_fits.
        filename : str or None
            Optional filename when saving.
        """
        Nx, Ny = self.Nx, self.Ny
        Nlayer = self.Ntot // 2

        histories = np.asarray(G_histories, dtype=np.complex128)
        if histories.ndim != 4:
            raise ValueError("G_histories must have shape (S, T, ..., ...)")
        S, T, dim1, dim2 = histories.shape
        if dim1 != dim2:
            raise ValueError(f"Non-square history blocks: got shape ({dim1},{dim2})")
        if S == 0 or T == 0:
            raise RuntimeError("No histories available.")

        if dim1 == self.Ntot:
            top_histories = histories[:, :, :Nlayer, :Nlayer]
        elif dim1 == Nlayer:
            top_histories = histories
        else:
            raise ValueError(
                f"plot_corr_y_scaling expects last dims {self.Ntot} or {Nlayer}; got {dim1}"
            )

        # Default x positions: domain-wall edges and midpoint where available.
        if x0_list is None:
            if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
                x_mid = ((int(self.DW_loc[0]) + int(self.DW_loc[1])) // 2) % Nx
                candidates = [xL, x_mid, xR]
            else:
                candidates = [0, Nx // 2, (Nx - 1)]
            seen = set()
            x0_list = []
            for x in candidates:
                x_mod = int(x) % Nx
                if x_mod not in seen:
                    x0_list.append(x_mod)
                    seen.add(x_mod)
        else:
            dedup = []
            seen = set()
            for item in x0_list:
                x_val = int(item) % Nx
                if x_val not in seen:
                    dedup.append(x_val)
                    seen.add(x_val)
            if not dedup:
                raise ValueError("x0_list must contain at least one valid index.")
            x0_list = dedup

        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)
        if ry_vals.size < 2:
            raise ValueError("Need at least two r_y values for fitting; increase ry_max.")

        # Helpers replicated locally
        def _two_point_kernel_top(G_in):
            Gin = np.asarray(G_in, dtype=np.complex128)
            if Gin.ndim == 6:
                return Gin
            if Gin.shape != (Nlayer, Nlayer):
                raise ValueError(f"Expected top-layer covariance shape ({Nlayer},{Nlayer}); got {Gin.shape}")
            G2 = 0.5 * (Gin + np.eye(Nlayer, dtype=np.complex128))
            G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
            return np.transpose(G6, (1, 2, 0, 4, 5, 3))

        def _C_xslice_from_kernel(Gker, x0, ry_vals_arr):
            x0 = int(x0) % Nx
            ry_arr = np.atleast_1d(ry_vals_arr).astype(int)
            Ny_loc = Gker.shape[1]
            Gx = Gker[x0, :, :, x0, :, :]                # (Ny,2,Ny,2)
            Y  = np.arange(Ny_loc, dtype=np.intp)[:, None]
            Yp = (Y + ry_arr[None, :]) % Ny_loc
            Gx_re   = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny_loc*Ny_loc, 2, 2)
            flat_ix = (Y * Ny_loc + Yp).reshape(-1)
            blocks  = Gx_re[flat_ix].reshape(Ny_loc, ry_arr.size, 2, 2)
            return np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny_loc)

        # Aggregate the final snapshot across samples (average to reduce noise).
        G_final = np.mean(top_histories[:, -1], axis=0)
        Gker_final = _two_point_kernel_top(G_final)

        try:
            bulk_gap = self._bulk_band_gap()
        except Exception:
            bulk_gap = None

        def exp_model(r, a, b):
            return a * np.exp(b * r)

        def power_model(r, a, b, c):
            r = np.asarray(r, dtype=float)
            r_shift = r - c
            r_safe = np.where(r_shift > 0, r_shift, np.nan)
            return a * np.power(r_safe, b)

        fit_results = {}

        fig_width = 7.0 * 1.5
        fig_height = (3.6 * len(x0_list)) * 1.5
        fig, axes = plt.subplots(len(x0_list), 1, figsize=(fig_width, fig_height), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, x0 in zip(axes, x0_list):
            C_vals = _C_xslice_from_kernel(Gker_final, x0, ry_vals).real
            ax.plot(ry_vals, C_vals, marker='o', ms=4, lw=1, label=r"data", color="tab:blue")

            # Use r_y > 0 for fitting to avoid singularity in power-law.
            fit_mask = ry_vals > 0
            r_fit = ry_vals[fit_mask].astype(float)
            y_fit = C_vals[fit_mask].astype(float)
            if r_fit.size < 2:
                raise RuntimeError("Not enough points with r_y > 0 for fitting.")

            # Initial guesses
            y_adj = np.where(y_fit > 1e-12, y_fit, 1e-12)
            with np.errstate(divide='ignore', invalid='ignore'):
                logy = np.log(y_adj)
            slope = intercept = None
            if np.all(np.isfinite(logy)) and np.ptp(r_fit) > 0:
                slope, intercept = np.polyfit(r_fit, logy, 1)

            if exp_fit:
                a0_exp = np.exp(intercept) if (intercept is not None and np.isfinite(intercept)) else y_fit[0]
                if not np.isfinite(a0_exp) or a0_exp <= 0:
                    a0_exp = max(y_fit[0], 1e-12)

                if bulk_gap is not None and np.isfinite(bulk_gap) and bulk_gap > 1e-12:
                    b0_exp = -1.0 / bulk_gap
                elif slope is not None and np.isfinite(slope):
                    b0_exp = slope
                else:
                    b0_exp = -1.0 / max(r_fit.max(), 1.0)
                p0_exp = (
                    a0_exp if np.isfinite(a0_exp) else y_fit[0],
                    b0_exp if np.isfinite(b0_exp) else -1.0,
                )

                try:
                    popt_exp, pcov_exp = curve_fit(exp_model, r_fit, y_fit, p0=p0_exp, maxfev=20000)
                    perr_exp = np.sqrt(np.diag(pcov_exp))
                except Exception:
                    popt_exp = [np.nan, np.nan]
                    perr_exp = [np.nan, np.nan]
            else:
                popt_exp = [np.nan, np.nan]
                perr_exp = [np.nan, np.nan]

            if power_fit:
                a0_pow = y_fit[0]
                b0_pow = -1.0
                c0_pow = 0.0
                p0_pow = (
                    a0_pow if np.isfinite(a0_pow) else 1.0,
                    b0_pow,
                    c0_pow,
                )

                try:
                    popt_pow, pcov_pow = curve_fit(power_model, r_fit, y_fit, p0=p0_pow, maxfev=20000)
                    perr_pow = np.sqrt(np.diag(pcov_pow))
                except Exception:
                    popt_pow = [np.nan, np.nan, np.nan]
                    perr_pow = [np.nan, np.nan, np.nan]
            else:
                popt_pow = [np.nan, np.nan, np.nan]
                perr_pow = [np.nan, np.nan, np.nan]

            fit_results[x0] = {
                "exp": {"params": popt_exp, "stderr": perr_exp} if exp_fit else None,
                "power": {"params": popt_pow, "stderr": perr_pow} if power_fit else None,
            }

            r_plot = np.linspace(r_fit.min(), r_fit.max(), 400)
            if exp_fit and np.all(np.isfinite(popt_exp)):
                ax.plot(r_plot, exp_model(r_plot, *popt_exp), color="tab:orange",
                        linestyle=(0, (6, 2)), label="exp fit")
            if power_fit and np.all(np.isfinite(popt_pow)):
                ax.plot(r_plot, power_model(r_plot, *popt_pow), color="tab:green",
                        linestyle=(0, (3, 2, 1, 2)), label="power fit")

            text_lines = []
            if exp_fit and np.all(np.isfinite(popt_exp)):
                a, b = popt_exp
                da, db = perr_exp
                text_lines.append(r"$f_{\exp}(r)=a\,\exp(b r)$")
                text_lines.append(
                    f"a={a:.3f}±{da:.3f}, "
                    + f"b={b:.3f}±{db:.3f}"
                )
            elif exp_fit:
                text_lines.append("exp fit failed")
            if power_fit and np.all(np.isfinite(popt_pow)):
                a, b, c = popt_pow
                da, db, dc = perr_pow
                text_lines.append(r"$f_{\mathrm{pow}}(r)=a\,(r-c)^{b}$")
                text_lines.append(
                    f"a={a:.3f}±{da:.3f}, "
                    + f"b={b:.3f}±{db:.3f}, "
                    + f"c={c:.3f}±{dc:.3f}"
                )
            elif power_fit:
                text_lines.append("power fit failed")
            if not exp_fit and not power_fit:
                text_lines.append("no fits requested")

            ax.text(0.02, 0.02, "\n".join(text_lines), transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax.set_title(f"x0 = {x0}")
            ax.set_xlabel(r"$r_y$")
            ax.set_ylabel(r"$C_G(x_0,r_y)$")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"Correlation scaling fits (cycles={T}, samples={S})")
        plt.show()

        saved_path = None
        if save:
            outdir = self._ensure_outdir(os.path.join("figs", "corr_y_fits"))
            if filename is None:
                xdesc = "-".join(str(x) for x in x0_list)
                filename = f"corr_y_scaling_N{Nx}_xs_{xdesc}_S{S}_T{T}.pdf"
            saved_path = os.path.join(outdir, filename)
            fig.savefig(saved_path, bbox_inches="tight")
            plt.close(fig)

        return saved_path, fit_results
    
    # ------------------ Entanglement Contour Block ------------------



    def entanglement_contour(self, Gtt):
        '''Compute the entanglement contour s(r) for the top-layer covariance.'''
        Nx, Ny = self.Nx, self.Ny
        arr = np.asarray(Gtt, dtype=np.complex128)
        if arr.ndim != 2:
            raise ValueError(f"entanglement_contour expects a 2D covariance; got shape {arr.shape}")

        Nlayer = arr.shape[0]
        I  = np.eye(Nlayer, dtype=np.complex128)
        G2 = 0.5 * (I + arr)

        # spectral functional calculus
        evals, vecs = np.linalg.eigh(G2)
        evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
        f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))  # shape (Nlayer,)

        # F = V diag(f_eigs) V† ; we only need diag(F)
        # diag(F) = sum_k f_eigs[k] * |vecs[i,k]|^2
        diagF = np.einsum("ik,k,ik->i", vecs, f_eigs, vecs.conj(), optimize=True).real

        # reshape i = μ + 2*x + 2*Nx*y (μ fastest)
        diagF = diagF.reshape(2, Nx, Ny, order="F")   # (μ, x, y)
        s = diagF.sum(axis=0)                         # sum over μ -> (Nx,Ny)
        return s

    def entanglement_contour_suite(
        self,
        G_histories,
        filename_profiles=None,
        filename_prefix_dyn=None,
        save=True,
        save_suffix=None,
        custom_x_positions=None,
    ):
        r'''
        Entanglement-contour analysis driven by supplied histories.

        Parameters
        ----------
        G_histories : ndarray
            Array of shape (S, T, Ntot, Ntot) containing full-layer covariances.
        filename_profiles, filename_prefix_dyn : str, optional
            Override output filenames for static plots / GIFs.
        save : bool
            Whether to write outputs to disk.
        save_suffix : str, optional
            When provided, this string is appended (before the extension) to every file
            path that this routine writes or returns.
        custom_x_positions : iterable, optional
            Custom x-position list; supply integers or (index, label) pairs. When omitted,
            a heuristic set tied to DW locations is used.

        Always shows:
        Row 1: time-profiles at smart x0 — LEFT = traj-resolved \overline{s_G}(t), RIGHT = s_{Ḡ}(t)
        Row 2: eigenvalue spectra (final-step) — traj vs Ḡ
        Row 3: final maps — s(G_final^{(traj)}) vs s(Ḡ_final)

        Also produces two GIFs (traj map & traj-avg map) with colorbars.
        '''
        Nx, Ny = self.Nx, self.Ny
        Nlayer = self.Ntot // 2

        suffix = "" if save_suffix is None else str(save_suffix)

        def _with_suffix(path: str) -> str:
            if not suffix:
                return path
            root, ext = os.path.splitext(path)
            return f"{root}{suffix}{ext}"
        
        histories = np.asarray(G_histories, dtype=np.complex128)
        if histories.ndim != 4:
            raise ValueError("G_histories must have shape (S, T, ... , ...)")
        S, T, dim1, dim2 = histories.shape
        if dim1 != dim2:
            raise ValueError(f"Non-square blocks in histories: ({dim1},{dim2})")
        if S == 0 or T == 0:
            raise RuntimeError("No histories available.")
        if dim1 == self.Ntot:
            top_histories = histories[:, :, :Nlayer, :Nlayer]
        elif dim1 == Nlayer:
            top_histories = histories
        else:
            raise ValueError(
                f"plot_corr_y_profiles_v2 expects last dims {self.Ntot} or {Nlayer}; got {dim1}"
            )
        contours = np.empty((S, T, Nx, Ny), dtype=np.float64)
        for s in range(S):
            for t in range(T):
                contours[s, t] = self.entanglement_contour(top_histories[s, t])
        cycles = T
        multi_sample = S > 1

        # smart x-positions
        if custom_x_positions is None:
            def _pick_x_positions():
                if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                    xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
                    xs = [
                        (xL // 2) % Nx,
                        (xL - 1) % Nx, xL % Nx, (xL + 1) % Nx,
                        ((xL + xR) // 2) % Nx,
                        (xR - 1) % Nx, xR % Nx, (xR + 1) % Nx,
                        (xR + (Nx // 2)) % Nx,
                    ]
                    seen, uniq = set(), []
                    for x in xs:
                        if x not in seen:
                            uniq.append(int(x)); seen.add(int(x))
                    return [(x, f"{x}") for x in uniq]
                else:
                    xs = np.linspace(0, Nx-1, 9, dtype=int)
                    return [(int(x), f"{int(x)}") for x in xs]
            x_positions = _pick_x_positions()
        else:
            parsed = []
            seen = set()
            for item in custom_x_positions:
                if isinstance(item, (list, tuple)):
                    if not item:
                        continue
                    x_idx = int(item[0]) % Nx
                    label = str(item[1]) if len(item) > 1 else f"{x_idx}"
                else:
                    x_idx = int(item) % Nx
                    label = f"{x_idx}"
                if x_idx in seen:
                    continue
                seen.add(x_idx)
                parsed.append((x_idx, label))
            if not parsed:
                raise ValueError("custom_x_positions provided no valid indices.")
            x_positions = parsed
        xs_only = [x for x, _ in x_positions]

        # Build time-profiles
        traj_profiles_mean = {
            x0: np.array([
                np.mean([float(np.sum(contours[s, t, x0, :])) for s in range(S)])
                for t in range(T)
            ], dtype=float)
            for x0 in xs_only
        }

        if multi_sample:
            Gavg_hist = np.mean(top_histories, axis=0)
            avg_maps  = np.mean(contours, axis=0)
            avg_profiles = {
                x0: np.array([float(np.sum(avg_maps[t, x0, :])) for t in range(T)], dtype=float)
                for x0 in xs_only
            }
        else:
            Gavg_hist = None
            avg_maps = None
            avg_profiles = None

        # spectra + final maps
        G_final_traj = top_histories[0, -1]
        G_final_avg  = Gavg_hist[-1] if multi_sample else None
        ev_traj = np.linalg.eigvalsh(G_final_traj)
        ev_avg  = np.linalg.eigvalsh(G_final_avg) if multi_sample else None
        final_traj_map = contours[0, -1]
        final_avg_map  = avg_maps[-1] if multi_sample else None

        # --------------- profiles + spectra + final maps ---------------
        outdir_prof = self._ensure_outdir("figs/entanglement_contour")
        if filename_profiles is None:
            xdesc = "-".join(f"{x}" for x in xs_only)
            filename_profiles = f"entanglement_suite_yprofiles_N{Nx}_xs_{xdesc}_S{S}.pdf"
        profiles_pdf = _with_suffix(os.path.join(outdir_prof, filename_profiles))

        if multi_sample:
            fig = plt.figure(constrained_layout=True, figsize=(12.5, 16.0))
            gs  = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1.1, 1.0, 1.0, 1.0])
            axP1 = fig.add_subplot(gs[0, 0]); axP2 = fig.add_subplot(gs[0, 1])
            axS1 = fig.add_subplot(gs[1, 0]); axS2 = fig.add_subplot(gs[1, 1])
            axM1 = fig.add_subplot(gs[2, 0]); axM2 = fig.add_subplot(gs[2, 1])
            axChern1 = fig.add_subplot(gs[3, 0]); axChern2 = fig.add_subplot(gs[3, 1])
        else:
            fig = plt.figure(constrained_layout=True, figsize=(6.0, 16.0))
            gs  = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1.1, 1.0, 1.0, 1.0])
            axP1 = fig.add_subplot(gs[0, 0])
            axS1 = fig.add_subplot(gs[1, 0])
            axM1 = fig.add_subplot(gs[2, 0])
            axChern1 = fig.add_subplot(gs[3, 0])

        t_vals = np.arange(1, T + 1)

        for x0, lbl in x_positions:
            axP1.plot(t_vals, traj_profiles_mean[x0], label=lbl, marker='o', ms=3)
        axP1.set_xlabel("cycle t"); axP1.set_ylabel(r"$\sum_y s(x_0,y)$")
        title_str = r"$s_G$" if not multi_sample else r"$\overline{s_{G}}$ (traj-resolved mean)"
        axP1.set_title(title_str)
        if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
            axP1.text(
                0.02,
                0.94,
                fr"DWs at $x_0={int(self.DW_loc[0])}, {int(self.DW_loc[1])}$",
                transform=axP1.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", alpha=0.6),
            )
        axP1.set_yscale("log"); axP1.grid(True, alpha=0.3); axP1.legend(fontsize=7, ncol=2)

        if multi_sample:
            for x0, lbl in x_positions:
                axP2.plot(t_vals, avg_profiles[x0], label=lbl, marker='o', ms=3)
            axP2.set_xlabel("cycle t"); axP2.set_ylabel(r"$\sum_y s(x_0,y)$")
            axP2.set_title(r"$s_{\overline{G}}$"); axP2.set_yscale("log"); axP2.grid(True, alpha=0.3); axP2.legend(fontsize=7, ncol=2)

            y1 = axP1.get_ylim(); y2 = axP2.get_ylim()
            ymin = min(y1[0], y2[0]); ymax = max(y1[1], y2[1])
            axP1.set_ylim(ymin, ymax); axP2.set_ylim(ymin, ymax)

            axS1.plot(np.arange(len(ev_traj)), np.sort(ev_traj), '.', ms=3); axS1.grid(True, alpha=0.3)
            axS2.plot(np.arange(len(ev_avg)),  np.sort(ev_avg),  '.', ms=3); axS2.grid(True, alpha=0.3)
            axS1.set_title(r"eigvals($G_{\mathrm{final}}$) (traj)"); axS2.set_title(r"eigvals($\overline{G}_{\mathrm{final}}$)")
            axS1.set_xlabel("index"); axS1.set_ylabel("eigenvalue")
            axS2.set_xlabel("index"); axS2.set_ylabel("eigenvalue")

            im1 = axM1.imshow(final_traj_map, cmap="Blues", origin="upper", aspect="equal")
            im2 = axM2.imshow(final_avg_map,  cmap="Blues", origin="upper", aspect="equal")
            for ax in (axM1, axM2):
                ax.set_xlabel("y"); ax.set_ylabel("x")
            fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04)
            axM1.set_title("Final $s_G$ (traj)"); axM2.set_title(r"Final $s_{\overline{G}}$")
        else:
            axP1.legend(fontsize=7, ncol=2)
            axS1.plot(np.arange(len(ev_traj)), np.sort(ev_traj), '.', ms=3); axS1.grid(True, alpha=0.3)
            axS1.set_title(r"eigvals($G_{\mathrm{final}}$) (traj)"); axS1.set_xlabel("index"); axS1.set_ylabel("eigenvalue")

            im1 = axM1.imshow(final_traj_map, cmap="Blues", origin="upper", aspect="equal")
            axM1.set_xlabel("y"); axM1.set_ylabel("x")
            fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)
            axM1.set_title("Final $s_G$ (traj)")

        chern_traj = self.local_chern_marker_flat(G_final_traj)
        chern_vmax = float(np.max(np.abs(chern_traj)))
        if chern_vmax <= 0:
            chern_vmax = 1.0
        imC1 = axChern1.imshow(
            chern_traj,
            cmap="RdBu_r",
            origin="upper",
            aspect="equal",
            vmin=-chern_vmax,
            vmax=chern_vmax,
        )
        axChern1.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ (traj)")
        axChern1.set_xlabel("y"); axChern1.set_ylabel("x")
        fig.colorbar(imC1, ax=axChern1, fraction=0.046, pad=0.04)

        if multi_sample:
            chern_avg = self.local_chern_marker_flat(G_final_avg)
            chern_vmax = max(chern_vmax, float(np.max(np.abs(chern_avg))))
            if chern_vmax <= 0:
                chern_vmax = 1.0
            imC1.set_clim(-chern_vmax, chern_vmax)
            imC2 = axChern2.imshow(
                chern_avg,
                cmap="RdBu_r",
                origin="upper",
                aspect="equal",
                vmin=-chern_vmax,
                vmax=chern_vmax,
            )
            axChern2.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ (traj-avg)")
            axChern2.set_xlabel("y"); axChern2.set_ylabel("x")
            fig.colorbar(imC2, ax=axChern2, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Entanglement contour (cycles={cycles}, samples={S})\n",
            y=1.03
        )
        plt.show()
        if save:
            fig.savefig(profiles_pdf, bbox_inches="tight", dpi=140); plt.close(fig)

        # --------------- dynamics GIFs ---------------
        outdir_dyn = self._ensure_outdir("figs/entanglement_contour_dynamics")
        if filename_prefix_dyn is None:
            filename_prefix_dyn = f"entanglement_dyn_N{Nx}_S{S}"

        if multi_sample:
            traj_maps_for_gif = np.mean(contours, axis=0)
        else:
            traj_maps_for_gif = contours[0]

        dyn_final_png = _with_suffix(os.path.join(outdir_dyn, f"{filename_prefix_dyn}_final.png"))
        if multi_sample:
            figF = plt.figure(constrained_layout=True, figsize=(10, 4))
            axsF = figF.subplots(1, 2, squeeze=True)
            imf1 = axsF[0].imshow(traj_maps_for_gif[-1], cmap="Blues", origin="upper", aspect="equal")
            imf2 = axsF[1].imshow(avg_maps[-1], cmap="Blues", origin="upper", aspect="equal")
            for ax in axsF:
                ax.set_xlabel("y"); ax.set_ylabel("x")
            figF.colorbar(imf1, ax=axsF[0], fraction=0.046, pad=0.04)
            figF.colorbar(imf2, ax=axsF[1], fraction=0.046, pad=0.04)
        else:
            figF = plt.figure(constrained_layout=True, figsize=(5.0, 4.0))
            axsF = [figF.add_subplot(111)]
            imf1 = axsF[0].imshow(traj_maps_for_gif[-1], cmap="Blues", origin="upper", aspect="equal")
            axsF[0].set_xlabel("y"); axsF[0].set_ylabel("x")
            figF.colorbar(imf1, ax=axsF[0], fraction=0.046, pad=0.04)
        figF.suptitle(r"Dynamics final frames — initial top layer maximally mixed ($G_{tt}=0$)", y=1.02)
        plt.show()

        if save:
            figF.savefig(dyn_final_png, dpi=150, bbox_inches="tight")

        def _make_gif(maps, fname, title):
            fig = plt.figure(constrained_layout=True, figsize=(5.6, 4.6))
            ax = fig.add_subplot(111)
            first = maps[0]
            initial_max = float(np.max(first))
            if initial_max <= 0:
                initial_max = 1.0
            im = ax.imshow(first, cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=initial_max)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title); ax.set_xlabel("y"); ax.set_ylabel("x")
            def update(i):
                frame = maps[i]
                im.set_data(frame)
                vmax = float(np.max(frame))
                if vmax <= 0:
                    vmax = 1.0
                im.set_clim(0.0, vmax)
                cbar.update_normal(im)
                ax.set_title(f"{title}, cycle {i}")
                return [im]
            ani = animation.FuncAnimation(fig, update, frames=len(maps), interval=400, blit=True)
            ani.save(_with_suffix(os.path.join(outdir_dyn, fname)), writer="pillow", dpi=120)
            plt.close(fig)

        dyn_gif_traj = f"{filename_prefix_dyn}_traj.gif"
        dyn_gif_avg  = f"{filename_prefix_dyn}_avg.gif" if multi_sample else None
        _make_gif(traj_maps_for_gif, dyn_gif_traj, r"$s_G(r,t)$ (traj-resolved)")
        if multi_sample:
            _make_gif(avg_maps, dyn_gif_avg,  r"$s_{\overline{G}}(r,t)$ (traj-averaged)")

        return {
            "profiles_pdf": profiles_pdf,
            "dyn_dir": outdir_dyn,
            "dyn_gif_traj": _with_suffix(os.path.join(outdir_dyn, dyn_gif_traj)),
            "dyn_gif_avg":  _with_suffix(os.path.join(outdir_dyn, dyn_gif_avg)) if dyn_gif_avg else None,
            "dyn_final_png": dyn_final_png,
        }
   
