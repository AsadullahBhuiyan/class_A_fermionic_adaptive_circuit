import numpy as np
import math
import time
import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from contextlib import contextmanager
try:
    from tqdm.auto import tqdm
except Exception:
    from tqdm import tqdm

# Configure thread counts for BLAS libraries to avoid oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# Optionally pin the parent process to scheduler-provided CPUs.
try:
    _CPUS = int(
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("MY_CPU_COUNT")
        or 4
    )
    os.sched_setaffinity(0, set(range(_CPUS)))
    print(f"[info] CPU affinity pinned to {_CPUS} cores.")
except Exception as e:
    print(f"[warn] Could not set CPU affinity: {e}")


class CI_Lindblad_DW:
    '''Lindbladian evolution helper for single-layer correlation matrices on a torus.'''

    # ------------------------------ Init & Setup ------------------------------

    def __init__(self,
                 Nx, Ny,
                 dt=5e-2,
                 nshell=None,
                 DW=True,
                 n_a=0.5,
                 alpha_1=3,
                 alpha_2=1):
        '''Initialize lattice geometry and simulation parameters.'''

        self.Nx, self.Ny = int(Nx), int(Ny)
        self.dt = float(dt)
        self.nshell = None if nshell is None else int(nshell)
        self.n_a = float(n_a)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        # Mass profile alpha(x) for the Dirac-CI model
        self.alpha_profile = np.full((Nx, Ny), float(alpha_1), dtype=np.complex128)
        if DW:
            self.create_domain_wall(alpha_1=self.alpha_1, alpha_2=self.alpha_2)
        else:
            self.DW_loc = None

        # Set after an evolution run
        self.steps_per_cycle = None
        self.max_steps = None
        self.G_history = None
        self.history_steps = None
        self._decoh_last = None

    @property
    def Ntot(self):
        return 2 * self.Nx * self.Ny

    # --------------------------- Internal helpers ---------------------------

    def _require_block(self, G_block):
        '''Validate and return a single snapshot shaped (Nx, Ny, 2, Nx, Ny, 2).'''
        expected = (self.Nx, self.Ny, 2, self.Nx, self.Ny, 2)
        arr = np.asarray(G_block, dtype=np.complex128)
        if arr.shape != expected:
            raise ValueError(f"Snapshot must have shape {expected}, got {arr.shape}.")
        return arr

    def _coerce_block_snapshot(self, G):
        '''Convert a variety of G shapes to the canonical (Nx, Ny, 2, Nx, Ny, 2) block form.'''
        expected = (self.Nx, self.Ny, 2, self.Nx, self.Ny, 2)
        arr = np.asarray(G, dtype=np.complex128)
        if arr.shape == expected:
            return arr
        if arr.ndim == 2 and arr.shape == (self.Ntot, self.Ntot):
            return arr.reshape(expected, order='C')
        if arr.ndim == 7 and arr.shape[1:] == expected:
            if arr.shape[0] == 1:
                return arr[0]
            raise ValueError("Received a time-history array; please supply a single snapshot.")
        raise ValueError(f"G cannot be coerced to shape {expected}; got {arr.shape}.")

    def _require_block_history(self, G_data):
        '''Validate and return a stack of block-shaped snapshots.'''
        expected_single = (self.Nx, self.Ny, 2, self.Nx, self.Ny, 2)

        if isinstance(G_data, (list, tuple)):
            if not G_data:
                raise ValueError("Empty snapshot sequence provided.")
            stacked = [self._require_block(snap) for snap in G_data]
            return np.stack(stacked, axis=0)

        arr = np.asarray(G_data, dtype=np.complex128)
        if arr.ndim == 6:
            if arr.shape != expected_single:
                raise ValueError(f"Snapshot must have shape {expected_single}, got {arr.shape}.")
            return arr[np.newaxis, ...]
        if arr.ndim == 7:
            if arr.shape[1:] != expected_single:
                raise ValueError("History array must have shape (T, Nx, Ny, 2, Nx, Ny, 2).")
            if arr.shape[0] == 0:
                raise ValueError("History array must contain at least one snapshot.")
            return arr

        raise ValueError("G_data must be block-shaped with ndim 6 or 7, or a sequence of snapshots.")

    def _block_to_dense(self, G_block):
        '''Flatten a block snapshot to its (Ntot, Ntot) dense representation.'''
        block = self._require_block(G_block)
        return block.reshape(self.Ntot, self.Ntot, order='C')

    def _coerce_dense_snapshot(self, G):
        '''Convert a snapshot to its dense (Ntot, Ntot) single-particle matrix form.'''
        return self._block_to_dense(self._coerce_block_snapshot(G))

    def _matrix_to_col_vec(self, M):
        '''Column-stack a dense single-particle matrix.'''
        dense = np.asarray(M, dtype=np.complex128)
        expected = (self.Ntot, self.Ntot)
        if dense.shape != expected:
            raise ValueError(f"Matrix must have shape {expected}, got {dense.shape}.")
        return dense.reshape(self.Ntot * self.Ntot, order='F')

    def _col_vec_to_matrix(self, vec):
        '''Inverse of _matrix_to_col_vec for dense single-particle matrices.'''
        arr = np.asarray(vec, dtype=np.complex128)
        expected = (self.Ntot * self.Ntot,)
        if arr.shape != expected:
            raise ValueError(f"Vector must have shape {expected}, got {arr.shape}.")
        return arr.reshape(self.Ntot, self.Ntot, order='F')

    def _lift_affine_generator(self, generator, source):
        '''Embed d/dt vec(G) = source + generator @ vec(G) into a homogeneous linear system.'''
        gen = np.asarray(generator, dtype=np.complex128)
        src = np.asarray(source, dtype=np.complex128)
        if gen.ndim != 2 or gen.shape[0] != gen.shape[1]:
            raise ValueError(f"Generator must be square, got {gen.shape}.")
        if src.ndim != 1:
            raise ValueError(f"Source must be one-dimensional, got {src.shape}.")
        dim = gen.shape[0]
        if src.shape[0] != dim:
            raise ValueError(
                f"Source length must match generator dimension {dim}, got {src.shape[0]}."
            )

        lifted = np.zeros((dim + 1, dim + 1), dtype=np.complex128)
        lifted[:-1, :-1] = gen
        lifted[:-1, -1] = src
        return lifted

    def _get_decoh_flag(self, fallback=False):
        '''Return the most recent decoherence flag for labeling plots.'''
        if self._decoh_last is None:
            return bool(fallback)
        return bool(self._decoh_last)

    def _domain_wall_hamiltonian(self, periodic=True, alpha=None):
        """
        Build the real-space Dirac/Chern Hamiltonian with spatially varying mass alpha(x,y).
        Returns a (2*Nx*Ny, 2*Nx*Ny) matrix in the same single-particle basis as G.
        """
        if alpha is None:
            if not hasattr(self, "alpha_profile"):
                self.create_domain_wall(alpha_1=self.alpha_1, alpha_2=self.alpha_2)
            alpha = self.alpha_profile

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
        hop_y = -0.5 * sigma_z - 0.5j * sigma_y

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

    def _flatten_full_matrix(self, G6):
        '''Flatten snapshot to (2*Nx*Ny, 2*Nx*Ny) using Fortran ordering over (μ,x,y).'''
        block = self._require_block(G6)
        Nx, Ny = self.Nx, self.Ny
        block = np.transpose(block, (2, 0, 1, 5, 3, 4))
        return block.reshape(2 * Nx * Ny, 2 * Nx * Ny, order='F')

    def _fft_y_blocks_from_flat(self, G6, norm='ortho', hermitize=True):
        '''Return ky values and contiguous (2*Nx)x(2*Nx) blocks after y FFTs.'''
        block = self._coerce_block_snapshot(G6)
        Nx, Ny = self.Nx, self.Ny

        Gk = np.fft.fft(block, axis=1, norm=norm)
        Gk = np.fft.ifft(Gk, axis=4, norm=norm)

        G_flat = self._flatten_full_matrix(Gk)
        if hermitize:
            G_flat = 0.5 * (G_flat + G_flat.conj().T)

        blocks = []
        for j in range(Ny):
            lo = 2 * j * Nx
            hi = 2 * (j + 1) * Nx
            Bj = G_flat[lo:hi, lo:hi]
            if hermitize:
                Bj = 0.5 * (Bj + Bj.conj().T)
            blocks.append(Bj)

        ky_vals = 2 * np.pi * np.fft.fftfreq(Ny, d=1.0)
        return ky_vals, blocks

    def _y_partial_fourier_unitary(self, norm='ortho'):
        '''Unitary that Fourier transforms the translation-invariant y direction in the dense basis.'''
        Fy = np.fft.fft(np.eye(self.Ny, dtype=np.complex128), norm=norm)
        return np.kron(
            np.eye(self.Nx, dtype=np.complex128),
            np.kron(Fy, np.eye(2, dtype=np.complex128)),
        )

    def _ky_basis_indices(self, ky_index):
        '''Dense-basis indices for a fixed ky block after the y partial Fourier transform.'''
        idx = []
        for x in range(self.Nx):
            base = 2 * (x * self.Ny + int(ky_index))
            idx.extend([base, base + 1])
        return np.asarray(idx, dtype=int)

    def _vec_subspace_indices(self, basis_indices):
        '''Vectorized indices associated with matrices supported on the supplied dense-basis subset.'''
        basis = np.asarray(basis_indices, dtype=int)
        return np.asarray(
            [row + self.Ntot * col for col in basis for row in basis],
            dtype=int,
        )

    def _q_sector_vec_indices(self, q_index):
        '''Vectorized indices for the exact momentum-transfer sector q = ky_row - ky_col (mod Ny).'''
        q = int(q_index) % self.Ny
        sector = []
        for ky_col in range(self.Ny):
            ky_row = (ky_col + q) % self.Ny
            row_basis = self._ky_basis_indices(ky_row)
            col_basis = self._ky_basis_indices(ky_col)
            sector.extend([row + self.Ntot * col for col in col_basis for row in row_basis])
        return np.asarray(sector, dtype=int)

    def _partial_fourier_single_particle_blocks(self, matrix, norm='ortho'):
        '''Fourier transform a dense single-particle matrix in y and return its diagonal ky blocks.'''
        dense = np.asarray(matrix, dtype=np.complex128)
        expected = (self.Ntot, self.Ntot)
        if dense.shape != expected:
            raise ValueError(f"Matrix must have shape {expected}, got {dense.shape}.")

        U_y = self._y_partial_fourier_unitary(norm=norm)
        dense_k = U_y.conj().T @ dense @ U_y
        blocks = []
        for j in range(self.Ny):
            basis_idx = self._ky_basis_indices(j)
            blocks.append(dense_k[np.ix_(basis_idx, basis_idx)].copy())
        return dense_k, blocks

    def _ky_spectrum_evals(self, G6, norm='ortho', hermitize=True):
        '''Compute eigenvalues of contiguous ky blocks for snapshot G6.'''
        ky_vals, blocks = self._fft_y_blocks_from_flat(G6, norm=norm, hermitize=hermitize)
        Nx = self.Nx
        evals = np.empty((len(blocks), 2 * Nx), dtype=float)
        for j, block in enumerate(blocks):
            w = np.linalg.eigvalsh(block)
            evals[j, :] = np.sort(np.real_if_close(w))
        return ky_vals, evals

    def plot_ky_spectrum(self, G, norm='ortho', hermitize=True, ax=None, marker='.', ms=3):
        '''
        FFT on y (bra) and IFFT on y' (ket), then flatten and slice contiguous
        (2*Nx)x(2*Nx) blocks in the k_y basis:
            block j = G_flat[2*j*Nx:2*(j+1)*Nx, 2*j*Nx:2*(j+1)*Nx]
        Diagonalize each block and plot ordered eigenvalues vs k_y.

        Returns:
            ky_vals (Ny,), evals (Ny, 2*Nx), ax
        '''
        block = self._coerce_block_snapshot(G)
        ky_vals, blocks = self._fft_y_blocks_from_flat(block, norm=norm, hermitize=hermitize)

        Nx, Ny = self.Nx, self.Ny
        evals = np.empty((Ny, 2 * Nx), dtype=float)
        for j, B in enumerate(blocks):
            w = np.linalg.eigvalsh(B)
            evals[j, :] = np.sort(np.real_if_close(w))

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.2))
        else:
            fig = ax.figure

        for band in evals.T:
            ax.plot(ky_vals, band, marker, ms=ms, lw=0)

        ax.set_xlabel(r"$k_y$")
        ax.set_ylabel("eigenvalue")
        ax.set_title(r"Spectrum of $G(k_y)$ blocks (contiguous $2N_x$ rule)")
        ax.grid(True, alpha=0.3)
        return ky_vals, evals, ax

    # ---------------------------- Outdir & Caching ----------------------------

    def _ensure_outdir(self, path):
        '''Create the output directory if needed and return its path.'''
        os.makedirs(path, exist_ok=True)
        return path

    # ----------------------- Domain Wall & OW Construction ----------------------
    def create_domain_wall(self, alpha_1, alpha_2):
        '''Build the domain-wall mass profile and store its location.'''
        Nx = self.Nx
        half = Nx // 2
        w = max(1, int(np.floor(0.2 * Nx)))
        x0 = max(0, half - w)
        x1 = min(Nx, half + w + 1)  # inclusive slab -> slice end-exclusive
        self.alpha_profile[x0:x1, :] = alpha_2        # topological region
        print(f"DWs at x=({int(x0)}, {int(x1-1)})")
        self.DW_loc = [int(x0), int(x1-1)]

    def construct_OW_functions(self):
        '''Assemble overcomplete Wannier projectors and derived tensors.'''

        alpha = self.alpha_profile

        # k-grids
        kx = 2*np.pi * np.fft.fftfreq(self.Nx, d=1.0)
        ky = 2*np.pi * np.fft.fftfreq(self.Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        # model vector n(k)
        nx = np.sin(KX)[:, :, None, None]
        ny = np.sin(KY)[:, :, None, None]
        nz = alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None]
        nmag = np.sqrt(nx**2 + ny**2 + nz**2)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        # Pauli
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        Id = np.eye(2, dtype=complex)

        hk = (nx[..., None, None]*sx + ny[..., None, None]*sy + nz[..., None, None]*sz) / nmag[..., None, None]
        self.Pminus = 0.5 * (Id - hk)
        self.Pplus  = 0.5 * (Id + hk)

        # Local τA/τB spinors
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)

        Rx_grid = np.arange(self.Nx)
        Ry_grid = np.arange(self.Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])
        phase   = phase_x * phase_y

        def k2_to_r2(Ak):
            return np.fft.fft2(Ak, axes=(0, 1))

        def _square_window_mask(nshell):
            Nx, Ny = self.Nx, self.Ny
            x = np.arange(Nx)[:, None, None, None]
            y = np.arange(Ny)[None, :, None, None]
            Rx = np.arange(Nx)[None, None, :, None]
            Ry = np.arange(Ny)[None, None, None, :]
            dx_wrap = ((x - Rx + Nx//2) % Nx) - Nx//2
            dy_wrap = ((y - Ry + Ny//2) % Ny) - Ny//2
            return (np.abs(dx_wrap) <= nshell) & (np.abs(dy_wrap) <= nshell)

        def make_W(Pband, tau, phase):
            tau_dag = tau[:, 0].conj()
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband)  # (...,2)

            F0 = phase * psi_k[..., 0]
            F1 = phase * psi_k[..., 1]
            W0 = k2_to_r2(F0)
            W1 = k2_to_r2(F1)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)  # (Nx,Ny,2,Rx,Ry)

            if self.nshell is not None:
                mask = _square_window_mask(self.nshell)          # (Nx,Ny,Rx,Ry)
                W = W * mask[:, :, None, :, :]
                norm2 = np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)
                W = np.where(norm2 > 1e-15, W / (np.sqrt(norm2) + 1e-15), W)
            else:
                denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
                W = W / denom
            return W

        self.W_A_plus  = make_W(self.Pplus,  tauA, phase)
        self.W_B_plus  = make_W(self.Pplus,  tauB, phase)
        self.W_A_minus = make_W(self.Pminus, tauA, phase)
        self.W_B_minus = make_W(self.Pminus, tauB, phase)

        def make_V(W):
            return np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True)

        self.V_A_minus = make_V(self.W_A_minus)
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus   = self.V_A_minus + self.V_B_minus

        self.V_A_plus  = make_V(self.W_A_plus)
        self.V_B_plus  = make_V(self.W_B_plus)
        self.V_plus    = self.V_A_plus + self.V_B_plus

    def G_CI(self, alpha=1.0, k_is_centered=False, norm='backward'):
        '''Return the CI lower-band covariance in block form.'''

        Nx, Ny = self.Nx, self.Ny
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        nx = np.sin(KX)
        ny = np.sin(KY)
        nz = float(alpha) - np.cos(KX) - np.cos(KY)

        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        n_mag = np.where(n_mag == 0, 1e-15, n_mag)

        def _k_to_r_rel(nk, k_centered=False, fft_norm='backward'):
            arr = np.fft.ifftshift(nk) if k_centered else nk
            nR = np.fft.ifft2(arr, norm=fft_norm)
            nR = np.real_if_close(nR, tol=1e3)

            x = np.arange(Nx)
            y = np.arange(Ny)
            dX = (x[:, None, None, None] - x[None, None, :, None]) % Nx
            dY = (y[None, :, None, None] - y[None, None, None, :]) % Ny
            return nR[dX, dY]

        nx_real = _k_to_r_rel(nx / n_mag, k_centered=k_is_centered, fft_norm=norm)
        ny_real = _k_to_r_rel(ny / n_mag, k_centered=k_is_centered, fft_norm=norm)
        nz_real = _k_to_r_rel(nz / n_mag, k_centered=k_is_centered, fft_norm=norm)

        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = 1j * np.array([[0, -1], [1, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        h_real = (nx_real[..., None, None] * sx +
                  ny_real[..., None, None] * sy +
                  nz_real[..., None, None] * sz)
        h_real = np.moveaxis(h_real, 4, 2)

        eye_block = np.eye(self.Ntot, dtype=np.complex128).reshape(Nx, Ny, 2, Nx, Ny, 2, order='C')
        Pminus = 0.5 * (eye_block - h_real)
        return np.ascontiguousarray(Pminus.conj())

    def _ensure_OW_ready(self):
        '''Lazily construct Wannier data required by evolution operators.'''
        needed = ("Pminus","Pplus","W_A_plus","W_B_plus","W_A_minus","W_B_minus",
                  "V_A_plus","V_B_plus","V_A_minus","V_B_minus","V_plus","V_minus")
        for attr in needed:
            if not hasattr(self, attr):
                self.construct_OW_functions()
                break

    @contextmanager
    def _tqdm_progress(self, total, enabled=True, **kwargs):
        if not enabled:
            yield None
            return
        with tqdm(total=total, **kwargs) as pbar:
            yield pbar

    # --------------------------- Evolution (generation) -----------------------
    def G_evolution(self,
                    T_cycles=20,
                    G_init=None,
                    init_kind="default",
                    DW=True,
                    decoh=False,
                    progress=True,
                    store_history=True,
                    track_all_time_steps=False,
                    n_a=None):
        '''Evolve G via RK4 and return the sampled snapshots/steps in memory.'''

        self.steps_per_cycle = max(1, int(round(1.0 / self.dt)))
        self.max_steps = max(1, int(round(float(T_cycles) / self.dt)))

        dt = self.dt
        max_steps = self.max_steps
        steps_per_cycle = self.steps_per_cycle
        if n_a is None:
            n_a = self.n_a
        self._decoh_last = bool(decoh)
        if decoh:
            print("[info] Non-Gaussian decoherence channel enabled in Lindbladian evolution.")

        # Ensure OW data exist (used only by evolution)
        self._ensure_OW_ready()

        # Initialize G
        Nx, Ny = self.Nx, self.Ny
        expected = (Nx, Ny, 2, Nx, Ny, 2)
        if G_init is not None:
            G = np.asarray(G_init, dtype=np.complex128)
            if G.shape != expected:
                raise ValueError(f"G_init must have shape {expected}, got {G.shape}.")
        else:
            if str(init_kind).lower() == "maxmix":
                Ntot = Nx * Ny * 2
                G = (0.5 * np.eye(Ntot, dtype=np.complex128)).reshape(expected, order='C')
            else:
                G = np.zeros(expected, dtype=np.complex128)

        snapshots = [G.copy()]
        saved_steps = [0]
        record_all = bool(store_history) and bool(track_all_time_steps)

        # Evolution loop
        tmp = {}
        t0 = time.time()
        with self._tqdm_progress(
            total=int(max_steps),
            enabled=progress,
            desc="evolving",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
        ) as pbar:
            for k in range(1, int(max_steps) + 1):
                G, tmp = self.rk4_Lindblad_evolver(G, dt, n_a=n_a, decoh=decoh, tmp=tmp)
                if record_all:
                    snapshots.append(G.copy())
                    saved_steps.append(k)
                elif (k % steps_per_cycle) == 0:
                    snapshots.append(G.copy())
                    saved_steps.append(k)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        "dt": f"{dt:.2e}",
                        "saved": len(snapshots),
                        "elapsed": f"{(time.time()-t0):.2f}s",
                    })

        # Ensure final step included
        if saved_steps[-1] != max_steps:
            snapshots.append(G.copy())
            saved_steps.append(max_steps)

        if store_history:
            self.G_history = [snap.copy() for snap in snapshots]
            self.history_steps = list(saved_steps)
        else:
            self.G_history = None
            self.history_steps = None

        return snapshots, saved_steps

    def chirality_test_quench(self,
                              T_bg_cycles=20,
                              T_probe_cycles=10,
                              x0=None,
                              y0=None,
                              epsilon=1e-3,
                              orbital=None,
                              init_kind="default",
                              G_init=None,
                              n_a=None,
                              decoh=False,
                              progress=True,
                              store_snapshots=False,
                              sum_orbitals=True):
        '''Run a pulse-probe quench and return delta-density snapshots.'''

        if n_a is None:
            n_a = self.n_a
        self._decoh_last = bool(decoh)

        # Prepare the steady mixed state background.
        bg_snaps, _bg_steps = self.G_evolution(
            T_cycles=T_bg_cycles,
            G_init=G_init,
            init_kind=init_kind,
            DW=True,
            decoh=decoh,
            progress=progress,
            store_history=False,
            n_a=n_a
        )
        G_bg = self._require_block(bg_snaps[-1])

        # Choose a default injection point on the domain wall.
        Nx, Ny = self.Nx, self.Ny
        if x0 is None:
            if self.DW_loc is not None:
                x0 = int(round(0.5 * (self.DW_loc[0] + self.DW_loc[1])))
            else:
                x0 = Nx // 2
        if y0 is None:
            y0 = Ny // 2
        if not (0 <= x0 < Nx and 0 <= y0 < Ny):
            raise ValueError(f"(x0, y0)=({x0}, {y0}) must lie within the lattice.")

        # Build the pulse state by adding a local density deviation.
        G_pulse = G_bg.copy()
        if orbital is None:
            # Split epsilon across both orbitals to keep total injected density epsilon.
            for mu in (0, 1):
                G_pulse[x0, y0, mu, x0, y0, mu] += 0.5 * epsilon
        else:
            mu = int(orbital)
            if mu not in (0, 1):
                raise ValueError("orbital must be 0, 1, or None.")
            G_pulse[x0, y0, mu, x0, y0, mu] += epsilon

        # Evolve both states forward in lockstep and record delta-density.
        steps_per_cycle = max(1, int(round(1.0 / self.dt)))
        max_steps = max(1, int(round(float(T_probe_cycles) / self.dt)))
        dt = self.dt

        idx_x = np.arange(Nx)[:, None, None]
        idx_y = np.arange(Ny)[None, :, None]
        idx_mu = np.arange(2)[None, None, :]

        delta_hist = []
        saved_steps = []
        if store_snapshots:
            G_bg_hist = []
            G_pulse_hist = []
        else:
            G_bg_hist = None
            G_pulse_hist = None

        def _save_snapshot(step, G_bg_snap, G_pulse_snap):
            delta = G_pulse_snap - G_bg_snap
            diag = delta[idx_x, idx_y, idx_mu, idx_x, idx_y, idx_mu]
            diag = np.real_if_close(diag)
            if sum_orbitals:
                diag = diag.sum(axis=2)
            delta_hist.append(diag.copy())
            saved_steps.append(step)
            if store_snapshots:
                G_bg_hist.append(G_bg_snap.copy())
                G_pulse_hist.append(G_pulse_snap.copy())

        _save_snapshot(0, G_bg, G_pulse)

        tmp_bg = {}
        tmp_pulse = {}
        iterator = range(1, int(max_steps) + 1)
        pbar = tqdm(iterator, desc="probe", unit="step", disable=not progress)
        t0 = time.time()
        for k in pbar:
            G_bg, tmp_bg = self.rk4_Lindblad_evolver(G_bg, dt, n_a=n_a, decoh=decoh, tmp=tmp_bg)
            G_pulse, tmp_pulse = self.rk4_Lindblad_evolver(G_pulse, dt, n_a=n_a, decoh=decoh, tmp=tmp_pulse)
            if (k % steps_per_cycle) == 0:
                _save_snapshot(k, G_bg, G_pulse)
            if progress:
                pbar.set_postfix({
                    "dt": f"{dt:.2e}",
                    "saved": len(delta_hist),
                    "elapsed": f"{(time.time()-t0):.2f}s",
                })
        if hasattr(pbar, "close"):
            pbar.close()

        if saved_steps[-1] != max_steps:
            _save_snapshot(max_steps, G_bg, G_pulse)

        times = np.asarray(saved_steps, dtype=float) * dt
        return {
            "G_bg": G_bg,
            "G_pulse": G_pulse,
            "delta_n": np.stack(delta_hist, axis=0),
            "steps": np.asarray(saved_steps, dtype=int),
            "times": times,
            "G_bg_history": None if not store_snapshots else np.stack(G_bg_hist, axis=0),
            "G_pulse_history": None if not store_snapshots else np.stack(G_pulse_hist, axis=0),
            "pulse_site": (int(x0), int(y0)),
            "pulse_orbital": None if orbital is None else int(orbital),
        }

    # ----------------------------- Lindbladian core ---------------------------
    def Lgain(self, G, n_a):
        '''Gain contribution of the Lindbladian superoperator.'''
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_minus, optimize=True) +
                    np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_minus, G, optimize=True))
        return n_a*(self.V_minus + Y)

    def Lloss(self, G, n_a):
        '''Loss contribution of the Lindbladian superoperator.'''
        Y = -((1-n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_plus, optimize=True) +
                          np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_plus, G, optimize=True))
        return Y

    def double_comm(self, G, W, V):
        '''Evaluate the double-commutator term used in decoherence.'''
        VG = np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True)
        GV = np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True)
        s_ab = np.einsum('ijkab, ijkpqr, pqrab -> ab', W.conj(), G, W, optimize=True)
        nonlinear = np.einsum('ijkab, ab, pqrab -> ijkpqr', W, s_ab, W.conj(), optimize=True)
        return (VG + GV) - 2.0 * nonlinear

    def Ldecoh(self, G, n_a):
        '''Decoherence channel formed from the domain-wall projectors.'''
        upper = self.double_comm(G, self.W_A_plus,  self.V_A_plus)  + self.double_comm(G, self.W_B_plus,  self.V_B_plus)
        lower = self.double_comm(G, self.W_A_minus, self.V_A_minus) + self.double_comm(G, self.W_B_minus, self.V_B_minus)
        return -(1/2)*((2-n_a)*lower + (1+n_a)*upper)

    def Lcycle(self, G, n_a=None, decoh=True):
        '''Aggregate gain, loss, and optional decoherence contributions.'''
        if n_a is None:
            n_a = self.n_a
        if decoh:
            return self.Lgain(G, n_a) + self.Lloss(G, n_a) + self.Ldecoh(G, n_a)
        else:
            return self.Lgain(G, n_a) + self.Lloss(G, n_a)

    def vectorized_single_particle_superoperator(self, G=None, n_a=None, decoh=True):
        '''
        Return the dense column-stacked single-particle Lindbladian.

        The convention is
            d/dt vec(G) = source + generator @ vec(G),
        where ``vec`` stacks dense matrices column-by-column.

        The returned ``generator`` is square with shape ``(Ntot**2, Ntot**2)``.
        To remove the affine source term, the returned ``lifted_generator`` embeds
        the dynamics into a homogeneous square system of shape
        ``(Ntot**2 + 1, Ntot**2 + 1)``.

        Parameters
        ----------
        G : array-like or None
            Optional snapshot. If provided, the returned dictionary also contains
            ``vec_G`` and the action ``source + generator @ vec_G``.
        n_a : float or None
            Ancilla filling. Defaults to ``self.n_a``.
        decoh : bool
            Include or omit the decoherence contribution.

        Returns
        -------
        dict
            Keys are ``generator``, ``source``, ``action``, ``vec_G``,
            ``lifted_generator``, ``lifted_vec_G``, ``lifted_action``,
            ``gain_generator``, ``loss_generator``, and ``decoh_generator``.
        '''
        if n_a is None:
            n_a = self.n_a
        self._ensure_OW_ready()

        N = self.Ntot
        eye = np.eye(N, dtype=np.complex128)
        V_minus = self._block_to_dense(self.V_minus)
        V_plus = self._block_to_dense(self.V_plus)

        gain_generator = -0.5 * float(n_a) * (
            np.kron(eye, V_minus) + np.kron(V_minus.T, eye)
        )
        loss_generator = -0.5 * (1.0 - float(n_a)) * (
            np.kron(eye, V_plus) + np.kron(V_plus.T, eye)
        )
        decoh_generator = np.zeros((N * N, N * N), dtype=np.complex128)

        if decoh:
            projector_sets = (
                (2.0 - float(n_a), self.W_A_minus),
                (2.0 - float(n_a), self.W_B_minus),
                (1.0 + float(n_a), self.W_A_plus),
                (1.0 + float(n_a), self.W_B_plus),
            )
            for coeff, Wset in projector_sets:
                for Rx in range(self.Nx):
                    for Ry in range(self.Ny):
                        w = np.asarray(Wset[:, :, :, Rx, Ry], dtype=np.complex128).reshape(N, order='C')
                        P = np.outer(w, w.conj())
                        p_vec = P.reshape(N * N, order='F')
                        p_t_vec = P.T.reshape(N * N, order='F')
                        decoh_generator += -0.5 * coeff * (
                            np.kron(eye, P)
                            + np.kron(P.T, eye)
                            - 2.0 * np.outer(p_vec, p_t_vec)
                        )

        generator = gain_generator + loss_generator + decoh_generator
        source = float(n_a) * self._matrix_to_col_vec(V_minus)
        lifted_generator = self._lift_affine_generator(generator, source)

        vec_G = None
        action = None
        lifted_vec_G = None
        lifted_action = None
        if G is not None:
            dense_G = self._coerce_dense_snapshot(G)
            vec_G = self._matrix_to_col_vec(dense_G)
            action = source + generator @ vec_G
            lifted_vec_G = np.empty(self.Ntot * self.Ntot + 1, dtype=np.complex128)
            lifted_vec_G[:-1] = vec_G
            lifted_vec_G[-1] = 1.0
            lifted_action = lifted_generator @ lifted_vec_G

        return {
            "generator": generator,
            "source": source,
            "action": action,
            "vec_G": vec_G,
            "lifted_generator": lifted_generator,
            "lifted_vec_G": lifted_vec_G,
            "lifted_action": lifted_action,
            "gain_generator": gain_generator,
            "loss_generator": loss_generator,
            "decoh_generator": decoh_generator,
        }

    def partial_fourier_transformed_superoperator(self, n_a=None, decoh=True, norm='ortho', include_lifted=True):
        '''
        Construct the full y-partially Fourier transformed vectorized single-particle superoperator.

        If ``U_y`` is the single-particle unitary that Fourier transforms the y direction,
        this method returns the transformed affine system
            d/dt vec(G_k) = source_k + generator_k @ vec(G_k),
        with
            vec(G_k) = (U_y^T kron U_y^dagger) vec(G).

        For ``decoh=False`` the exact diagonal-ky blocks should be built with
        ``partial_fourier_ky_superoperator_no_decoh``. For ``decoh=True`` the exact
        conserved sectors are labeled by the momentum transfer q = ky_row - ky_col,
        and should be built with ``partial_fourier_q_sector_superoperator``.
        '''
        info = self.vectorized_single_particle_superoperator(n_a=n_a, decoh=decoh)
        U_y = self._y_partial_fourier_unitary(norm=norm)
        vec_transform = np.kron(U_y.T, U_y.conj().T)

        generator_k = vec_transform @ info["generator"] @ vec_transform.conj().T
        source_k = vec_transform @ info["source"]

        result = {
            "generator": generator_k,
            "source": source_k,
            "single_particle_unitary": U_y,
            "vec_transform": vec_transform,
        }

        if include_lifted:
            result["lifted_generator"] = self._lift_affine_generator(generator_k, source_k)

        return result

    def partial_fourier_ky_superoperator_no_decoh(self, n_a=None, norm='ortho', include_lifted=True):
        '''
        Construct the exact ky-resolved superoperator for the decoherence-free dynamics.

        This is the exact restriction of the y-partially Fourier transformed generator to
        the translation-invariant q=0 subspace when ``decoh=False``. Each ky block acts on
        a vectorized (2*Nx)x(2*Nx) correlator.
        '''
        if n_a is None:
            n_a = self.n_a
        self._ensure_OW_ready()

        V_minus_dense = self._block_to_dense(self.V_minus)
        V_plus_dense = self._block_to_dense(self.V_plus)
        V_minus_k, V_minus_blocks = self._partial_fourier_single_particle_blocks(V_minus_dense, norm=norm)
        V_plus_k, V_plus_blocks = self._partial_fourier_single_particle_blocks(V_plus_dense, norm=norm)

        ky_vals = 2 * np.pi * np.fft.fftfreq(self.Ny, d=1.0)
        dim_block = 2 * self.Nx
        eye_block = np.eye(dim_block, dtype=np.complex128)

        ky_block_generators = []
        ky_block_sources = []
        ky_block_lifted_generators = []
        for Vm, Vp in zip(V_minus_blocks, V_plus_blocks):
            generator = -0.5 * float(n_a) * (
                np.kron(eye_block, Vm) + np.kron(Vm.T, eye_block)
            ) - 0.5 * (1.0 - float(n_a)) * (
                np.kron(eye_block, Vp) + np.kron(Vp.T, eye_block)
            )
            source = float(n_a) * Vm.reshape(dim_block * dim_block, order='F')
            ky_block_generators.append(generator)
            ky_block_sources.append(source)
            if include_lifted:
                ky_block_lifted_generators.append(self._lift_affine_generator(generator, source))

        result = {
            "ky_vals": ky_vals,
            "single_particle_V_minus": V_minus_k,
            "single_particle_V_plus": V_plus_k,
            "single_particle_V_minus_blocks": V_minus_blocks,
            "single_particle_V_plus_blocks": V_plus_blocks,
            "ky_block_generators": ky_block_generators,
            "ky_block_sources": ky_block_sources,
        }
        if include_lifted:
            result["ky_block_lifted_generators"] = ky_block_lifted_generators
        return result

    def partial_fourier_q_sector_superoperator(self, n_a=None, decoh=True, norm='ortho', include_lifted=True):
        '''
        Construct the exact momentum-transfer sector decomposition q = ky_row - ky_col.

        For ``decoh=True`` this is the exact block decomposition of the y-partially
        Fourier transformed vectorized superoperator. The q=0 sector contains the
        affine source; q != 0 sectors are homogeneous.
        '''
        transformed = self.partial_fourier_transformed_superoperator(
            n_a=n_a,
            decoh=decoh,
            norm=norm,
            include_lifted=False,
        )
        q_vals = 2 * np.pi * np.fft.fftfreq(self.Ny, d=1.0)

        q_sector_generators = []
        q_sector_sources = []
        q_sector_vec_indices = []
        q_sector_lifted_generators = []
        for q_idx in range(self.Ny):
            vec_idx = self._q_sector_vec_indices(q_idx)
            generator = transformed["generator"][np.ix_(vec_idx, vec_idx)].copy()
            source = transformed["source"][vec_idx].copy()
            q_sector_generators.append(generator)
            q_sector_sources.append(source)
            q_sector_vec_indices.append(vec_idx)
            if include_lifted:
                q_sector_lifted_generators.append(self._lift_affine_generator(generator, source))

        result = {
            "q_vals": q_vals,
            "generator": transformed["generator"],
            "source": transformed["source"],
            "single_particle_unitary": transformed["single_particle_unitary"],
            "vec_transform": transformed["vec_transform"],
            "q_sector_generators": q_sector_generators,
            "q_sector_sources": q_sector_sources,
            "q_sector_vec_indices": q_sector_vec_indices,
        }
        if include_lifted:
            result["q_sector_lifted_generators"] = q_sector_lifted_generators
            result["lifted_generator"] = self._lift_affine_generator(
                transformed["generator"],
                transformed["source"],
            )
        return result

    def rk4_Lindblad_evolver(self, G, dt, n_a=None, decoh=True, tmp=None):
        '''Advance a single RK4 step for the Lindblad equation.'''
        if tmp is None: tmp = {}
        if n_a is None:
            n_a = self.n_a
        k1 = tmp.get('k1'); k2 = tmp.get('k2'); k3 = tmp.get('k3'); k4 = tmp.get('k4'); Y = tmp.get('Y')
        if k1 is None: k1 = tmp['k1'] = np.empty_like(G)
        if k2 is None: k2 = tmp['k2'] = np.empty_like(G)
        if k3 is None: k3 = tmp['k3'] = np.empty_like(G)
        if k4 is None: k4 = tmp['k4'] = np.empty_like(G)
        if Y  is None: Y  = tmp['Y']  = np.empty_like(G)

        k1[:] = self.Lcycle(G, n_a, decoh=decoh)
        np.multiply(k1, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k2[:] = self.Lcycle(Y, n_a, decoh=decoh)
        np.multiply(k2, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k3[:] = self.Lcycle(Y, n_a, decoh=decoh)
        np.multiply(k3, dt, out=Y); np.add(G, Y, out=Y)
        k4[:] = self.Lcycle(Y, n_a, decoh=decoh)

        np.add(k1, k4, out=Y)
        np.add(Y, 2.0*k2, out=Y)
        np.add(Y, 2.0*k3, out=Y)
        G += (dt/6.0) * Y
        return G, tmp

    # ---------------------------- Plotting (read-only) ------------------------

    def plot_spectrum_vs_time(self,
                               G_data,
                               steps=None,
                               times=None,
                               filename=None,
                               cmap='tab10'):
        '''Plot eigenvalues of G at requested physical times using supplied snapshots.'''
        if G_data is None:
            raise ValueError("G_data must be provided as an iterable of snapshots.")

        if steps is None:
            if self.G_history is not None and G_data is self.G_history and self.history_steps is not None:
                steps = self.history_steps
            else:
                raise ValueError("steps must be provided when metadata is unavailable.")

        snapshots = self._require_block_history(G_data)
        steps = np.asarray(steps, dtype=int)
        if steps.size != len(snapshots):
            raise ValueError("Length of steps must match the number of snapshots.")
        if steps.size == 0:
            raise ValueError("No history data provided for plotting.")

        dt = self.dt
        Tmax = steps[-1] * dt
        if times is None:
            times = steps * dt
        times = np.asarray(times, dtype=float)
        if np.any(times < 0) or np.any(times > Tmax + 1e-12):
            raise ValueError(f"Requested times must lie in [0, {Tmax:g}] for the provided steps.")

        target_steps = np.rint(times / dt).astype(int)
        k_map = np.empty_like(target_steps)
        for i, kt in enumerate(target_steps):
            idx = int(np.argmin(np.abs(steps - kt)))
            k_map[i] = steps[idx]

        step_to_G = {int(step): snap for step, snap in zip(steps, snapshots)}
        Nx, Ny = self.Nx, self.Ny
        Ntot = self.Ntot

        outdir = self._ensure_outdir('figs/spectrum_vs_time')
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(times)))

        for idx, (t, k) in enumerate(zip(times, k_map)):
            Gk = step_to_G[int(k)]
            vals = np.linalg.eigvalsh(self._block_to_dense(Gk))
            ax.plot(vals.real, linestyle='None',
                    marker='o', markersize=3,
                    markerfacecolor=colors[idx], markeredgecolor='none',
                    label=f"t≈{t:g} (step {int(k)})")

        ax.set_ylabel(r"eigvals($G$)")
        ax.set_title(f"Spectrum of $G$ vs time (N={Nx}×{Ny}, dt={dt:g})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()

        if filename is None:
            tdesc = "-".join(f"{tt:g}" for tt in times)
            decoh_tag = "_decoh_on" if self._get_decoh_flag() else "_decoh_off"
            T_total = steps[-1] * dt
            filename = f"spectrum_vs_time_real_N{self.Nx}_dt{dt:g}_t_{tdesc}_T{T_total:g}{decoh_tag}.pdf"
        fullpath = os.path.join(outdir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath

    def _smart_x_positions(self):
        '''Return representative x-positions guided by the domain-wall geometry.'''
        Nx = self.Nx
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
                xx = int(x)
                if xx not in seen:
                    uniq.append(xx); seen.add(xx)
            return uniq
        else:
            xs = np.linspace(0, Nx - 1, 9, dtype=int)
            return xs

    def plot_corr_y_profiles(self, G,
                              x_positions=None,
                              ry_max=None,
                              filename=None,
                              save=True,
                              curve_labels=False):
        '''Plot squared two-point correlators along y for selected x positions.'''
        if G is None:
            raise ValueError("G must be provided.")

        G_blocks = self._require_block(G)
        Nx, Ny = self.Nx, self.Ny

        # r_y range
        if ry_max is None:
            ry_max = Ny//2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # x positions
        if x_positions is None:
            if self.DW_loc is not None:
                x_positions = self._smart_x_positions()
            else:
                x_positions = np.arange(0, max(1, Nx//2), 2, dtype=int)
                if x_positions.size == 0:
                    x_positions = np.array([0], dtype=int)

        x_list = [(int(x), f"{int(x)}") for x in x_positions]

        outdir = self._ensure_outdir('figs/corr_y_profiles')
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for x0, lbl in x_list:
            C_vec = self.squared_two_point_corr_xslice(G_blocks, x0=int(x0), ry=ry_vals).real
            line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=rf"$x_0={lbl}$")

            # Inline label at right edge
            if curve_labels:
                finite = np.isfinite(C_vec)
                y_right = C_vec[finite][-1] if np.any(finite) else C_vec[-1]
                x_right = ry_vals[-1] * 1.02 if ry_vals[-1] > 0 else ry_vals[-1] + 0.5
                ax.annotate(lbl, xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                        textcoords='data', ha='left', va='center', fontsize=9,
                        color=line.get_color())

        ax.set_xlabel(r"$r_y$")
        ax.set_ylabel(r"$C_G(x_0; r_y)$")
        ax.set_title(f"Squared correlator vs $r_y$ at fixed $x_0$ (N={Nx}, decoh={'on' if self._get_decoh_flag() else 'off'})")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        if self.DW_loc is not None:
            ax.text(0.02, 0.96, fr"DWs at $x_0 = {int(self.DW_loc[0])}, \ {int(self.DW_loc[1])}$",
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", alpha=0.6))
        fig.tight_layout()

        if filename is None:
            xdesc = "-".join(lbl for _, lbl in x_list)
            decoh_tag = "_decoh_on" if self._get_decoh_flag() else ""
            filename = f"corr2_y_profiles_N{Nx}_xs_{xdesc}{decoh_tag}.pdf"
        fullpath = os.path.join(outdir, filename)
        if save:
            fig.savefig(fullpath, bbox_inches='tight')
        return ax

    def chern_marker_dynamics(self, G_data,
                              steps=None,
                              fps=12,
                              cmap='RdBu_r',
                              outbasename=None,
                              vmin=-1.0,
                              vmax=1.0):
        '''Animate the local Chern marker across supplied snapshots and save outputs.'''
        if G_data is None:
            raise ValueError("G_data must be provided as an iterable of snapshots.")

        if steps is None:
            if self.G_history is not None and G_data is self.G_history and self.history_steps is not None:
                steps = self.history_steps
            else:
                raise ValueError("steps must be provided when metadata is unavailable.")

        snapshots = self._require_block_history(G_data)
        steps = np.asarray(steps, dtype=int)
        if steps.size != len(snapshots):
            raise ValueError("Length of steps must match the number of snapshots.")
        if steps.size == 0:
            raise ValueError("No history data available for animation.")

        Nx, Ny = self.Nx, self.Ny
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            decoh_tag = 'decoh_on' if self._get_decoh_flag() else 'decoh_off'
            T_total = steps[-1] * self.dt if steps else 0
            outbasename = f"chern_marker_dynamics_N{self.Nx}_T{T_total:g}_{decoh_tag}"
        gif_path = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.png")

        fig = plt.figure(figsize=(3.6, 4.0))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.zeros((Nx, Ny)), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color('black')
        ax.set_xlabel("y"); ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        writer = animation.PillowWriter(fps=fps)

        def _C_map(G_block):
            return self.local_chern_marker(G_block)

        with writer.saving(fig, gif_path, dpi=120):
            for t, Gt in zip(steps, snapshots):
                im.set_data(_C_map(Gt))
                ax.set_title(f"Local Chern marker (step={t}, decoh={'on' if self._get_decoh_flag() else 'off'})")
                writer.grab_frame()
        plt.close(fig)

        # Save final frame
        C_last = self.local_chern_marker(snapshots[-1])
        fig2, ax2 = plt.subplots(figsize=(3.6, 4.0))
        im2 = ax2.imshow(C_last, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xlabel("y"); ax2.set_ylabel("x")
        ax2.set_title("Local Chern marker — final frame (decoh " + ('on' if self._get_decoh_flag() else 'off') + ")")
        fig2.savefig(final_path, bbox_inches='tight', dpi=140)
        plt.close(fig2)

        return gif_path, final_path, C_last, snapshots[-1].copy()

    def entanglement_contour_suite(self, G_data,
                                   steps=None,
                                   x_positions=None,
                                   filename_profiles=None,
                                   filename_prefix_dyn=None,
                                   fps=10,
                                   save=True):
        '''Generate summary plots and animations for the entanglement contour history.'''
        if G_data is None:
            raise ValueError("G_data must be provided as an iterable of snapshots.")

        if steps is None:
            if self.G_history is not None and G_data is self.G_history and self.history_steps is not None:
                steps = self.history_steps
            else:
                raise ValueError("steps must be provided when metadata is unavailable.")

        snapshots = self._require_block_history(G_data)
        steps = np.asarray(steps, dtype=int)
        if steps.size != len(snapshots):
            raise ValueError("Length of steps must match the number of snapshots.")
        if steps.size == 0:
            raise ValueError("No history data available for the entanglement contour suite.")

        Nx, Ny = int(self.Nx), int(self.Ny)
        T = len(snapshots)

        if x_positions is None:
            x_positions = self._smart_x_positions()
        try:
            x_list = [(int(x), str(lbl)) for (x, lbl) in x_positions]
        except Exception:
            x_list = [(int(x), f"{int(x)}") for x in x_positions]

        ry_sum_profiles = {x0: np.zeros(T, dtype=float) for (x0, _) in x_list}
        ent_maps = []
        for idx, Gt in enumerate(snapshots):
            s_map = self.entanglement_contour(Gt)
            ent_maps.append(s_map)
            for (x0, _) in x_list:
                ry_sum_profiles[x0][idx] = float(np.sum(s_map[int(x0) % Nx, :]))

        G_final = snapshots[-1]
        ky_final, evals_ky = self._ky_spectrum_evals(G_final)
        s_final = ent_maps[-1]

        outdir = self._ensure_outdir("figs/entanglement_contour")
        if filename_profiles is None:
            xdesc = "-".join(lbl for _, lbl in x_list)
            decoh_tag = "_decoh_on" if self._get_decoh_flag() else ""
            T_total = steps[-1] * self.dt if steps.size else 0
            filename_profiles = f"entanglement_suite_N{self.Nx}_xs_{xdesc}_T{T}_T{T_total:g}{decoh_tag}.pdf"
        profiles_pdf = os.path.join(outdir, filename_profiles)

        fig = plt.figure(constrained_layout=True, figsize=(12.2, 11.6))
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.1, 1.0, 1.2])

        axP = fig.add_subplot(gs[0, :])
        t_vals = np.arange(T, dtype=int)
        for (x0, lbl) in x_list:
            axP.plot(t_vals, ry_sum_profiles[x0], marker='o', ms=3, lw=1, label=lbl)
        axP.set_xlabel(r"$T_{\mathrm{cycle}}$")
        axP.set_ylabel(r"$\sum_y\, s(x_0,y)$")
        axP.set_title("Entanglement-contour profiles vs time")
        axP.set_yscale("log")
        axP.grid(True, alpha=0.3)
        axP.legend(fontsize=8, ncol=3)
        if getattr(self, "DW_loc", None) is not None:
            try:
                dw_values = [int(val) for val in self.DW_loc]
            except Exception:
                dw_values = list(self.DW_loc)
            dw_desc = ", ".join(str(val) for val in dw_values)
            axP.text(
                0.02,
                0.04,
                rf"DWs at $x_0={dw_desc}$",
                transform=axP.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", alpha=0.6),
            )

        axE = fig.add_subplot(gs[1, :])
        for band in evals_ky.T:
            axE.plot(ky_final, band, '.', ms=2.5, lw=0)
        axE.set_title(r"spec($G(k_y)$) (final)")
        axE.set_xlabel(r"$k_y$")
        axE.set_ylabel("eigenvalue")
        axE.grid(True, alpha=0.3)

        axM1 = fig.add_subplot(gs[2, 0])
        im1 = axM1.imshow(s_final, cmap="Blues", origin="upper", aspect="equal")
        axM1.set_title("Final entanglement contour $s(x,y)$")
        axM1.set_xlabel("y")
        axM1.set_ylabel("x")
        axM1.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        axM1.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        axM1.grid(which="minor", color="k", linestyle=":", linewidth=0.4, alpha=0.35)
        axM1.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)

        axM2 = fig.add_subplot(gs[2, 1])
        C_tanh = self.local_chern_marker(G_final)
        im2 = axM2.imshow(C_tanh, cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="upper", aspect="equal")
        axM2.set_title(r"$\tanh\mathcal{C}(x,y)$ (final)")
        axM2.set_xlabel("y")
        axM2.set_ylabel("x")
        axM2.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        axM2.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        axM2.grid(which="minor", color="k", linestyle=":", linewidth=0.4, alpha=0.35)
        axM2.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04)

        decoh_tag = "on" if self._get_decoh_flag() else "off"
        spc = self.steps_per_cycle if self.steps_per_cycle is not None else 'NA'
        fig.suptitle(
            f"Entanglement contour suite — N={Nx}×{Ny}, snapshots={T}, spc={spc}, decoh={decoh_tag}",
            y=1.02
        )
        if save:
            fig.savefig(profiles_pdf, bbox_inches="tight", dpi=140)

        outdir_dyn = self._ensure_outdir("figs/entanglement_contour_dynamics")
        if filename_prefix_dyn is None:
            decoh_tag = "decoh_on" if self._get_decoh_flag() else "decoh_off"
            T_total = steps[-1] * self.dt if steps.size else 0
            filename_prefix_dyn = f"entanglement_dyn_N{self.Nx}_T{T_total:g}_{decoh_tag}"

        dyn_gif = os.path.join(outdir_dyn, f"{filename_prefix_dyn}.gif")
        final_png = os.path.join(outdir_dyn, f"{filename_prefix_dyn}_final.png")

        figG = plt.figure(constrained_layout=True, figsize=(5.6, 4.6))
        axG = figG.add_subplot(111)
        vmax_all = 0.0
        for s_map in ent_maps:
            vmax_all = max(vmax_all, float(np.max(s_map)))
        base_map = ent_maps[0]
        imG = axG.imshow(base_map, cmap="Blues", origin="upper", aspect="equal",
                         vmin=0.0, vmax=vmax_all if vmax_all > 0 else None)
        figG.colorbar(imG, ax=axG, fraction=0.046, pad=0.04)
        axG.set_xlabel("y")
        axG.set_ylabel("x")
        axG.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        axG.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        axG.grid(which="minor", color="k", linestyle=":", linewidth=0.4, alpha=0.35)
        axG.tick_params(which="minor", bottom=False, left=False)
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(figG, dyn_gif, dpi=120):
            for t, s_map in zip(steps, ent_maps):
                imG.set_data(s_map)
                axG.set_title(f"$s(x,y)$ — step {t}")
                writer.grab_frame()
        plt.close(figG)

        figF, axF = plt.subplots(figsize=(5.6, 4.6))
        imF = axF.imshow(s_final, cmap="Blues", origin="upper", aspect="equal")
        figF.colorbar(imF, ax=axF, fraction=0.046, pad=0.04)
        axF.set_xlabel("y")
        axF.set_ylabel("x")
        axF.set_title("Final $s(x,y)$")
        axF.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        axF.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        axF.grid(which="minor", color="k", linestyle=":", linewidth=0.4, alpha=0.35)
        axF.tick_params(which="minor", bottom=False, left=False)
        if save:
            figF.savefig(final_png, bbox_inches="tight", dpi=140)

        return {
            "profiles_pdf": profiles_pdf,
            "dyn_dir": outdir_dyn,
            "dyn_gif": dyn_gif,
            "final_png": final_png,
        }

    # ------------------------------ Analysis utils ----------------------------

    def local_chern_marker(self, G, mask_outside=False):
        '''Return the tanh-biased Bianco–Resta local Chern marker for snapshot ``G``.'''
        if G is None:
            raise ValueError("G must be provided for the local Chern marker.")

        G_block = self._require_block(G)
        P = G_block.conj()
        Nx, Ny = self.Nx, self.Ny

        X = np.arange(1, Nx + 1, dtype=float)
        Y = np.arange(1, Ny + 1, dtype=float)
        Xgrid, Ygrid = np.meshgrid(X, Y, indexing='ij')

        def right_X(A):
            return A * Xgrid[None, None, None, :, :, None]

        def right_Y(A):
            return A * Ygrid[None, None, None, :, :, None]

        def mm(A, B):
            return np.einsum('ijslmn,lmnopr->ijsopr', A, B, optimize=True)

        T = right_X(P)
        T = mm(T, P)
        T = right_Y(T)
        T = mm(T, P)

        U = right_Y(P)
        U = mm(U, P)
        U = right_X(U)
        U = mm(U, P)

        M = (2.0 * np.pi * 1j) * (T - U)

        ix = np.arange(Nx)[:, None, None]
        iy = np.arange(Ny)[None, :, None]
        ispin = np.arange(2)[None, None, :]
        diag_vals = M[ix, iy, ispin, ix, iy, ispin]
        C = np.tanh(np.real_if_close(diag_vals.sum(axis=2), tol=1e-9))

        if mask_outside and hasattr(self, "inside_mask") and self.inside_mask is not None:
            C = np.where(self.inside_mask, C, 0.0)
        return C

    def current_maps_gauge_invariant(self, G):
        '''Gauge-invariant currents flowing along +x and +y bonds for snapshot G.'''
        if G is None:
            raise ValueError("G must be provided for the current maps.")

        block = self._require_block(G)
        Nx, Ny = block.shape[:2]
        x_idx = np.arange(Nx)[:, None]
        y_idx = np.arange(Ny)[None, :]

        x_next = (x_idx + 1) % Nx
        y_next = (y_idx + 1) % Ny

        G11_x = block[x_idx, y_idx, 0, x_next, y_idx, 0]
        G22_x = block[x_idx, y_idx, 1, x_next, y_idx, 1]
        G12_x = block[x_idx, y_idx, 0, x_next, y_idx, 1]

        G11_y = block[x_idx, y_idx, 0, x_idx, y_next, 0]
        G22_y = block[x_idx, y_idx, 1, x_idx, y_next, 1]
        G12_y = block[x_idx, y_idx, 0, x_idx, y_next, 1]

        J_x = np.imag(-G11_x + G22_x + 1j * G12_x)
        J_y = np.imag(-G11_y + G22_y - G12_y)
        return J_x, J_y

    def squared_two_point_corr(self, G, rx=0, ry=0):
        '''Return |G|^2 averaged over lattice displacements (rx, ry).'''
        G = self._require_block(G)
        Nx, Ny, _, _, _, _ = G.shape
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
        rx_arr = np.atleast_1d(rx).astype(int)
        ry_arr = np.atleast_1d(ry).astype(int)
        Xb = X[:, :, None, None]
        Yb = Y[:, :, None, None]
        Xp = (Xb + rx_arr[None, None, :, None]) % Nx
        Yp = (Yb + ry_arr[None, None, None, :]) % Ny
        blocks = G[Xb, Yb, :, Xp, Yp, :]
        C = np.sum(np.abs(blocks)**2, axis=(0, 1, 4, 5)) / (2.0 * Nx * Ny)
        return C

    def squared_two_point_corr_xslice(self, G, x0=0, ry=0):
        '''Return |G|^2 along a fixed x-column for specified y displacements.'''
        G = self._require_block(G)
        Nx, Ny, _, _, _, _ = G.shape
        x0 = int(x0) % Nx
        Y = np.arange(Ny, dtype=np.intp)[:, None]
        ry_arr = np.atleast_1d(ry).astype(np.intp)
        R = ry_arr.size
        Yp = (Y + ry_arr[None, :]) % Ny
        Gx = G[x0, :, :, x0, :, :]                       # (Ny,2,Ny,2)
        Gx_re = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny*Ny, 2, 2)
        flat_idx = (Y * Ny + Yp).reshape(-1)
        blocks = Gx_re[flat_idx].reshape(Ny, R, 2, 2)
        C = np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny)
        return C

    # -------------------------- Real-space Chern number -----------------------

    def _build_tripartition_masks(self, R_frac=0.4):
        '''Construct tripartition masks used by the Chern number estimator.'''
        Nx, Ny = self.Nx, self.Ny
        R = R_frac * min(Nx, Ny)
        xref, yref = Nx // 2, Ny // 2
        inside = np.zeros((Nx, Ny), dtype=bool)
        A = np.zeros_like(inside)
        B = np.zeros_like(inside)
        C = np.zeros_like(inside)
        rr = R * R
        ymax = int(math.floor(R))
        a2 = 2*np.pi/3
        a4 = 4*np.pi/3
        for dy in range(-ymax, ymax + 1):
            y = yref + dy
            if y < 0 or y >= Ny:
                continue
            max_dx = int(math.floor(math.sqrt(max(0.0, rr - dy*dy))))
            x0 = max(0, xref - max_dx)
            x1 = min(Nx - 1, xref + max_dx)
            if x0 > x1:
                continue
            inside[x0:x1+1, y] = True
            dxs = np.arange(x0, x1+1) - xref
            dys = np.full_like(dxs, dy)
            theta = np.mod(np.arctan2(dys, dxs), 2*np.pi)
            A[x0:x1+1, y] = (theta >= 0)  & (theta < a2)
            B[x0:x1+1, y] = (theta >= a2) & (theta < a4)
            C[x0:x1+1, y] = (theta >= a4) & (theta < 2*np.pi)
        return A, B, C, inside

    def real_space_chern_number(self, G, A_mask=None, B_mask=None, C_mask=None):
        '''Compute the Bianco–Resta real-space Chern number from snapshot ``G``.'''
        if G is None:
            raise ValueError("G must be provided as a correlation snapshot.")

        G_block = self._require_block(G)
        Nx, Ny = self.Nx, self.Ny

        if A_mask is None or B_mask is None or C_mask is None:
            A_mask, B_mask, C_mask, _ = self._build_tripartition_masks()

        P = self._block_to_dense(G_block).conj()

        def sector_indices(mask_xy):
            sites = np.flatnonzero(mask_xy.ravel(order='C'))
            return np.concatenate((2*sites, 2*sites + 1))

        iA = sector_indices(A_mask); iB = sector_indices(B_mask); iC = sector_indices(C_mask)
        P_CA = P[np.ix_(iC, iA)]; P_AB = P[np.ix_(iA, iB)]; P_BC = P[np.ix_(iB, iC)]
        P_AC = P[np.ix_(iA, iC)]; P_CB = P[np.ix_(iC, iB)]; P_BA = P[np.ix_(iB, iA)]
        t1 = np.trace(P_CA @ P_AB @ P_BC)
        t2 = np.trace(P_AC @ P_CB @ P_BA)
        Y = 12 * np.pi * 1j * (t1 - t2)
        return np.real_if_close(Y, tol=1e-6)


    def entanglement_contour(self, G):
        '''Compute the entanglement-contour map s(x,y) from snapshot ``G``.'''
        if G is None:
            raise ValueError("G must be provided as a correlation snapshot.")

        block = self._require_block(G)
        Nx, Ny = int(self.Nx), int(self.Ny)
        C = self._block_to_dense(block)

        # Eigen-decomposition and entropy kernel
        evals, vecs = np.linalg.eigh(C)
        # Numerical safety: clip to (0,1)
        evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
        h = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))  # (Ntot,)

        # diag(F) = sum_k h_k * |vecs[i,k]|^2
        diagF = np.einsum("ik,k,ik->i", vecs, h, vecs.conj(), optimize=True).real  # (Ntot,)

        # reshape back to (Nx, Ny, 2) with C-order mapping, then sum over μ
        diagF = diagF.reshape(Nx, Ny, 2, order='C')  # (x, y, μ)
        s = diagF.sum(axis=2)                        # (Nx, Ny)
        return s
