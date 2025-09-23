import numpy as np
import math
import time
import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl



class classA_U1FGTN:
    def __init__(self, Nx, Ny, DW=True, cycles=None, samples=None, nshell=None, filling_frac=1/2, G0=None):
        """
        1) Initialize random complex covariance over top and bottom layers
           with prescribed filling fraction.
        2) Build overcomplete Wannier spinors for a Chern insulator model.
        """
        self.time_init = time.time()
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.DW = bool(DW)
        self.Ntot = 4 * self.Nx * self.Ny     # 2 orbitals × Nx × Ny × 2 layers
        self.nshell = nshell

        self.cycles = 5 if cycles is None else int(cycles)
        self.samples = 5 if samples is None else int(samples)

        if G0 is None:
            G = self.random_complex_fermion_covariance(N=self.Ntot, filling_frac=filling_frac)
            # initialize bottom layer as a product state (stochastic measurement)
            self.G0 = self.measure_all_bottom_modes(G)
        else:
            self.G0 = np.asarray(G0, dtype=np.complex128)

        # Build OW data
        self.construct_OW_projectors(nshell=nshell, DW=self.DW)

        print("------------------------- classA_U1FGTN Initialized -------------------------")

    # ------------------------------ Utilities ------------------------------

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

    def random_unitary(self, N, rng=None):
        """
        Generate a random unitary U = V exp(i diag(w)) V^† by diagonalizing a random
        Hermitian matrix H. rng: numpy Generator (optional).
        """
        rng = np.random.default_rng() if rng is None else rng
        M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H = 0.5 * (M + M.conj().T)
        w, V = np.linalg.eigh(H)
        U = V @ np.diag(np.exp(1j * w)) @ V.conj().T
        return U

    def random_complex_fermion_covariance(self, N, filling_frac, rng=None):
        """
        Build G = U^† D U with D = diag(+1...+1, -1...-1) at the specified filling fraction.
        """
        assert N % 2 == 0, "Total dimension N must be even."
        rng = np.random.default_rng() if rng is None else rng

        Nfill = int(round(filling_frac * N))
        Nfill = max(0, min(N, Nfill))
        diag = np.concatenate([np.ones(Nfill), -np.ones(N - Nfill)])
        D = np.diag(diag).astype(np.complex128)

        U = self.random_unitary(N, rng=rng)
        return U.conj().T @ D @ U

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

    def construct_OW_projectors(self, nshell, DW):
        Nx, Ny = self.Nx, self.Ny

        # Mass profile alpha(x) for the Dirac-CI model
        if DW:
            alpha = np.full((Nx, Ny), 3, dtype=np.complex128)  # trivial
            half = Nx // 2
            w = max(1, int(np.floor(0.2 * Nx)))
            x0 = max(0, half - w)
            x1 = min(Nx, half + w + 1)  # inclusive slab -> slice end-exclusive
            alpha[x0:x1, :] = 1         # topological region
            print(f"DWs at x=({int(x0-1)}, {int(x1-1)})")
            self.DW_loc = [int(x0-1), int(x1-1)]
        else:
            alpha = np.ones((Nx, Ny), dtype=np.complex128)
        self.alpha = alpha

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

        # Projectors per center: χ χ†
        def projectors(WF):   # WF: (D, Rx, Ry)
            return np.einsum('dxy,exy->dexy', WF, WF.conj(), optimize=True)

        self.P_Ap = projectors(self.WF_Ap)
        self.P_Bp = projectors(self.WF_Bp)
        self.P_Am = projectors(self.WF_Am)
        self.P_Bm = projectors(self.WF_Bm)

    # --------------------------- Measurement updates ---------------------------

    def measure_bottom_layer(self, G, P, particle=True, symmetrize=True):
        """
        Charge-conserving EPR update on bottom layer with projector P.
        """
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

        K = np.block([[Gbb,    -Il],
                      [-Il,     H11]])
        L = self._block_diag2(Gtb, H21)

        invK_Ldag = self._solve_regularized(K, L.conj().T, eps=1e-9)

        M = self._block_diag2(Gtt, H22)
        Gp = M - L @ invK_Ldag

        if symmetrize:
            Gp = 0.5 * (Gp + Gp.conj().T)
        return Gp

    def measure_top_layer(self, G, P, particle=True, symmetrize=True):
        """
        Charge-conserving EPR update on top layer with projector P.
        """
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

        K = np.block([[H11,  -Il],
                      [-Il,   Gtt]])
        L = self._block_diag2(H21, Gbt)

        invK_Ldag = self._solve_regularized(K, L.conj().T, eps=1e-9)

        M = self._block_diag2(H22, Gbb)
        Gp = M - L @ invK_Ldag

        if symmetrize:
            Gp = 0.5 * (Gp + Gp.conj().T)
        return Gp

    def fSWAP(self, chi_top, chi_bottom):
        """
        Exact subspace swap of the top Wannier mode chi_top with the bottom local mode chi_bottom.
        U is unitary, Hermitian, and an involution: U^2 = I.
        Safe for read-only/memmapped inputs by copying before normalization.
        """
        Ntot = self.Ntot
        Nlayer = Ntot // 2
    
        # Make explicit writable copies (avoid in-place on read-only views)
        ct = np.asarray(chi_top,    dtype=np.complex128).reshape(-1).copy()
        cb = np.asarray(chi_bottom, dtype=np.complex128).reshape(-1).copy()
    
        # Normalize without in-place division
        nct = np.linalg.norm(ct) + 1e-15
        ncb = np.linalg.norm(cb) + 1e-15
        ct = ct / nct
        cb = cb / ncb
    
        # Embed into full top/bottom layer vectors
        psi_t = np.zeros(Ntot, dtype=np.complex128); psi_t[:Nlayer]  = ct
        psi_b = np.zeros(Ntot, dtype=np.complex128); psi_b[Nlayer:]  = cb
    
        # Projectors and swap unitary
        Pt  = np.outer(psi_t, psi_t.conj())
        Pb  = np.outer(psi_b, psi_b.conj())
        Xtb = np.outer(psi_t, psi_b.conj())
        Xbt = np.outer(psi_b, psi_t.conj())
        U   = (np.eye(Ntot, dtype=np.complex128) - Pt - Pb) + (Xtb + Xbt)
        return U
    
        # ---------------------- Local feedback / post-selection ----------------------

    def top_layer_meas_feedback(self, G, Rx, Ry):
        """
        Measurement-feedback at a given center (Rx,Ry).
        """
        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        Il = np.eye(Nlayer, dtype=np.complex128)
        Gtt_2pt = 0.5 * (G[:Nlayer, :Nlayer] + Il)

        P_Ap = self.P_Ap[:, :, Rx, Ry]; P_Bp = self.P_Bp[:, :, Rx, Ry]
        P_Am = self.P_Am[:, :, Rx, Ry]; P_Bm = self.P_Bm[:, :, Rx, Ry]

        chi_Ap = self.WF_Ap[:, Rx, Ry]; chi_Bp = self.WF_Bp[:, Rx, Ry]
        chi_Am = self.WF_Am[:, Rx, Ry]; chi_Bm = self.WF_Bm[:, Rx, Ry]

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
        """
        Directly impose the four outcomes (no feedback unitary).
        """
        P_Ap = self.P_Ap[:, :, Rx, Ry]
        P_Bp = self.P_Bp[:, :, Rx, Ry]
        P_Am = self.P_Am[:, :, Rx, Ry]
        P_Bm = self.P_Bm[:, :, Rx, Ry]

        G = self.measure_top_layer(G, P_Ap, particle=False)  # Ap unocc
        G = self.measure_top_layer(G, P_Bp, particle=False)  # Bp unocc
        G = self.measure_top_layer(G, P_Am, particle=True)   # Am occ
        G = self.measure_top_layer(G, P_Bm, particle=True)   # Bm occ
        return G

    def measure_all_bottom_modes(self, G):
        """
        Measure all local mode occupancies in bottom layer (one pass, correct Bernoulli).
        Prints elapsed time when finished (unless suppressed).
        """
        start = time.time()

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

        self.bottom_layer_mode_meas_time = time.time() - start
        if not getattr(self, "_suppress_bottom_measure_prints", False):
            print()
            print(f"\nAll bottom layer modes measured | Time elapsed: {self.bottom_layer_mode_meas_time:.3f} s", flush=True)

        return G

    def randomize_bottom_layer(self, G):
        """
        Apply a random unitary on the bottom layer only.
        """
        G = np.asarray(G, dtype=np.complex128)
        Ntot = self.Ntot
        Nlayer = Ntot // 2
        Il = np.eye(Nlayer, dtype=np.complex128)

        U_bott = self.random_unitary(Nlayer)
        U_tot = self._block_diag2(Il, U_bott)
        return U_tot.conj().T @ G @ U_tot

    # ------------------------------ Circuit driver ------------------------------

    def run_adaptive_circuit(
    self,
    G_history=True,
    tol=1e-8,
    progress=True,
    postselect=False,
    print_eta_rate=3.0,
):
        """
        Adaptive circuit with ETA that explicitly models:
          - per-site work via EMA (_ema_site)
          - once-per-cycle bottom-layer measurement via EMA (_ema_bottom)
        ETA left = [sites_left * _ema_site + bottom_ops_left * _ema_bottom] / effective_parallel
        Print throttled by `print_eta_rate` seconds.
        """

        # fresh state
        self.G = np.array(self.G0, copy=True)
        if G_history:
            self.G_list = []
        self.g2_flags = []

        Nx, Ny = int(self.Nx), int(self.Ny)
        D = int(self.Ntot)
        I = np.eye(D, dtype=np.complex128)

        total_sites = self.cycles * Nx * Ny
        sites_done = 0

        # EMAs (persist across calls if present, else bootstrap at first measurements)
        self._ema_site = float(getattr(self, "_ema_site", 0.0))
        self._ema_bottom = float(getattr(self, "_ema_bottom", 0.0))
        alpha_site = 0.15
        alpha_bottom = 0.25

        # backend-aware parallel factor (set by corr_y_profiles_v2)
        raw_parallel = int(getattr(self, "_eta_parallel_factor", 1))
        backend = str(getattr(self, "backend", "")).lower()
        if backend == "loky":
            eff = 0.90
        elif backend == "threading":
            eff = 0.50  # threads often underutilize on NumPy-heavy code
        else:
            eff = 0.80
        effective_parallel = max(1, int(max(1, raw_parallel) * eff))

        # tqdm/print timers
        t_total0 = time.perf_counter()
        last_eta_print = 0.0
        print_eta_rate = 10.0 if print_eta_rate is None else float(print_eta_rate)

        def fmt(t):
            return tqdm.format_interval(max(0.0, float(t)))

        # OUTER: cycles
        outer_iter = range(self.cycles)
        if progress:
            outer_iter = tqdm(outer_iter, total=self.cycles, leave=True, desc="Cycles")

        for c in outer_iter:
            t_cycle0 = time.perf_counter()
            # bottom measurement for this cycle is still pending (if not postselect)
            in_cycle_bottom_pending = (not postselect)

            # MIDDLE: rows
            row_iter = range(Nx)
            if progress:
                row_iter = tqdm(row_iter, total=Nx, leave=False, desc="Rx Loop")

            for Rx in row_iter:
                t_row0 = time.perf_counter()

                # INNER: columns
                col_iter = range(Ny)
                if progress:
                    col_iter = tqdm(col_iter, total=Ny, leave=False, desc="Ry Loop:")

                for Ry in col_iter:
                    t_step0 = time.perf_counter()

                    # step
                    if not postselect:
                        self.G = self.top_layer_meas_feedback(self.G, Rx, Ry)
                    else:
                        self.G = self.post_selection_top_layer(self.G, Rx, Ry)

                    # G^2≈I
                    ok = int(np.allclose(self.G @ self.G, I, atol=tol))
                    self.g2_flags.append(ok)

                    # counters
                    sites_done += 1

                    # update EMA for per-site
                    dt_site = time.perf_counter() - t_step0
                    if self._ema_site == 0.0:
                        self._ema_site = dt_site
                    else:
                        self._ema_site = (1.0 - alpha_site) * self._ema_site + alpha_site * dt_site

                    # tqdm cosmetics
                    if progress:
                        # inner (row)
                        elapsed_row = time.perf_counter() - t_row0
                        frac_row = (Ry + 1) / Ny
                        eta_row_total = (elapsed_row / frac_row - elapsed_row) if frac_row > 0 else 0.0
                        try:
                            col_iter.set_postfix(
                                {"elapsed": fmt(elapsed_row), "eta_total": fmt(eta_row_total)},
                                refresh=False,
                            )
                        except Exception:
                            pass
                        # outer (total)
                        elapsed_total = time.perf_counter() - t_total0
                        frac_total = sites_done / max(1, total_sites)
                        eta_total_total = (elapsed_total / frac_total - elapsed_total) if frac_total > 0 else 0.0
                        try:
                            outer_iter.set_postfix_str(
                                f"elapsed={fmt(elapsed_total)} | eta_total={fmt(eta_total_total)}\nG2==I: {ok}",
                                refresh=False,
                            )
                        except Exception:
                            pass

                    # ---- throttled single-line ETA print (project remaining) ----
                    now = time.time()
                    if now - last_eta_print >= print_eta_rate:
                        # sites left (this cycle remainder + remaining full cycles)
                        sites_done_in_cycle = Rx * Ny + (Ry + 1)
                        sites_left_in_cycle = Nx * Ny - sites_done_in_cycle
                        cycles_left_after = self.cycles - (c + 1)
                        sites_left_total = sites_left_in_cycle + cycles_left_after * Nx * Ny

                        # bottom-layer measurements left:
                        # one for current cycle if still pending, plus one per remaining full cycle
                        bottom_ops_left = (1 if in_cycle_bottom_pending else 0) + max(0, cycles_left_after)

                        eta_left_wall = (
                            sites_left_total * self._ema_site + bottom_ops_left * self._ema_bottom
                        ) / max(1, effective_parallel)

                        # total ETA (from start) ~ cycles * (Nx*Ny*ema_site + ema_bottom) / parallel
                        work_per_cycle = Nx * Ny * self._ema_site + self._ema_bottom
                        eta_total_wall = (self.cycles * work_per_cycle) / max(1, effective_parallel)

                        print(
                            f"\rETA total: {self.format_interval(eta_total_wall)} | "
                            f"ETA left: {self.format_interval(eta_left_wall)}",
                            end="",
                            flush=True,
                        )
                        last_eta_print = now

            # end of cycle: bottom-layer randomize + measure (once per cycle)
            if not postselect:
                self.G = self.randomize_bottom_layer(self.G)

                t_bottom0 = time.perf_counter()
                self.G = self.measure_all_bottom_modes(self.G)  # sets self.bottom_layer_mode_meas_time
                dt_bottom = time.perf_counter() - t_bottom0
                # prefer the routine’s own timer if it exists
                dt_meas = float(getattr(self, "bottom_layer_mode_meas_time", dt_bottom))
                if self._ema_bottom == 0.0:
                    self._ema_bottom = dt_meas
                else:
                    self._ema_bottom = (1.0 - alpha_bottom) * self._ema_bottom + alpha_bottom * dt_meas

                in_cycle_bottom_pending = False  # done for this cycle

            if G_history:
                self.G_list.append(self.G.copy())

        # finish the inline ETA line with a newline
        print()

    # ---------------------------- Chern observables ----------------------------

    def real_space_chern_number(self, G=None):
        """
        Real-space Chern number from top-layer projector P = G_2pt^*:
        12π i [ Tr(P_CA P_AB P_BC) − Tr(P_AC P_CB P_BA) ]  (return real part).
        """
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

        Guse = self.G if G is None else np.asarray(G, dtype=np.complex128)
        Gtt = Guse[:Nlayer, :Nlayer]
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

    def local_chern_marker_flat(self, G=None, mask_outside=False, inside_mask=None):
        """
        Local Chern marker from a FLAT top-layer covariance using the reshaped kernel.
        """
        Nx, Ny = self.Nx, self.Ny
        Nlayer = 2 * Nx * Ny

        if G is None:
            Gflat = np.asarray(self.G, dtype=np.complex128)[:Nlayer, :Nlayer]
        else:
            G = np.asarray(G, dtype=np.complex128)
            if G.shape == (self.Ntot, self.Ntot):
                Gflat = G[:Nlayer, :Nlayer]
            elif G.shape == (Nlayer, Nlayer):
                Gflat = G
            else:
                raise ValueError(f"G must be ({self.Ntot},{self.Ntot}) or ({Nlayer},{Nlayer}), got {G.shape}")

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

        C = np.tanh(np.real_if_close(C, tol=1e-9))
        if mask_outside and inside_mask is not None:
            C = np.where(inside_mask, C, 0.0)
        return C

    # ------------------------- Visualization helpers -------------------------

    def plot_real_space_chern_history(self, filename=None):
        """
        Plot the real-space Chern number across the *existing* history self.G_list.

        Requirements
        ------------
        - Uniform system only: raises if self.DW is True.
        - self.G_list must exist and be non-empty (i.e., run with G_history=True beforehand).

        Returns
        -------
        fig, ax, cherns
          fig/ax : Matplotlib figure/axes
          cherns : np.ndarray of real parts of the Chern number per history frame
        """

        # Guardrails
        if getattr(self, "DW", True):
            raise ValueError("plot_real_space_chern_history: only supported for uniform systems (self.DW == False).")
        if not hasattr(self, "G_list") or not isinstance(self.G_list, list) or len(self.G_list) == 0:
            raise RuntimeError("No history found. Run the circuit with G_history=True to populate self.G_list.")

        # Compute Chern per stored snapshot
        cherns = np.empty(len(self.G_list), dtype=float)
        for k, Gk in enumerate(self.G_list):
            cherns[k] = np.real(self.real_space_chern_number(Gk))

        # Plot
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        x = np.arange(1, len(cherns) + 1)
        ax.plot(x, cherns, marker="o", lw=1.25)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Real-space Chern Number")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save if requested
        if filename is not None:
            outdir = self._ensure_outdir(os.path.dirname(filename) or "figs/chern_history")
            base = os.path.basename(filename)
            fullpath = os.path.join(outdir, base)
            fig.savefig(fullpath, bbox_inches="tight")

        return fig, ax, cherns

    def chern_marker_dynamics(self, outbasename=None, vmin=-1.0, vmax=1.0, cmap='RdBu_r'):
        """
        Animate local Chern marker over the cached history self.G_list (or current self.G).
        Shows fixed-position textboxes for:
          - cycle index t (starting from 1),
          - domain-wall locations (if self.DW_loc exists).
        """
        Nx, Ny = self.Nx, self.Ny
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            nshell_str = getattr(self, "nshell", None)
            nshell_str = "None" if nshell_str is None else str(nshell_str)
            outbasename = f"chern_marker_dynamics_N={Nx}_nshell={nshell_str}_cycles={getattr(self,'cycles','NA')}_DWis{getattr(self,'DW','NA')}"
        gif_path   = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.pdf")

        # Choose history
        if hasattr(self, "G_list") and isinstance(self.G_list, list) and len(self.G_list) > 0:
            history = self.G_list
        else:
            if not hasattr(self, "G"):
                raise RuntimeError("No state available: run the circuit first.")
            history = [self.G]

        # --- Figure for animation ---
        fig = plt.figure(figsize=(3.2, 3.8))
        ax  = fig.add_axes([0.12, 0.10, 0.78, 0.78])
        im  = ax.imshow(np.zeros((Nx, Ny)), cmap=cmap, vmin=vmin, vmax=vmax,
                        origin='upper', aspect='equal')

        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color('black')
        ax.set_xlabel("y"); ax.set_ylabel("x")
        ax.set_xticks(np.arange(0, Ny, max(1, Ny//10)))
        ax.set_yticks(np.arange(0, Nx, max(1, Nx//10)))
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)

        # Colorbar
        cax = fig.add_axes([0.32, 0.90, 0.36, 0.06])
        fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
        fig.text(0.70, 0.91, r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=12)

        # Fixed-position textbox for cycle index, placed to the LEFT of the colorbar
        tbox = fig.text(
            0.26, 0.91, "t = 1",
            fontsize=11, ha='right', va='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.25')
        )

        # Optional fixed-position textbox for DW locations (if available)
        if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
            fig.text(
                0.26, 0.86, f"DW x₀ = {self.DW_loc[0]}, {self.DW_loc[1]}",
                fontsize=10, ha='right', va='center',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.25')
            )

        # Animate
        C = None
        writer = animation.PillowWriter(fps=1)
        with writer.saving(fig, gif_path, dpi=120):
            for t, Gf in enumerate(tqdm(history, desc="chern_marker_frames", unit="frame"), start=1):
                C = self.local_chern_marker_flat(Gf)
                im.set_data(C)
                tbox.set_text(f"t = {t}")   # update cycle index
                writer.grab_frame()
        plt.close(fig)

        # --- Final static frame with the same textboxes (t = total frames) ---
        C_last = np.round(C, 2)
        fig2 = plt.figure(figsize=(3.2, 3.8))
        ax2  = fig2.add_axes([0.12, 0.10, 0.78, 0.78])
        im2  = ax2.imshow(C_last, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', aspect='equal')
        for sp in ax2.spines.values():
            sp.set_linewidth(1.5); sp.set_color('black')
        ax2.set_xlabel("y"); ax2.set_ylabel("x")
        ax2.set_xticks(np.arange(0, Ny, max(1, Ny//10)))
        ax2.set_yticks(np.arange(0, Nx, max(1, Nx//10)))
        ax2.tick_params(axis='both', labelsize=8)
        ax2.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        ax2.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)

        cax2 = fig2.add_axes([0.32, 0.90, 0.36, 0.06])
        fig2.colorbar(im2, cax=cax2, orientation='horizontal', ticks=[-1, 0, 1])
        fig2.text(0.70, 0.91, r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=12)

        # Same textbox positions & styles
        fig2.text(
            0.26, 0.91, f"t = {len(history)}",
            fontsize=11, ha='right', va='center',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.25')
        )
        if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
            fig2.text(
                0.26, 0.86,
                rf"$\text{{DW at }} x_0 = {self.DW_loc[0]},\;{self.DW_loc[1]}$",
                fontsize=10, ha='right', va='center',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.85, boxstyle='round,pad=0.25')
            )

        fig2.savefig(final_path, bbox_inches='tight')
        plt.show()
        plt.close(fig2)

        return gif_path, final_path, C_last, history[-1]

    # ---------------------- Parallel-safe spawner for v2 ----------------------

    def _spawn_for_parallel(self):
        """Make a worker copy with shared read-only big arrays to avoid races."""
        child = object.__new__(self.__class__)
        # copy scalars
        child.Nx, child.Ny = self.Nx, self.Ny
        child.Ntot = self.Ntot
        child.nshell = self.nshell
        child.DW = self.DW
        child.cycles = self.cycles
        child.samples = self.samples
        child.time_init = self.time_init
        # share read-only big arrays (must not be mutated in workers!)
        child.Pminus = self.Pminus
        child.Pplus  = self.Pplus
        child.WF_Ap, child.WF_Bp, child.WF_Am, child.WF_Bm = self.WF_Ap, self.WF_Bp, self.WF_Am, self.WF_Bm
        child.P_Ap, child.P_Bp, child.P_Am, child.P_Bm = self.P_Ap, self.P_Bp, self.P_Am, self.P_Bm
        child.alpha = self.alpha
        child.DW_loc = getattr(self, "DW_loc", None)
        # stateful things initialized per worker
        child.G0 = np.array(self.G0, copy=True)
        child.G = None
        child.G_list = []
        child.g2_flags = []
        # ETA/prints off in workers
        child._eta_step_baseline = None
        child._suppress_bottom_measure_prints = True
        return child

    # ---------------------- Parallelized corr y-profiles (v2) ----------------------

    def plot_corr_y_profiles_v2(self, samples=None, n_jobs=None, save_data=True, filename=None, ry_max=None, backend="loky"):
        """
        Parallelized y-profile computation that ALWAYS produces both:
          - \\overline{C}_G   (sample-average of C per trajectory)
          - C_{\\overline{G}} (C of the sample-averaged G)
        And plots a 3x2 panel: (top) the two y-profiles, (mid) eigenvalue spectra,
        (bottom) local Chern marker images.

        Parallelism: joblib with backend selectable ('loky' preferred).
        Each worker runs on a light cloned instance created by _spawn_for_parallel().
        """
        Nx, Ny = self.Nx, self.Ny
        Ntot   = self.Ntot
        Nlayer = Ntot // 2

        if samples is None:
            samples = int(getattr(self, "samples", 4))
        samples = int(max(1, samples))

        if n_jobs is None:
            n_jobs = min(samples, os.cpu_count() or 1)
        n_jobs = max(1, int(n_jobs))

        self.backend = backend        # "loky" or "threading"
        self._eta_parallel_factor = n_jobs

        # pick 9 x0 positions from DW_loc (your spec)
        def _pick_x_positions():
            if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
                xs = [
                    (xL // 2) % Nx,                   # middle of left trivial
                    (xL - 1) % Nx, xL % Nx, (xL + 1) % Nx,
                    ((xL + xR) // 2) % Nx,            # middle of non-trivial
                    (xR - 1) % Nx, xR % Nx, (xR + 1) % Nx,
                    (xR + (Nx // 2)) % Nx,            # middle of right trivial (simple symmetric)
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

        # y-separations
        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # helpers
        def _coerce_to_two_point_kernel_top(G_in):
            Gin = np.asarray(G_in, dtype=np.complex128)
            if Gin.ndim == 6:
                return Gin
            elif Gin.ndim == 2:
                if Gin.shape == (Ntot, Ntot):
                    Gtt = Gin[:Nlayer, :Nlayer]
                elif Gin.shape == (Nlayer, Nlayer):
                    Gtt = Gin
                else:
                    raise ValueError(f"G has incompatible shape {Gin.shape}.")
                G2 = 0.5 * (Gtt + np.eye(Nlayer, dtype=np.complex128))
                G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
                return np.transpose(G6, (1, 2, 0, 4, 5, 3))
            else:
                raise ValueError("G must be (4N,4N), (2N,2N), or (Nx,Ny,2,Nx,Ny,2).")

        def _C_xslice_from_kernel(Gker, x0, ry_vals):
            x0 = int(x0) % Nx
            ry_arr = np.atleast_1d(ry_vals).astype(int)
            Ny_loc = Gker.shape[1]
            Gx = Gker[x0, :, :, x0, :, :]                # (Ny,2,Ny,2)
            Y  = np.arange(Ny_loc, dtype=np.intp)[:, None]
            Yp = (Y + ry_arr[None, :]) % Ny_loc
            Gx_re   = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny_loc*Ny_loc, 2, 2)
            flat_ix = (Y * Ny_loc + Yp).reshape(-1)
            blocks  = Gx_re[flat_ix].reshape(Ny_loc, ry_arr.size, 2, 2)
            return np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny_loc)

        def _eigvals_sorted(A):
            try:
                w = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                w = np.linalg.eigvalsh(0.5*(A + A.conj().T))
            return np.sort(np.real_if_close(w))

        def _chern_from_Gtop(G_top):
            return self.local_chern_marker_flat(G_top)

        # accumulators
        C_accum = {x0: np.zeros_like(ry_vals, dtype=float) for x0, _ in x_positions}
        Gsum    = np.zeros((Nlayer, Nlayer), dtype=np.complex128)
        last_Gtt = None

        # scale ETA inside RAC and silence bottom-layer prints in workers
        self._eta_parallel_factor = n_jobs
        old_suppress = getattr(self, "_suppress_bottom_measure_prints", False)
        self._suppress_bottom_measure_prints = True

        # seeds
        if backend == "loky":
            ss = np.random.SeedSequence()
            seed_ints = ss.generate_state(samples, dtype=np.uint32).tolist()
        else:
            # threading: avoid touching global RNG state with per-thread seeding
            seed_ints = [None] * samples
            print("[corr_y_profiles_v2] Warning: threading backend shares NumPy RNG; results will be non-reproducible per-thread.")

        # worker functions
        def _task_loky(seed):
            child = self._spawn_for_parallel()
            np.random.seed(int(seed) & 0xFFFFFFFF)  # safe per-process seeding
            child.run_adaptive_circuit(cycles=child.cycles, G_history=False, progress=False)
            Gtt  = np.asarray(child.G)[:Nlayer, :Nlayer]
            Gker = _coerce_to_two_point_kernel_top(Gtt)
            C_this = {x0: _C_xslice_from_kernel(Gker, x0, ry_vals).real for x0, _ in x_positions}
            return Gtt, C_this

        def _task_thread(_seed_unused):
            # clone object to avoid shared mutable state (still shares big read-only arrays)
            child = self._spawn_for_parallel()
            # do NOT np.random.seed here (threads share global RNG)
            child.run_adaptive_circuit(cycles=child.cycles, G_history=False, progress=False)
            Gtt  = np.asarray(child.G)[:Nlayer, :Nlayer]
            Gker = _coerce_to_two_point_kernel_top(Gtt)
            C_this = {x0: _C_xslice_from_kernel(Gker, x0, ry_vals).real for x0, _ in x_positions}
            return Gtt, C_this

        # pick worker based on backend
        worker_fn = _task_loky if backend == "loky" else _task_thread

        # run parallel
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(worker_fn)(s) for s in seed_ints
        )

        # restore flags
        self._suppress_bottom_measure_prints = old_suppress
        if hasattr(self, "_eta_parallel_factor"):
            del self._eta_parallel_factor

        # accumulate results
        for Gtt, C_this in results:
            last_Gtt = Gtt
            Gsum += Gtt
            for x0, _ in x_positions:
                C_accum[x0] += C_this[x0]

        # build datasets
        C_resolved = {x0: C_accum[x0] / samples for x0, _ in x_positions}     # \overline{C}_G
        Gavg = Gsum / samples
        Gker_avg = _coerce_to_two_point_kernel_top(Gavg)
        C_avg = {x0: _C_xslice_from_kernel(Gker_avg, x0, ry_vals).real for x0, _ in x_positions}  # C_{\overline{G}}

        # spectra & Chern
        evals_last = _eigvals_sorted(last_Gtt)
        evals_avg  = _eigvals_sorted(Gavg)
        Chern_last = _chern_from_Gtop(last_Gtt)
        Chern_avg  = _chern_from_Gtop(Gavg)

        # ---------------- plotting: 3 x 2 ----------------
        outdir = self._ensure_outdir('figs/corr_y_profiles')
        if filename is None:
            xdesc = "-".join(f"{x}" for x, _ in x_positions)
            filename = f"corr2_y_profiles_v2_N{Nx}_xs_{xdesc}_S{samples}_J{n_jobs}.pdf"
        fullpath = os.path.join(outdir, filename)

        # Make sure TeX is off; we’ll use mathtext only
        mpl.rcParams['text.usetex'] = False

        # Use constrained layout (more robust than tight_layout with mathtext)
        fig = plt.figure(figsize=(12.5, 12.0), constrained_layout=True)
        gs  = fig.add_gridspec(nrows=3, ncols=2,
                               height_ratios=[1.0, 1.0, 1.04],  # tiny extra height for colorbar row
                               width_ratios=[1.0, 1.0])

        axC1 = fig.add_subplot(gs[0, 0])
        axC2 = fig.add_subplot(gs[0, 1])
        axE1 = fig.add_subplot(gs[1, 0], sharex=None)
        axE2 = fig.add_subplot(gs[1, 1], sharex=None)
        axM1 = fig.add_subplot(gs[2, 0])
        axM2 = fig.add_subplot(gs[2, 1])

        # Top row: y-profiles
        for ax, Cdict, ylab in (
            (axC1, C_resolved, r"$\overline{C}_G(x_0,r_y)$"),
            (axC2, C_avg,      r"$C_{\overline{G}}(x_0,r_y)$"),
        ):
            for x0, lbl in x_positions:
                C_vec = Cdict[x0]
                line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=fr"$x_0 = {lbl}$")
                finite = np.isfinite(C_vec)
                y_right = C_vec[finite][-1] if np.any(finite) else C_vec[-1]
                x_right = ry_vals[-1] * 1.02 if ry_vals[-1] > 0 else ry_vals[-1] + 0.5
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
        ymin = min(axC1.get_ylim()[0], axC2.get_ylim()[0])
        ymax = max(axC1.get_ylim()[1], axC2.get_ylim()[1])
        axC1.set_ylim(ymin, ymax)
        axC2.set_ylim(ymin, ymax)

        # Middle row: eigenvalue spectra
        axE1.plot(np.arange(len(evals_last)), np.sort(evals_last), '.', ms=3)
        axE1.set_title(r"eigvals($G_{\mathrm{final}}$)")
        axE1.set_xlabel("index"); axE1.set_ylabel("eigenvalue"); axE1.grid(True, alpha=0.3)

        axE2.plot(np.arange(len(evals_avg)),  np.sort(evals_avg),  '.', ms=3)
        axE2.set_title(r"eigvals($\overline{G}$)")
        axE2.set_xlabel("index"); axE2.set_ylabel("eigenvalue"); axE2.grid(True, alpha=0.3)

        # Bottom row: Chern maps — keep the same column widths by anchoring colorbars to each axis
        im1 = axM1.imshow(Chern_last, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
        axM1.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for final $G$")
        axM1.set_xlabel("y"); axM1.set_ylabel("x"); axM1.grid(False)
        fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04, anchor=(0.5, 0.0))

        im2 = axM2.imshow(Chern_avg,  cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
        axM2.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for $\overline{G}$")
        axM2.set_xlabel("y"); axM2.set_ylabel("x"); axM2.grid(False)
        fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04, anchor=(0.5, 0.0))

        # Title & save — no tight_layout call
        fig.suptitle(f"Correlation profiles & spectra (N={Nx}, samples={samples}, n_jobs={n_jobs})")
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)

        # save datasets
        if save_data:
            xs_list = [x for x, _ in x_positions]
            C_res_mat = np.vstack([C_resolved[x] for x in xs_list])
            C_avg_mat = np.vstack([C_avg[x]       for x in xs_list])
            np.savez(
                os.path.join(outdir, f"corr2_y_profiles_v2_data_N{Nx}_S{samples}_J{n_jobs}_{backend}.npz"),
                ry_vals=ry_vals,
                x_positions=np.array(xs_list, dtype=int),
                Cbar_of_G=C_res_mat,   # \overline{C}_G
                C_of_Gbar=C_avg_mat    # C_{\overline{G}}
            )

        return fullpath

    # ------------------------------ Exact CI state ------------------------------

    def G_CI(self, alpha=1.0, k_is_centered=False, norm='backward'):
        """
        Build the complex covariance G = 2 P_-^* - I for the uniform CI lower band (top layer).
        Flattening index: i = μ + 2*x + 2*Nx*y (μ fastest; Fortran order).
        """
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