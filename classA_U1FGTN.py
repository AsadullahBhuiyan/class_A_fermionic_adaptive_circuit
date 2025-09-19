import numpy as np
import math
import time
import os
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tqdm import tqdm

class classA_U1FGTN:

    def __init__(self, Nx, Ny, alpha_init = 1.0, DW = True, nshell=None, cycles = 5, keep_history_init=True, filling_frac = 1/2, G0 = None):
        """
        1) Initialize random complex covariance over top and bottom layers with init charge in [Q_tot/4, 3Q_tot/4]

        2) Build overcomplete Wannier spinors for a Chern insulator model.

        """
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.cycles = int(cycles)
        self.keep_history_default = bool(keep_history_init)
        
        if DW is not True:
            self.alpha = alpha_init

        self.Ntot = 2*self.Nx*self.Ny*2 # orbtial * xdim * ydim * layers 

        if G0 is None:  
            self.G0 = self.random_complex_fermion_covariance(N = self.Ntot, filling_frac = filling_frac)
        
        # initialize bottom layer as product state
        self.G0 = self.measure_all_bottom_modes(self.G0)
        
        # --- Build overcomplete Wannier data; evolve only if not already cached ---
        # Construct OW (may set a spatially varying alpha if DW=True)
        self.construct_OW_projectors(nshell=nshell, DW = DW)

    def random_unitary(self, N, rng=None):
        """
        Generate a random unitary U = exp(i H),
        where H is Hermitian from a Gaussian matrix M.
        """
        if rng is None:
            rng = np.random.default_rng()
        M = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        H = 0.5 * (M + M.conj().T)

        # Diagonalize H and exponentiate eigenvalues
        w, V = np.linalg.eigh(H)      # H = V diag(w) V^†
        U = V @ np.diag(np.exp(1j * w)) @ V.conj().T
        return U    
    
    def random_complex_fermion_covariance(self, N, filling_frac, rng=None):
        """
        Build G = U^† D U, where D = diag(1,...,1,-1,...,-1) with half +1 and half -1.
        Dimension is 2*Nx*Ny x 2*Nx*Ny.
        """
        assert N % 2 == 0, "Total dimension N must be even."

        if rng is None:
            rng = np.random.default_rng()

        Nfill = np.floor(filling_frac*N)
        Nempty = np.floor((1-filling_frac)*N)
        diag = np.concatenate([np.ones(Nfill), -np.ones(Nempty)])
        D = np.diag(diag).astype(np.complex128)

        U = self.random_unitary(N, rng=rng)
        G = U.conj().T @ D @ U
        return G

    def construct_OW_projectors(self, nshell, DW):
        
        Nx, Ny = self.Nx, self.Ny

        if DW:
            # Create two domain walls: alpha = 1 in the central slab
            # [Nx//2 - floor(0.2*Nx), Nx//2 + floor(0.2*Nx)] × [0, Ny)
            # and alpha = 3 elsewhere.
            alpha = np.full((Nx, Ny), 3, dtype=complex)  # outside slab (Chern=0)
            half = self.Nx // 2
            w = int(np.floor(0.2 * Nx))
            x0 = max(0, half - w)
            # Python slices are end-exclusive; include the right edge by +1, then clamp
            x1 = min(Nx, half + w + 1)
            alpha[x0:x1, :] = 1  # inside slab (Chern=1)
            print(f"DWs at x=({int(x0)}, {int(x1-1)})")
            self.alpha = alpha
        else:
            alpha = np.ones((Nx, Ny))
            self.alpha = alpha

        # k-grids in radians per lattice spacing (FFT ordering)
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=1.0)     # shape (Nx,)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=1.0)     # shape (Ny,)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # shape (Nx, Ny)

        # model vector n(k)
        nx = np.sin(KX)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        ny = np.sin(KY)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        nz = alpha[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None] # (Nx, Ny, Rx, Ry)
        nmag = np.sqrt(nx**2 + ny**2 + nz**2) # (Nx, Ny, Rx, Ry)
        nmag = np.where(nmag == 0, 1e-15, nmag)  # avoid divide-by-zero

        # --- Pauli matrices ---
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # construct 2x2 identity (broadcasts over k-grid)
        Id = np.eye(2, dtype=complex)

        # construct k-space single-particle h(k) = n̂ · σ (unit vector)
        hk = (nx[..., None, None] * pauli_x +
              ny[..., None, None] * pauli_y +
              nz[..., None, None] * pauli_z) / nmag[..., None, None]  # (Nx, Ny, Rx, Ry, 2, 2)

        # construct upper and lower band projectors
        self.Pminus = 0.5 * (Id - hk)   # (Nx, Ny, Rx, Ry, 2, 2)
        self.Pplus  = 0.5 * (Id + hk)   # (Nx, Ny, Rx, Ry, 2, 2)

        # local choice of orbitals
        tauA = (1/np.sqrt(2)) * np.array([[1], [1]], dtype=complex)   # (2,1)
        tauB = (1/np.sqrt(2)) * np.array([[1], [-1]], dtype=complex)  # (2,1)

        # --- phases for all centers (R_x, R_y) ---
        Rx_grid = np.arange(Nx)                                   # (Rx,)
        Ry_grid = np.arange(Ny)                                   # (Ry,)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Rx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ry)
        phase   = phase_x * phase_y                                # (Nx,Ny,Rx,Ry)

        # Helper: do FFT over k-axes only (0,1), for every (R_x,R_y)
        def k2_to_r2(Ak):  # Ak shape (Nx, Ny, Rx, Ry)
            return np.fft.fft2(Ak, axes=(0, 1))

        # ------------------------------
        # helper to build a normalized W for every center
        # row spinor in k-space: psi_k = tau^† P(k)
        # ------------------------------
        def make_W(Pband, tau, phase):
            # tau^\dagger P(k): (2,)^* with (...,2,2) over left index -> (...,2)
            tau_dag = tau[:, 0].conj()                              # (2,)
            psi_k   = np.einsum('m,...mn->...n', tau_dag, Pband)    # (Nx,Ny,Rx,Ry,2)

            # Broadcast psi_k components over centers, then FFT over k-axes
            F0 = phase * psi_k[..., 0]  # (Nx,Ny,Rx,Ry)
            F1 = phase * psi_k[..., 1]  # (Nx,Ny,Rx,Ry)
            W0 = k2_to_r2(F0)           # (Nx,Ny,Rx,Ry)
            W1 = k2_to_r2(F1)           # (Nx,Ny,Rx,Ry)

            # Stack μ=0,1 as the 3rd axis → (Nx,Ny,2,Rx,Ry)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2) # (Nx,Ny,2,Rx,Ry)

            # --- Helper: construct a square window mask for truncation ---
            def _square_window_mask(nshell):
                # Returns mask of shape (Nx, Ny, Rx, Ry): True if (x,y) within window of nshell around (Rx,Ry)
                x = np.arange(Nx)[:, None, None, None]  # (Nx,1,1,1)
                y = np.arange(Ny)[None, :, None, None]  # (1,Ny,1,1)
                Rx = np.arange(Nx)[None, None, :, None] # (1,1,Rx,1)
                Ry = np.arange(Ny)[None, None, None, :] # (1,1,1,Ry)
                dx_wrap = ((x - Rx + Nx//2) % Nx) - Nx//2  # (Nx,1,Rx,1)
                dy_wrap = ((y - Ry + Ny//2) % Ny) - Ny//2  # (1,Ny,1,Ry)
                mask = (np.abs(dx_wrap) <= nshell) & (np.abs(dy_wrap) <= nshell)  # (Nx,Ny,Rx,Ry)
                return mask
            
            # If truncation is requested, apply window and renormalize per center
            if nshell is not None:
                mask = _square_window_mask(nshell)  # (Nx,Ny,Rx,Ry)
                # Expand mask to (Nx,Ny,1,Rx,Ry) to match W (Nx,Ny,2,Rx,Ry)
                mask_exp = mask[:, :, None, :, :]  # (Nx,Ny,1,Rx,Ry)
                W = W * mask_exp  # (Nx,Ny,2,Rx,Ry)
                # Compute norm per center (Rx,Ry), sum over (x,y,μ)
                norm2 = np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)  # (1,1,1,Rx,Ry)
                # Avoid division by zero: only normalize where norm2 > 1e-15
                norm_mask = (norm2 > 1e-15)
                W_normed = np.zeros_like(W)
                W_normed[..., :, :] = W  # default (will overwrite below)
                # Only normalize where norm2 > 1e-15
                W_normed = np.where(norm_mask, W / (np.sqrt(norm2) + 1e-15), W)
                return W_normed
            else:
                # Normalize per center (R_x,R_y) across (x,y,μ)
                denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
                return W / denom
        
        
        # Build the four overcomplete Wannier spinors for ALL centers: (Nx,Ny,2,Rx,Ry)
        W_Ap  = make_W(self.Pplus, tauA, phase)
        W_Bp  = make_W(self.Pplus, tauB, phase)
        W_Am = make_W(self.Pminus, tauA, phase)
        W_Bm = make_W(self.Pminus, tauB, phase)

        def flatten_centers(W):
            # W: (Nx, Ny, 2, Rx, Ry)
            # Move (μ, x, y) up front => (2, Nx, Ny, Rx, Ry)
            W_mu_xy = np.transpose(W, (2, 0, 1, 3, 4))
            # Flatten (μ, x, y) into D = 2*Nx*Ny, keeping (Rx, Ry)
            return W_mu_xy.reshape(2 * Nx * Ny, Nx, Ny, order='F')

        self.WF_Ap = flatten_centers(W_Ap)  # (D=2*Nx*Ny, Rx, Ry)
        self.WF_Bp = flatten_centers(W_Bp)
        self.WF_Am = flatten_centers(W_Am)
        self.WF_Bm = flatten_centers(W_Bm)

        # Projectors per center: χ χ†  → (D,D,Rx,Ry)
        def projectors(WF):   # WF: (D, Rx, Ry)
            return np.einsum('dxy, exy->dexy', WF, WF.conj(), optimize=True)

        self.P_Ap = projectors(self.WF_Ap)
        self.P_Bp = projectors(self.WF_Bp)
        self.P_Am = projectors(self.WF_Am)
        self.P_Bm = projectors(self.WF_Bm)

    def measure_bottom_layer(self, G, P, particle = True, symmetrize=True):
        """
        Meausure bottom layer using a Charge-Conserving EPR-state via the Choi isomorphism.

        Inputs
        ------
        G : (4*Nx*Ny, 4*Nx*Ny) complex ndarray
            Complex fermion covariance on both layers (Hermitian).
        P : (2*Nx*Ny, 2*Nx*Ny) complex ndarray
            Projector (Hermitian, idempotent: P^2 = P). Specifies the measured subspace.
        occupied : bool
            If True, measure P (occupied). If False, measure (I - P) i.e. "unoccupied".
        rcond : float
            Tikhonov regularization magnitude used only if (H11 - inv(G)) is near singular.
        symmetrize : bool
            If True, return (G' + G'^†)/2 to clean up numerical noise.

        Returns
        -------
        G_prime : (4*Nx*Ny, 4*Nx*Ny) complex ndarray
            Updated covariance.
        """
        Ntot = self.Ntot
        Nlayer = np.floor(Ntot/2)
        G = np.asarray(G)
        P = np.asarray(P)
        if G.shape != (Ntot, Ntot):
            raise ValueError(f"G must have shape {(Ntot, Ntot)}, "
                             f"got {G.shape}")
        elif P.shape != (Nlayer, Nlayer):
            raise ValueError(f"P must have shape {(Nlayer, Nlayer)}, "
                             f"got {P.shape}")

        Ilayer = np.eye(Nlayer, dtype=P.dtype) # Identity matrix over a layer subspace

        Gtt = G[0:Nlayer, 0:Nlayer]
        Gbb = G[Nlayer:, Nlayer:]
        Gtb = G[0:Nlayer, Nlayer:]

        # Build H blocks
        if particle:
            # occupied: H = [[-P, I-P], [I-P, P]]
            H11 = -P
            H21 = Ilayer - P
            H22 = P
        else:
            # Unoccupied: H = [[P, I-P], [I-P, -P]]
            H11 = P
            H21 = Ilayer - P
            H22 = -P

        # Build inverse
        K = np.block([[Gbb, -Ilayer],[-Ilayer, H11]])
        L = np.block_diag(Gtb, H21)
        invK_Ldag = np.linalg.solve(K, L.conj().T) # solve for inv(K) @ L^\dagger        

        # update G
        M = np.block_diag(Gtt, H22)
        G_prime = M - L @ invK_Ldag

        if symmetrize:
            G_prime = 0.5 * (G_prime + G_prime.conj().T)

        return G_prime
    
    def measure_top_layer(self, G, P, particle = True, symmetrize=True):
        """
        Meausure top layer using a Charge-Conserving EPR-state via the Choi isomorphism.

        Inputs
        ------
        G : (4*Nx*Ny, 4*Nx*Ny) complex ndarray
            Complex fermion covariance on both layers (Hermitian).
        P : (2*Nx*Ny, 2*Nx*Ny) complex ndarray
            Projector (Hermitian, idempotent: P^2 = P). Specifies the measured subspace.
        occupied : bool
            If True, measure P (occupied). If False, measure (I - P) i.e. "unoccupied".
        rcond : float
            Tikhonov regularization magnitude used only if (H11 - inv(G)) is near singular.
        symmetrize : bool
            If True, return (G' + G'^†)/2 to clean up numerical noise.

        Returns
        -------
        G_prime : (4*Nx*Ny, 4*Nx*Ny) complex ndarray
            Updated covariance.
        """
        Ntot = self.Ntot
        Nlayer = Ntot//2
        G = np.asarray(G)
        P = np.asarray(P)
        if G.shape != (Ntot, Ntot):
            raise ValueError(f"G must have shape {(Ntot, Ntot)}, "
                             f"got {G.shape}")
        elif P.shape != (Nlayer, Nlayer):
            raise ValueError(f"P must have shape {(Nlayer, Nlayer)}, "
                             f"got {P.shape}")

        Ilayer = np.eye(Nlayer, dtype=P.dtype) # Identity matrix over a layer subspace

        Gtt = G[0:Nlayer, 0:Nlayer]
        Gbb = G[Nlayer:, Nlayer:]
        Gbt = G[Nlayer:, 0:Nlayer]

        # Build H blocks
        if particle:
            # occupied: H = [[-P, I-P], [I-P, P]]
            H11 = -P
            H21 = Ilayer - P
            H22 = P
        else:
            # Unoccupied: H = [[P, I-P], [I-P, -P]]
            H11 = P
            H21 = Ilayer - P
            H22 = -P

        # Build inverse
        K = np.block([[H11, -Ilayer],[-Ilayer, Gtt]])
        L = np.block_diag(H21, Gbt)
        invK_Ldag = np.linalg.solve(K, L.conj().T) # solve for inv(K) @ L^\dagger        

        # update G
        M = np.block_diag(H22, Gbb)
        G_prime = M - L @ invK_Ldag

        if symmetrize:
            G_prime = 0.5 * (G_prime + G_prime.conj().T)

        return G_prime
    
    def fSWAP(self, chi_top, chi_bottom):
        '''
            Construct an fSWAP unitary to swap local modes in the top and bottom layers
        '''

        Htt = chi_top[:, None].conj() * chi_top[None, :] 
        Htb = chi_top[:, None].conj() * chi_bottom[None, :]
        Hbb = chi_bottom[:, None].conj() * chi_bottom[None, :] 

        H = -(np.pi/2)*np.block([[Htt, -Htb],[-Htb.conj().T, Hbb]])

        D, V = np.linalg.eigh(H)
        return V @ np.diag(np.exp(1j*D)) @ V.conj().T
    
    def top_layer_meas_feedback(self, G, Rx, Ry):
        '''
            Run measurement-feedback for a single choice of (Rx, Ry)
        '''
        Ntot = self.Ntot
        Nx = self.Nx
        Nlayer = Ntot//2
        Ilayer = np.eye(Nlayer)
        Gtt_2pt = (G[0:Nlayer, 0:Nlayer]+Ilayer)/2 # return two-point <c^\dagger_i c_j> for top layer
        

        P_A_plus = self.P_Ap[:,Rx,Ry]
        P_B_plus = self.P_Bp[:,Rx,Ry]
        P_A_minus = self.P_Am[:,Rx,Ry]
        P_B_minus = self.P_Bm[:,Rx,Ry]

        Chi_A_plus_top = self.WF_Ap[:,Rx,Ry]
        Chi_B_plus_top = self.WF_Bp[:,Rx,Ry]
        Chi_A_minus_top = self.WF_Am[:,Rx,Ry]
        Chi_B_minus_top = self.WF_Bm[:,Rx,Ry]

        Chi_A_bottom = np.eye(Nlayer)[0 + 2*Rx + 2*Nx*Ry] # 1-hot vector for orbital A in bottom layer unit cell (Rx, Ry)
        Chi_B_bottom = np.eye(Nlayer)[1 + 2*Rx + 2*Nx*Ry]

        # 1) check if upper band mode A is unoccupied
        Born_A_plus = np.trace(Gtt_2pt @ P_A_plus)
        p = np.random.rand()
        if p < Born_A_plus:
            G = self.measure_top_layer(G, P_A_plus, particle = False) # upper band mode unoccupied
        else:
            G = self.measure_top_layer(G, P_A_plus, particle = True) # upper band mode occupied, SWAP out charge
            fSWAP = self.fSWAP(Chi_A_plus_top,Chi_A_bottom)
            G = fSWAP @ G @ fSWAP

        # 2) check if upper band mode B is unoccupied
        Born_B_plus = np.trace(Gtt_2pt @ P_B_plus)
        p = np.random.rand()
        if p < Born_B_plus:
            G = self.measure_top_layer(G, P_B_plus, particle = False) # upper band mode unoccupied
        else:
            G = self.measure_top_layer(G, P_B_plus, particle = True) # upper band mode occupied, SWAP out charge
            fSWAP = self.fSWAP(Chi_B_plus_top, Chi_B_bottom)
            G = fSWAP @ G @ fSWAP
        
        # 3) check if lower band mode A is occupied
        Born_A_minus = np.trace(Gtt_2pt @ P_A_minus)
        p = np.random.rand()
        if p > Born_A_minus:
            G = self.measure_top_layer(G, P_A_minus, particle = True) # lower band mode occupied
        else:
            G = self.measure_top_layer(G, P_A_minus, particle = False) # lower band mode unoccupied, SWAP out Charge
            fSWAP = self.fSWAP(Chi_A_minus_top, Chi_A_bottom)
            G = fSWAP @ G @ fSWAP

        # 4) check if lower band mode B is occupied
        Born_B_minus = np.trace(Gtt_2pt @ P_B_minus)
        p = np.random.rand()
        if p > Born_B_minus:
            G = self.measure_top_layer(G, P_B_minus, particle = False) # upper band mode unoccupied
        else:
            G = self.measure_top_layer(G, P_B_minus, particle = True) # upper band mode occupied
            fSWAP = self.fSWAP(Chi_B_minus_top, Chi_B_bottom)
            G = fSWAP @ G @ fSWAP

        return G
    
    def measure_all_bottom_modes(self, G):
        '''
            Measure all local mode occupancies in bottom layer
        '''

        Ntot = self.Ntot
        Nlayer = Ntot//2
        Ilayer = np.eye(Nlayer)

        for idx in range(Nlayer):
            chi_bott = Ilayer[idx]
            P_bott = chi_bott[:,None]*chi_bott[None,:]
            Gbb_2pt = (G[Nlayer:, Nlayer:]+Ilayer)/2 # return two-point <c^\dagger_i c_j> for bott layer
            Born_bott = np.trace(Gbb_2pt @ P_bott)
            p = np.random.rand()
            if p > Born_bott: 
                G = self.measure_bottom_layer(G, P_bott, particle = True) # bottom mode occupied
            else:
                G = self.measure_bottom_layer(G, P_bott, particle = False) # bottom mode unoccupied
        
        return G
    
    def randomize_bottom_layer(self, G):
        '''
            Randomize bottom layer with a random global unitary acting only on bottom layer modes
        '''
        Ntot = self.Ntot
        Nlayer = Ntot//2
        Ilayer = np.eye(Nlayer)

        U_bott = self.random_unitary(Nlayer)
        U_tot = np.block_diag(Ilayer, U_bott)

        G = U_tot.conj().T @ G @ U_tot

        return G
    
    def adaptive_cycle(self, G):
        '''
            Execute a single adaptive cycle for across all unit cells (Rx, Ry)
        '''

        for Rx in range(self.Nx):
            for Ry in range(self.Ny):
                G = self.top_layer_meas_feedback(G, Rx, Ry)
                G = self.randomize_bottom_layer(G)
                G = self.measure_all_bottom_modes(G)
        
        return G

    def run_adaptive_circuit(self, cycles=5, G_history=True, tol=1e-8, progress=True):
        """Execute repeated adaptive sweeps with nested progress tracking."""
        if G_history:
            self.G_list = []
        self.g2_flags = []

        self.G = self.G0
        Nx, Ny = int(self.Nx), int(self.Ny)
        D = int(self.Ntot)
        I = np.eye(D, dtype=complex)

        global_iter = 0

        outer_iter = range(cycles)
        if progress:
            outer_iter = tqdm(outer_iter, desc="Cycles", total=cycles, leave=True)

        for c in outer_iter:
            inner_iter = range(Nx)
            if progress:
                inner_iter = tqdm(inner_iter,
                                   desc=f"Sweep {c+1}/{cycles}",
                                   total=Nx,
                                   leave=False)

            for Rx in inner_iter:
                row_start = time.time()

                site_iter = range(Ny)
                if progress:
                    site_iter = tqdm(site_iter,
                                      desc=f"Row {Rx+1}/{Nx}",
                                      total=Ny,
                                      leave=False)

                for Ry in site_iter:
                    self.G = self.top_layer_meas_feedback(self.G, Rx, Ry)
                    self.G = self.randomize_bottom_layer(self.G)
                    self.G = self.measure_all_bottom_modes(self.G)

                    g2_ok = int(np.allclose(self.G @ self.G, I, atol=tol))
                    self.g2_flags.append(g2_ok)

                    global_iter += 1
                    if progress:
                        inner_post = {
                            "cycle": f"{c+1}/{cycles}",
                            "pos": f"({Rx},{Ry})",
                            "iter": global_iter,
                            "G2==I": g2_ok
                        }
                        if isinstance(site_iter, tqdm):
                            site_iter.set_postfix(inner_post, refresh=False)
                        if isinstance(outer_iter, tqdm):
                            outer_iter.set_postfix(inner_post, refresh=False)

                if progress and isinstance(site_iter, tqdm):
                    site_iter.close()

                if progress and isinstance(inner_iter, tqdm):
                    elapsed = time.time() - row_start
                    inner_iter.set_postfix({"row_sec": f"{elapsed:.2f}"}, refresh=False)

            if G_history:
                self.G_list.append(self.G.copy())


