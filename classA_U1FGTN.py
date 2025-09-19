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
    
    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path

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

        Nfill = int(round(filling_frac*N))
        Nempty = N - Nfill
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
        W_Ap = make_W(self.Pplus, tauA, phase)
        W_Bp = make_W(self.Pplus, tauB, phase)
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
            G = self.measure_top_layer(G, P_A_minus, particle = False) # lower band mode unoccupied, SWAP in Charge
            fSWAP = self.fSWAP(Chi_A_minus_top, Chi_A_bottom)
            G = fSWAP @ G @ fSWAP

        # 4) check if lower band mode B is occupied
        Born_B_minus = np.trace(Gtt_2pt @ P_B_minus)
        p = np.random.rand()
        if p > Born_B_minus:
            G = self.measure_top_layer(G, P_B_minus, particle = False) # lower band mode occupied
        else:
            G = self.measure_top_layer(G, P_B_minus, particle = True) # lower band mode unoccupied, SWAP in Charge
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
    
    
    def run_adaptive_circuit(self, cycles=5, G_history=True, tol=1e-8, progress=True):
        """
        Run the adaptive circuit for `cycles` sweeps over the lattice.
        - Shows nested tqdm bars (outer=cycles, inner=Nx*Ny) with ETA.
        - Postfix displays current (cycle, Rx, Ry), global iter, and G^2==I flag.
        - Records G history (optional) and per-iteration G^2 flags in self.g2_flags.

        Parameters
        ----------
        cycles : int
            Number of full lattice sweeps.
        G_history : bool
            If True, appends G after each cycle to self.G_list.
        tol : float
            Tolerance for checking G^2 ≈ I (np.allclose with atol=tol).
        progress : bool
            If True, show tqdm progress bars.
        """
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
                inner_iter = tqdm(inner_iter, desc=f"Sweep {c+1}/{cycles}", total=Nx, leave=False)

            for Rx in inner_iter:
                # optional: time per-row if you want your own timer (tqdm already shows ETA)
                row_start = time.time()

                for Ry in range(Ny):
                    # Core steps
                    self.G = self.top_layer_meas_feedback(self.G, Rx, Ry)
                    self.G = self.randomize_bottom_layer(self.G)
                    self.G = self.measure_all_bottom_modes(self.G)

                    # G^2 ≈ I check
                    g2_ok = int(np.allclose(self.G @ self.G, I, atol=tol))
                    self.g2_flags.append(g2_ok)

                    global_iter += 1
                    if progress:
                        # Update postfix with current position and G^2 flag
                        tqdm.tqdm.write if False else None  # (placeholder to avoid lint warnings)
                        inner_post = {
                            "cycle": f"{c+1}/{cycles}",
                            "pos": f"({Rx},{Ry})",
                            "iter": global_iter,
                            "G2==I": g2_ok
                        }
                        # Show on the *outer* bar too for visibility
                        if isinstance(outer_iter, tqdm):
                            outer_iter.set_postfix(inner_post, refresh=False)

                # (optional) per-row timing in postfix
                if progress and isinstance(inner_iter, tqdm):
                    elapsed = time.time() - row_start
                    inner_iter.set_postfix({"row_sec": f"{elapsed:.2f}"}, refresh=False)

            if G_history:
                self.G_list.append(self.G.copy())

    
    def real_space_chern_number(self, G=None):
        """
        Compute the real-space Chern number using the projector built from the two-point function G-.

        This implements the same formula as `chern_from_projector`, but takes the
        input as a general operator G and forms P = G.conj() before the block traces.

        Parameters
        ----------
        G : ndarray, shape (4*Nx*Ny, 4*Nx*Ny)
            Real-space operator. The projector used here is P = G.conj().
        A_mask, B_mask, C_mask : (Nx,Ny) boolean masks, optional
            If None, use the instance's stored tri-partition masks.

        Returns
        -------
        complex
            12π i [ Tr(P_CA P_AB P_BC) - Tr(P_AC P_CB P_BA) ].
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
        Ntot = 2*Nx*Ny*2
        Nlayer = Ntot//2

        # --- Build tri-partition masks A, B, C inside a circle of radius R ---
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

            A_mask[x0:x1+1, y] = (theta >= 0)   & (theta < a2)
            B_mask[x0:x1+1, y] = (theta >= a2)  & (theta < a4)
            C_mask[x0:x1+1, y] = (theta >= a4)  & (theta < 2*np.pi)
        
        if G is None:
            G = self.G
        
        Ilayer = np.eye(Nlayer)
        G_2pt = (1/2)*(Ilayer + G[0:Nlayer,0:Nlayer])


        P = G_2pt.conj()

        def sector_indices_from_mask(mask):
            """
            Flatten (μ, x, y) with i = μ + 2*x + 2*Nx*y (μ=0,1).
            Returns indices for both orbitals at each (x,y) where mask[x,y] is True.
            """
            mask = np.asarray(mask, dtype=bool)
            xs, ys = np.nonzero(mask)
            idx0 = 0 + 2*xs + 2*Nx*ys
            idx1 = 1 + 2*xs + 2*Nx*ys
            return np.sort(np.concatenate([idx0, idx1]))
        
        # Sector indices include both orbitals for each selected site
        iA = sector_indices_from_mask(A_mask)
        iB = sector_indices_from_mask(B_mask)
        iC = sector_indices_from_mask(C_mask)

        P_CA = P[np.ix_(iC, iA)]
        P_AB = P[np.ix_(iA, iB)]
        P_BC = P[np.ix_(iB, iC)]

        P_AC = P[np.ix_(iA, iC)]
        P_CB = P[np.ix_(iC, iB)]
        P_BA = P[np.ix_(iB, iA)]

        t1 = np.trace(P_CA @ P_AB @ P_BC)
        t2 = np.trace(P_AC @ P_CB @ P_BA)

        Y = 12 * np.pi * 1j * (t1 - t2)
        return np.real_if_close(Y, tol=1e-6)
    

    def local_chern_marker_flat(self, G=None, mask_outside=False, inside_mask=None):
        """
        Local Chern marker from a FLAT top-layer covariance using the reshaped kernel.

        Accepts:
          - G is None          -> uses current full self.G, top-layer block is taken
          - G shape (4N,4N)    -> uses its top-layer block
          - G shape (2N,2N)    -> treated as the flat top-layer block directly

        Steps:
          1) Two-point: G2 = (G_flat + I)/2
          2) Reshape G2 -> (2, Nx, Ny, 2, Nx, Ny) with order='F', then permute to (Nx,Ny,2,Nx,Ny,2)
          3) Projector: P = G2.conj()
          4) M = (2π i) [ P X P Y P − P Y P X P ], with X,Y non-modular (1..Nx, 1..Ny) on RIGHT indices
          5) C(x,y) = sum_μ diag(M) at (x,y,μ)
          6) Return tanh(Re C(x,y))
        """
    
        Nx, Ny = int(self.Nx), int(self.Ny)
        Nlayer = 2 * Nx * Ny

        # --- pick flat top-layer block ---
        if G is None:
            Gflat = np.asarray(self.G)[:Nlayer, :Nlayer]
        else:
            G = np.asarray(G)
            if G.shape == (self.Ntot, self.Ntot):
                Gflat = G[:Nlayer, :Nlayer]
            elif G.shape == (Nlayer, Nlayer):
                Gflat = G
            else:
                raise ValueError(f"G must be ({self.Ntot},{self.Ntot}) or ({Nlayer},{Nlayer}), got {G.shape}")

        # --- two-point & reshape to kernel (Nx,Ny,2, Nx,Ny,2) ---
        I = np.eye(Nlayer, dtype=complex)
        G2_flat = 0.5 * (Gflat + I)

        # i = μ + 2*x + 2*Nx*y  (μ fastest, then x, then y) -> use Fortran order
        G2_6 = G2_flat.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
        # put as (x,y,μ | x',y',ν)
        G2_6 = np.transpose(G2_6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,2, Nx,Ny,2)

        # projector kernel
        P = G2_6.conj()

        # --- right multiplications by non-modular X,Y (1..Nx, 1..Ny) ---
        X = np.arange(1, Nx + 1, dtype=float)
        Y = np.arange(1, Ny + 1, dtype=float)
        Xright = X[None, None, None, :, None, None]   # broadcast on right x'
        Yright = Y[None, None, None, None, :, None]   # broadcast on right y'

        def right_X(A): return A * Xright
        def right_Y(A): return A * Yright

        # contraction over right indices (x',y',ν): (i,j,s,l,m,n) × (l,m,n,o,p,r) -> (i,j,s,o,p,r)
        mm = lambda A, B: np.einsum('ijslmn,lmnopr->ijsopr', A, B, optimize=True)

        # BR kernel: M = (2π i) (P X P Y P − P Y P X P)
        T = right_X(P); T = mm(T, P); T = right_Y(T); T = mm(T, P)  # P X P Y P
        U = right_Y(P); U = mm(U, P); U = right_X(U); U = mm(U, P)  # P Y P X P
        M = (2.0 * np.pi * 1j) * (T - U)                            # (Nx,Ny,2, Nx,Ny,2)

        # --- take diagonal (x,y,μ; x,y,μ) and sum over μ ---
        ix = np.arange(Nx)[:, None, None]
        iy = np.arange(Ny)[None, :, None]
        ispin = np.arange(2)[None, None, :]
        diag_vals = M[ix, iy, ispin, ix, iy, ispin]  # (Nx,Ny,2)
        C = diag_vals.sum(axis=2)                    # (Nx,Ny)

        # stabilize & optional mask
        C = np.tanh(np.real_if_close(C, tol=1e-9))
        if mask_outside and inside_mask is not None:
            C = np.where(inside_mask, C, 0.0)
        return C
    
    def chern_marker_dynamics(self, outbasename=None, vmin=-1.0, vmax=1.0, cmap='RdBu_r'):
        """
        Animate the local Chern marker C(r) over time using the cached history self.G_list
        (if present); otherwise, show a single frame from the current self.G.
    
        - Saves a GIF at 1 fps and a static final frame (PDF).
        - Returns paths plus the final arrays.
    
        Returns
        -------
        gif_path : str
            Path to the saved GIF (may be single-frame if no history).
        final_path : str
            Path to the saved static final frame (PDF).
        C_last : ndarray, shape (Nx, Ny)
            Local Chern marker of the final frame.
        G_last : ndarray
            The final covariance used for the last frame.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
    
        # Output paths
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            outbasename = f"chern_marker_dynamics_N{Nx}"
        gif_path   = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.pdf")
    
        # Pick history (prefer recorded list; else single frame from current state)
        if hasattr(self, "G_list") and isinstance(self.G_list, list) and len(self.G_list) > 0:
            history = self.G_list
            G_last  = history[-1]
        else:
            if not hasattr(self, "G"):
                raise RuntimeError("No state available: run the circuit to populate self.G or self.G_list.")
            history = [self.G]
            G_last  = self.G
    
        # --- Figure setup ---
        fig = plt.figure(figsize=(3.2, 3.8))
        ax  = fig.add_axes([0.12, 0.10, 0.78, 0.78])
        im  = ax.imshow(np.zeros((Nx, Ny)), cmap=cmap, vmin=vmin, vmax=vmax,
                        origin='upper', aspect='equal')
    
        # Styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.5); spine.set_color('black')
        ax.set_xlabel("y"); ax.set_ylabel("x")
        ax.set_xticks(np.arange(0, Ny, max(1, Ny//10)))
        ax.set_yticks(np.arange(0, Nx, max(1, Nx//10)))
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)
    
        cax = fig.add_axes([0.32, 0.90, 0.36, 0.06])
        fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
        fig.text(0.70, 0.91, r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=12)
    
        # First frame
        C0 = self.local_chern_marker_flat(history[0])
        im.set_data(C0)
    
        # --- Write GIF at 1 fps ---
        writer = animation.PillowWriter(fps=1)
        with writer.saving(fig, gif_path, dpi=120):
            for G_frame in tqdm(history, desc="chern_marker_frames", unit="frame"):
                C = self.local_chern_marker_flat(G_frame)
                im.set_data(C)
                writer.grab_frame()
        plt.close(fig)
    
        # Final arrays
        C_last = np.round(C, 2)  # last computed in loop
        G_last = history[-1]
    
        # Save static final frame
        fig2 = plt.figure(figsize=(3.2, 3.8))
        ax2  = fig2.add_axes([0.12, 0.10, 0.78, 0.78])
        im2  = ax2.imshow(C_last, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', aspect='equal')
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5); spine.set_color('black')
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
    
        fig2.savefig(final_path, bbox_inches='tight')
        plt.show()
        plt.close(fig2)
    
        return gif_path, final_path, C_last, G_last

    def plot_corr_y_profiles(self, G=None, x_positions=None, ry_max=None, filename=None,
                             samples=1, trajectory_resolved=False, trajectory_averaged=False):
        """
        Plot squared two-point correlation vs y-separation at selected x columns.

        Averaging modes:
          - trajectory_resolved=True, trajectory_averaged=False:
              run 'samples' independent circuits; for each, compute C(x0, ry),
              then average C over samples → label: \bar{C}_G(x_0, r_y)
          - trajectory_averaged=True, trajectory_resolved=False:
              run 'samples' independent circuits; average the resulting top-layer G,
              then compute C once from the averaged G → label: C_{\bar{G}}(x_0, r_y)
          - neither True:
              run one circuit (if G is None) and compute C(x0, ry) → label: C_G(x_0, r_y)

        If both flags are True, raises ValueError.

        If G is provided, it is used directly (no circuit run). G may be (4N,4N), (2N,2N),
        or already a kernel (Nx,Ny,2,Nx,Ny,2). Only the top-layer block is used if needed.
        """

        if trajectory_resolved and trajectory_averaged:
            raise ValueError("Choose only one averaging mode: set either "
                             "`trajectory_resolved=True` or `trajectory_averaged=True` (not both).")

        Nx, Ny = int(self.Nx), int(self.Ny)
        Ntot = int(self.Ntot)
        Nlayer = Ntot // 2
        samples = int(max(1, samples))

        # ---------- helpers ----------
        def _coerce_to_two_point_kernel_top(G_in):
            """
            Accepts G_in as (4N,4N), (2N,2N), or (Nx,Ny,2,Nx,Ny,2) and returns
            the 6-index top-layer two-point kernel: (Nx,Ny,2,Nx,Ny,2).
            Uses i = μ + 2*x + 2*Nx*y (Fortran reshape).
            """
            Gin = np.asarray(G_in)
            if Gin.ndim == 6:
                # Assume already two-point kernel (Nx,Ny,2,Nx,Ny,2)
                if Gin.shape[:2] != (Nx, Ny) or Gin.shape[2] != 2 or Gin.shape[3:5] != (Nx, Ny) or Gin.shape[5] != 2:
                    raise ValueError("6-index G has incompatible shape.")
                return Gin
            elif Gin.ndim == 2:
                if Gin.shape == (Ntot, Ntot):
                    # take top-layer block first
                    Gtt = Gin[:Nlayer, :Nlayer]
                elif Gin.shape == (Nlayer, Nlayer):
                    Gtt = Gin
                else:
                    raise ValueError(f"G has incompatible shape {Gin.shape}.")
                I = np.eye(Nlayer, dtype=complex)
                G2 = 0.5 * (Gtt + I)  # two-point on top layer
                G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
                G6 = np.transpose(G6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,2,Nx,Ny,2)
                return G6
            else:
                raise ValueError("G must be (4N,4N), (2N,2N), or (Nx,Ny,2,Nx,Ny,2).")

        def _C_xslice_from_kernel(Gker, x0, ry_vals):
            """
            Vectorized C(x0; ry) from kernel (Nx,Ny,2,Nx,Ny,2) for all ry in ry_vals.
            C = (1/(2 Ny)) sum_{y,μ,ν} |G[(x0,y,μ),(x0,y+ry,ν)]|^2
            """
            x0 = int(x0) % Nx
            ry_arr = np.atleast_1d(ry_vals).astype(int)
            Ny_loc = Gker.shape[1]

            # Gx: (Ny, 2, Ny, 2)
            Gx = Gker[x0, :, :, x0, :, :]
            Y = np.arange(Ny_loc, dtype=np.intp)[:, None]     # (Ny,1)
            Yp = (Y + ry_arr[None, :]) % Ny_loc               # (Ny,R)

            # Reorder -> (Ny*Ny, 2, 2), gather blocks by flat (y, yp)
            Gx_re = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny_loc*Ny_loc, 2, 2)
            flat_idx = (Y * Ny_loc + Yp).reshape(-1)          # (Ny*R,)
            blocks = Gx_re[flat_idx].reshape(Ny_loc, ry_arr.size, 2, 2)   # (Ny,R,2,2)

            C = np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny_loc)  # (R,)
            return C

        # ---------- choose x-positions (smart default from domain walls) ----------
        def _smart_x_positions_from_alpha(alpha2d):
            a = np.asarray(alpha2d)
            if a.shape != (Nx, Ny):
                # fallback: evenly spaced
                xs = np.unique(np.clip(np.array([Nx//6, Nx//2, 5*Nx//6]), 0, Nx-1))
                return [(int(x), f"x0={int(x)}") for x in xs]

            col_vals = a[:, 0]  # uniform in y by construction
            topo_mask = np.isclose(col_vals, 1.0, atol=1e-12)
            if not np.any(topo_mask) or np.all(topo_mask):
                xs = np.unique(np.clip(np.array([Nx//6, Nx//2, 5*Nx//6]), 0, Nx-1))
                return [(int(x), f"x0={int(x)}") for x in xs]

            topo_idx = np.where(topo_mask)[0]
            diffs = (np.diff(np.r_[topo_idx, topo_idx[0] + Nx]) == 1)
            breaks = np.where(~diffs)[0]
            if breaks.size == 0:
                xs = np.unique(np.clip(np.array([Nx//6, Nx//2, 5*Nx//6]), 0, Nx-1))
                return [(int(x), f"x0={int(x)}") for x in xs]
            start_pos = (breaks[0] + 1) % topo_idx.size
            ordered = np.r_[topo_idx[start_pos:], topo_idx[:start_pos]]
            # longest contiguous run (we built exactly one)
            run = [ordered[0]]
            for k in range(1, ordered.size):
                if (ordered[k] - ordered[k-1]) % Nx == 1:
                    run.append(ordered[k])
                else:
                    break
            run = np.array(run, dtype=int)

            x0 = run[0]
            x1 = (run[-1] + 1) % Nx
            topo_center = run[len(run)//2]
            triv_left_center  = (x0 // 2)
            triv_right_center = ((x1 + Nx) // 2) % Nx
            x_wall_L = (x0 - 1) % Nx
            x_wall_R = x1 % Nx

            picks = [
                (int(x_wall_L),       "wall L"),
                (int(x_wall_R),       "wall R"),
                (int(topo_center),    "topo bulk"),
                (int(triv_left_center),  "trivial L"),
                (int(triv_right_center), "trivial R"),
            ]
            uniq, seen = [], set()
            for x, lab in picks:
                x = int(x % Nx)
                if x not in seen:
                    uniq.append((x, lab))
                    seen.add(x)
            return uniq

        # default y-separations
        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # normalize x_positions
        if x_positions is None:
            if hasattr(self, "alpha"):
                norm_positions = _smart_x_positions_from_alpha(self.alpha)
            else:
                xs = np.arange(0, Nx, max(1, Nx // 8))
                norm_positions = [(int(x), f"x0={int(x)}") for x in xs]
        else:
            norm_positions = []
            try:
                for x0, label in x_positions:
                    norm_positions.append((int(x0) % Nx, str(label)))
            except Exception:
                for x0 in np.atleast_1d(x_positions):
                    norm_positions.append((int(x0) % Nx, f"x0={int(x0)%Nx}"))

        # ---------- decide what to compute ----------
        if G is not None:
            # use provided G as-is; no evolution
            Gker = _coerce_to_two_point_kernel_top(G)
            mode_label = r"$C_G(x_0,r_y)$"
            C_dict = {}
            for x0, _ in norm_positions:
                C_dict[x0] = _C_xslice_from_kernel(Gker, x0, ry_vals)
        else:
            # need to evolve
            if trajectory_resolved:
                # average C across samples: \bar{C}_G
                C_accum = {x0: np.zeros_like(ry_vals, dtype=float) for x0, _ in norm_positions}
                for s in range(samples):
                    self.run_adaptive_circuit(cycles=self.cycles, G_history=False, progress=False)
                    Gker = _coerce_to_two_point_kernel_top(self.G)
                    for x0, _ in norm_positions:
                        C_accum[x0] += _C_xslice_from_kernel(Gker, x0, ry_vals).real
                C_dict = {x0: C_accum[x0] / samples for x0, _ in norm_positions}
                mode_label = r"$\overline{C}_G(x_0,r_y)$"
            elif trajectory_averaged:
                # average G across samples, then compute C once: C_{\bar{G}}
                Gsum = np.zeros((Nlayer, Nlayer), dtype=complex)
                for s in range(samples):
                    self.run_adaptive_circuit(cycles=self.cycles, G_history=False, progress=False)
                    Gsum += np.asarray(self.G)[:Nlayer, :Nlayer]
                Gavg = Gsum / samples
                Gker = _coerce_to_two_point_kernel_top(Gavg)
                C_dict = {x0: _C_xslice_from_kernel(Gker, x0, ry_vals).real for x0, _ in norm_positions}
                mode_label = r"$C_{\overline{G}}(x_0,r_y)$"
            else:
                # single trajectory: C_G
                self.run_adaptive_circuit(cycles=self.cycles, G_history=False, progress=False)
                Gker = _coerce_to_two_point_kernel_top(self.G)
                C_dict = {x0: _C_xslice_from_kernel(Gker, x0, ry_vals).real for x0, _ in norm_positions}
                mode_label = r"$C_G(x_0,r_y)$"

        # ---------- plot ----------
        outdir = self._ensure_outdir('figs/corr_y_profiles')
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for x0, lbl in norm_positions:
            C_vec = C_dict[x0]
            line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=lbl)
            # inline label near right edge
            finite = np.isfinite(C_vec)
            y_right = C_vec[finite][-1] if np.any(finite) else C_vec[-1]
            x_right = ry_vals[-1] * 1.02 if ry_vals[-1] > 0 else ry_vals[-1] + 0.5
            ax.annotate(lbl, xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                        textcoords='data', ha='left', va='center', fontsize=9,
                        color=line.get_color())

        ax.set_xlabel(r"$r_y$")
        ax.set_ylabel(mode_label)
        ax.set_title(f"Squared correlator vs $r_y$ at selected $x_0$ (N={Nx})")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),
            ncol=3,
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            borderpad=0.4,
            handlelength=1.5,
            handletextpad=0.6,
            columnspacing=0.9,
            labelspacing=0.3
        )
        fig.tight_layout()

        if filename is None:
            xdesc = "-".join(f"{x}" for x, _ in norm_positions)
            tag = ("trajRES" if trajectory_resolved else
                   "trajAVG" if trajectory_averaged else
                   "single")
            filename = f"corr2_y_profiles_N{Nx}_xs_{xdesc}_{tag}_S{samples}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath