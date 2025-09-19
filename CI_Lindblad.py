import numpy as np
import math
import time
import os
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import pyplot as plt

class CI_Lindblad:
    def __init__(self, Nx, Ny, decoh=True, alpha_init=1.0, nshell=None, dt_init=5e-2, max_steps_init=250, n_a_init=0.5, keep_history_init=True, G_init=None):
        """
        1) Build overcomplete Wannier spinors for a Chern insulator model.

        2) Generate masks for each partition A,B,C for computation of real space chern number
        """
        self.Nx, self.Ny = Nx, Ny
        self.decoh = decoh
        self.alpha = None
        self.dt_default = float(dt_init)
        self.max_steps_default = int(max_steps_init)
        self.n_a_default = float(n_a_init)
        self.keep_history_default = bool(keep_history_init)

        # --- Build tri-partition masks A, B, C inside a circle of radius R ---
        R = 0.4 * min(Nx, Ny)
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
            max_dx = int(math.floor(math.sqrt(rr - dy*dy)))
            x0 = max(0, xref - max_dx)
            x1 = min(Nx - 1, xref + max_dx)
            if x0 > x1:
                continue

            inside[x0:x1+1, y] = True
            dxs = np.arange(x0, x1+1) - xref
            dys = np.full_like(dxs, dy)
            theta = np.mod(np.arctan2(dys, dxs), 2*np.pi)

            A[x0:x1+1, y] = (theta >= 0)   & (theta < a2)
            B[x0:x1+1, y] = (theta >= a2)  & (theta < a4)
            C[x0:x1+1, y] = (theta >= a4)  & (theta < 2*np.pi)

        # store on the instance
        self.A_mask = A
        self.B_mask = B
        self.C_mask = C
        self.inside_mask = inside
        self.R = R
        self.ref = (xref, yref)

        print('Class CI_Lindblad has been Initialized')

        # --- Build overcomplete Wannier data; evolve only if not already cached ---
        # Construct OW (may set a spatially varying alpha if DW=True)
        self.construct_OW_functions(alpha_init, nshell=nshell)
        # Only perform the initial evolution if we don't already have cached results
        if not hasattr(self, "G_last") or self.G_last is None:
            G_last, G_hist = self.G_evolution(max_steps=self.max_steps_default,
                                              dt=self.dt_default,
                                              G_init=G_init,
                                              keep_history=self.keep_history_default)
            self.G_last = G_last
            self.G_history = G_hist

    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def _ensure_OW_ready(self, alpha_hint=1.0):
        """Ensure overcomplete Wannier (OW) structures exist before evolution.
        If they are missing, build them with the provided alpha hint (default 1.0).
        """
        need = False
        for attr in ("Pminus_k", "Pplus_k",
                     "W_A_plus","W_B_plus","W_A_minus","W_B_minus",
                     "V_A_plus","V_B_plus","V_A_minus","V_B_minus",
                     "V_plus","V_minus"):
            if not hasattr(self, attr):
                need = True
                break
        if need:
            ah = float(alpha_hint if self.alpha is None else self.alpha)
            self.construct_OW_functions(ah)

    def _ensure_evolved(self):
        """Ensure self.G_last / self.G_history exist (run one short evolution if missing)."""
        if not hasattr(self, "G_last") or self.G_last is None:
            G_last, G_hist = self.G_evolution(max_steps=self.max_steps_default,
                                              dt=self.dt_default,
                                              G_init=None,
                                              keep_history=self.keep_history_default)
            self.G_last = G_last
            self.G_history = G_hist

    def construct_OW_functions(self, alpha, nshell=None):
        '''
        Build overcomplete Wannier data for a **uniform** Chern insulator
        with scalar mass parameter alpha (float).

        Produces four fields:
          self.W_A_plus, self.W_B_plus, self.W_A_minus, self.W_B_minus,
        each of shape (Nx, Ny, 2, Rx, Ry) with axes:
          (x, y, mu, R_x, R_y)

        Also stores k-space band projectors **without** center axes:
          self.Pminus_k, self.Pplus_k  (Nx, Ny, 2, 2)

        If nshell is provided (integer), the real-space Wannier functions W are truncated
        to a square window of size (2*nshell+1)×(2*nshell+1) centered at each (R_x, R_y)
        using minimal-image (periodic) distances on the torus, then re-normalized per center.
        '''
        Nx, Ny = int(self.Nx), int(self.Ny)
        self.alpha = float(alpha)

        # k-grid
        kx = 2*np.pi * np.fft.fftfreq(Nx, d=1.0)
        ky = 2*np.pi * np.fft.fftfreq(Ny, d=1.0)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')  # (Nx,Ny)

        # Model vector n(k) for uniform alpha (all 2D, no center axes)
        nx = np.sin(KX)                 # (Nx,Ny)
        ny = np.sin(KY)                 # (Nx,Ny)
        nz = (self.alpha - np.cos(KX) - np.cos(KY))  # (Nx,Ny)
        nmag = np.sqrt(nx*nx + ny*ny + nz*nz)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        # Pauli matrices
        sx = np.array([[0,1],[1,0]], dtype=complex)
        sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        # hk(k) and band projectors in k-space (NO center axes)
        hk_k = (nx[...,None,None]*sx + ny[...,None,None]*sy + nz[...,None,None]*sz) / nmag[...,None,None]  # (Nx,Ny,2,2)
        self.Pminus_k = 0.5*(I2 - hk_k)   # (Nx,Ny,2,2)
        self.Pplus_k  = 0.5*(I2 + hk_k)   # (Nx,Ny,2,2)

        # Center phases for *all* centers (Rx,Ry)
        Rx_grid = np.arange(Nx)  # (Rx,)
        Ry_grid = np.arange(Ny)  # (Ry,)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Rx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ry)
        phase   = phase_x * phase_y                                                # (Nx,Ny,Rx,Ry)

        # Helper: FFT over k-axes only
        def k2_to_r2(Ak):    # Ak: (Nx,Ny,Rx,Ry)
            return np.fft.fft2(Ak, axes=(0,1))

        # local orbital spinors
        tauA = (1/np.sqrt(2))*np.array([[1],[1]],  dtype=complex)  # (2,1)
        tauB = (1/np.sqrt(2))*np.array([[1],[-1]], dtype=complex)  # (2,1)

        # Build normalized W for all centers from **4D** projectors (broadcast through phase)
        def make_W(Pband_k, tau, phase):
            # tau^† P(k): (Nx,Ny,2)
            tau_dag = tau[:,0].conj()                          # (2,)
            psi_k   = np.einsum('m,ijmn->ijn', tau_dag, Pband_k)  # (Nx,Ny,2)

            # Apply center phases; add center axes via broadcasting
            F0 = phase * psi_k[:,:,0][..., None, None]         # (Nx,Ny,Rx,Ry)
            F1 = phase * psi_k[:,:,1][..., None, None]         # (Nx,Ny,Rx,Ry)

            # FFT k→r *per center* for each spinor component
            W0 = k2_to_r2(F0)                                  # (Nx,Ny,Rx,Ry)
            W1 = k2_to_r2(F1)                                  # (Nx,Ny,Rx,Ry)

            # Stack spinor components → (Nx,Ny,2,Rx,Ry)
            W  = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)

            # Optional truncation and per-center normalization
            if nshell is not None:
                x = np.arange(Nx)[:,None,None,None]  # (Nx,1,1,1)
                y = np.arange(Ny)[None,:,None,None]  # (1,Ny,1,1)
                Rx = np.arange(Nx)[None,None,:,None] # (1,1,Rx,1)
                Ry = np.arange(Ny)[None,None,None,:] # (1,1,1,Ry)
                dx = ((x - Rx + Nx//2) % Nx) - Nx//2
                dy = ((y - Ry + Ny//2) % Ny) - Ny//2
                mask = (np.abs(dx) <= nshell) & (np.abs(dy) <= nshell)   # (Nx,Ny,Rx,Ry)
                W *= mask[:, :, None, :, :]  # expand to (Nx,Ny,2,Rx,Ry)

            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0,1,2), keepdims=True)) + 1e-15  # (1,1,1,Rx,Ry)
            return W / denom

        # Overcomplete Wannier spinors for all centers
        self.W_A_plus  = make_W(self.Pplus_k,  tauA, phase)
        self.W_B_plus  = make_W(self.Pplus_k,  tauB, phase)
        self.W_A_minus = make_W(self.Pminus_k, tauA, phase)
        self.W_B_minus = make_W(self.Pminus_k, tauB, phase)

        # V = sum_R |W_R⟩⟨W_R|
        def make_V(W):
            return np.einsum('ijklm, pqrlm -> ijkpqr', W, W.conj(), optimize=True)

        self.V_A_minus = make_V(self.W_A_minus)
        self.V_B_minus = make_V(self.W_B_minus)
        self.V_minus   = self.V_A_minus + self.V_B_minus

        self.V_A_plus  = make_V(self.W_A_plus)
        self.V_B_plus  = make_V(self.W_B_plus)
        self.V_plus    = self.V_A_plus + self.V_B_plus     
    
    def Lgain(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''
        Y = -(1/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_minus, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_minus, G, optimize=True))   
        return n_a*(self.V_minus + Y)
    
    def Lloss(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        Y = -((1-n_a)/2)*(np.einsum('ijklmn, lmnpqr -> ijkpqr', G, self.V_plus, optimize=True) + np.einsum('ijklmn, lmnpqr -> ijkpqr', self.V_plus, G, optimize=True))   
        return Y
    
    def double_comm(self, G, W, V):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)
        '''

        # Linear part: P G + G P
        VG = np.einsum('ijklmn, lmnpqr -> ijkpqr', V, G, optimize=True)
        GV = np.einsum('ijklmn, lmnpqr -> ijkpqr', G, V, optimize=True)
        linear_term = VG + GV

        # Scalar weights per center: s_ab = <w_ab| G |w_ab>
        s_ab = np.einsum('ijkab, ijkpqr, pqrab -> ab', W.conj(), G, W, optimize=True)

        # Nonlinear term: Σ_ab s_ab * |w_ab><w_ab|
        nonlinear_term = np.einsum('ijkab, ab, pqrab -> ijkpqr', W, s_ab, W.conj(), optimize=True)

        Y = linear_term - 2.0 * nonlinear_term
        return Y
    
    def Ldecoh(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        upperband_term = self.double_comm(G, self.W_A_plus, self.V_A_plus) + self.double_comm(G, self.W_B_plus, self.V_B_plus)
        lowerband_term = self.double_comm(G, self.W_A_minus, self.V_A_minus) + self.double_comm(G, self.W_B_minus, self.V_B_minus)
        
        Y = -(1/2)*((2-n_a)*lowerband_term + (1+n_a)*upperband_term)
        return Y
    
    def Lcycle(self, G, n_a):
        '''
        G in (Nx, Ny, 2, Nx, Ny, 2)            
        '''
        if self.decoh:
            Y = self.Lgain(G, n_a) + self.Lloss(G, n_a) + self.Ldecoh(G, n_a)
        else:
            Y = self.Lgain(G, n_a) + self.Lloss(G, n_a)
        return Y
        
    def rk4_Lindblad_evolver(self, G, dt, n_a = 0.5, tmp=None):
        """
        In-place RK4 for G' = Lindblad(G). Returns G (updated).
        tmp: optional dict of preallocated buffers {'k1','k2','k3','k4','Y'}
        """
        if tmp is None:
            tmp = {}
        k1 = tmp.get('k1'); k2 = tmp.get('k2'); k3 = tmp.get('k3'); k4 = tmp.get('k4'); Y = tmp.get('Y')
        # allocate once with the right dtype/shape
        if k1 is None: k1 = tmp['k1'] = np.empty_like(G)
        if k2 is None: k2 = tmp['k2'] = np.empty_like(G)
        if k3 is None: k3 = tmp['k3'] = np.empty_like(G)
        if k4 is None: k4 = tmp['k4'] = np.empty_like(G)
        if Y  is None: Y  = tmp['Y']  = np.empty_like(G)
        # k1 = f(G)
        k1[:] = self.Lcycle(G, n_a)
        # k2 = f(G + dt/2 * k1)
        np.multiply(k1, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k2[:] = self.Lcycle(Y, n_a)
        # k3 = f(G + dt/2 * k2)
        np.multiply(k2, 0.5*dt, out=Y); np.add(G, Y, out=Y)
        k3[:] = self.Lcycle(Y, n_a)
        # k4 = f(G + dt * k3)
        np.multiply(k3, dt, out=Y); np.add(G, Y, out=Y)
        k4[:] = self.Lcycle(Y, n_a)
        # G += dt/6 * (k1 + 2k2 + 2k3 + k4)
        # Use Y as accumulator: Y = k1 + 2k2 + 2k3 + k4
        np.add(k1, k4, out=Y)
        np.add(Y, 2.0*k2, out=Y)
        np.add(Y, 2.0*k3, out=Y)
        G += (dt/6.0) * Y
        return G, tmp
        

        
    def G_evolution(self, max_steps = 500, dt = 5e-2, G_init=None, keep_history=True, dtype=complex):
        """
        Evolve the real-space correlator/state G under G' = Lcycle(G) using RK4.

        Parameters
        ----------
        dt : float
            Time step.
        max_steps : int
            Number of RK4 steps to take.
        n_a : float, optional
            Parameter forwarded to Lcycle.
        G_init : ndarray or None, optional
            Initial state. If None, uses the zero-charge state with shape
            (Nx, Ny, 2, Nx, Ny, 2) where Nx=self.Nx, Ny=self.Ny.
        keep_history : bool, optional
            If True, records G after each step; length of the returned list is max_steps.
        dtype : dtype, optional
            Dtype for the zero state if G_init is None.

        Returns
        -------
        G : ndarray
            Final state with shape (Nx, Ny, 2, Nx, Ny, 2).
        G_history : list of ndarray
            If keep_history=True, a list of length max_steps with copies of G at each step;
            otherwise an empty list.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)

        # Make sure OW data (V_± etc.) are available before calling Lcycle
        self._ensure_OW_ready(alpha_hint=1.0)

        # Initialize G
        if G_init is None:
            G = np.zeros((Nx, Ny, 2, Nx, Ny, 2), dtype=dtype)
        else:
            G = np.array(G_init, copy=True)
            expected_shape = (Nx, Ny, 2, Nx, Ny, 2)
            if G.shape != expected_shape:
                raise ValueError(f"G_init must have shape {expected_shape}, got {G.shape}")

        G_history = []

        # Main evolution loop with live progress
        # Insert timer start
        t_start = time.time()
        for step in range(1, int(max_steps) + 1):
            t0 = time.time()
            G, tmp = self.rk4_Lindblad_evolver(G, dt)
            iter_time = time.time() - t0
            if keep_history:
                G_history.append(G.copy())
            # live status line
            clear_output(wait=True)
            print(f"[G_evolution] iter {step}/{int(max_steps)} | dt={dt:.3e} | itertime={iter_time:.3e}s | total={time.time()-t_start:.3e}s | N=({self.Nx},{self.Ny})")

        # Cache last and history for reuse
        self.G_last = G
        self.G_history = list(G_history) if keep_history else []
        return G, G_history

    # -----------------------------
    # Chern number 
    # -----------------------------

    def real_space_chern_number(self, G=None, A_mask=None, B_mask=None, C_mask=None):
        """
        Compute the real-space Chern number using the projector built from G.

        This implements the same formula as `chern_from_projector`, but takes the
        input as a general operator G and forms P = G.conj() before the block traces.

        Parameters
        ----------
        G : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
            Real-space operator. The projector used here is P = G.conj().
        A_mask, B_mask, C_mask : (Nx,Ny) boolean masks, optional
            If None, use the instance's stored tri-partition masks.

        Returns
        -------
        complex
            12π i [ Tr(P_CA P_AB P_BC) - Tr(P_AC P_CB P_BA) ].
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        Nx, Ny = int(self.Nx), int(self.Ny)
        expected = (Nx, Ny, 2, Nx, Ny, 2)
        if G.shape != expected:
            raise ValueError(f"G must have shape {expected}, got {G.shape}")

        # Default masks from the instance unless provided
        A_mask = self.A_mask if A_mask is None else A_mask
        B_mask = self.B_mask if B_mask is None else B_mask
        C_mask = self.C_mask if C_mask is None else C_mask

        # Build projector from G: P = G.conj()
        Ntot = Nx * Ny * 2
        P = np.asarray(G, dtype=complex, order='C').conj().reshape(Ntot, Ntot)

        def sector_indices_from_mask(mask_xy):
            """
            Return flattened (x,y,s) indices that include both orbitals for all True sites.
            """
            sites = np.flatnonzero(mask_xy.ravel(order='C'))  # j = x*Ny + y
            return np.concatenate((2*sites, 2*sites + 1))     # include both orbitals s=0,1
        
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

    def squared_two_point_corr(self, G=None, rx=0, ry=0):
        """
        Squared two-point correlator on a torus for separations r = (rx, ry).

            C_G(rx, ry) = (1 / (2 Nx Ny)) * sum_{mu,mu', r'}
                           | G[(r',mu), (r'+(rx,ry), mu')] |^2

        This version returns a **2D array** over all combinations of `rx` and `ry`:
        shape = (len(rx), len(ry)). If either `rx` or `ry` is a scalar, it is
        treated as a 1-element list, yielding shape (1, len(ry)) or (len(rx), 1).

        Periodic boundary conditions are enforced by modular addition in x and y.

        Parameters
        ----------
        G : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
            Real-space two-point function (can be complex).
        rx, ry : int or array-like of int
            Lattice separations along x and y. Scalars or 1D arrays.

        Returns
        -------
        C : ndarray, shape (len(rx), len(ry))
            2D grid of squared two-point correlators for all (rx, ry) pairs.
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")

        # Prepare base grids of starting sites r' = (x,y)
        X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

        # Ensure rx, ry are 1D integer arrays
        rx_arr = np.atleast_1d(rx).astype(int)
        ry_arr = np.atleast_1d(ry).astype(int)

        # Vectorized over all (rx, ry): build broadcastable index grids
        Rx, Ry = rx_arr.size, ry_arr.size
        Xb = X[:, :, None, None]                      # (Nx,Ny,1,1)
        Yb = Y[:, :, None, None]                      # (Nx,Ny,1,1)
        Xp = (Xb + rx_arr[None, None, :, None]) % Nx  # (Nx,Ny,Rx,1)
        Yp = (Yb + ry_arr[None, None, None, :]) % Ny  # (Nx,Ny,1,Ry)

        # Advanced indexing automatically broadcasts index arrays to a common shape.
        # Result has shape (Nx,Ny,Rx,Ry,2,2)
        blocks = G[Xb, Yb, :, Xp, Yp, :]

        # Sum over r'=(x,y) and spins (mu, mu') → (Rx,Ry)
        C = np.sum(np.abs(blocks)**2, axis=(0, 1, 4, 5)) / (2.0 * Nx * Ny)
        return C


    def squared_two_point_corr_xslice(self, G=None, x0=0, ry=0):
        """
        Squared two-point correlator along y for a fixed x = x0:

            C_x0(ry) = (1 / (2 Ny)) * sum_{mu,mu', y'}
                        | G[(x0,y',mu), (x0,y'+ry, mu')] |^2

        Parameters
        ----------
        G  : ndarray, shape (Nx, Ny, 2, Nx, Ny, 2)
        x0 : int
            Fixed x-column at which to compute the correlator (mod Nx).
        ry : int or array-like of int
            y-separations. Scalars or 1D array.

        Returns
        -------
        C : ndarray, shape (len(ry),)
            Squared correlator vs ry at fixed x0.
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last
        G = np.asarray(G)
        Nx, Ny, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")
        x0 = int(x0) % Nx

        # y' grid and separations
        Y = np.arange(Ny, dtype=np.intp)[:, None]        # (Ny, 1)
        ry_arr = np.atleast_1d(ry).astype(np.intp)       # (R,)
        R = ry_arr.size
        Yp = (Y + ry_arr[None, :]) % Ny                  # (Ny, R)

        # Extract the x0-column-to-x0-column block once: shape (Ny, 2, Ny, 2)
        Gx = G[x0, :, :, x0, :, :]                       # (Ny, 2, Ny, 2)

        # Reorder so the two y-axes are adjacent, then flatten (y, y') → flat index
        # Gx_re has shape (Ny*Ny, 2, 2) where flat = y * Ny + y'
        Gx_re = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny*Ny, 2, 2)

        # Build flat indices for all (y, y+ry) pairs, then gather and reshape to (Ny, R, 2, 2)
        flat_idx = (Y * Ny + Yp).reshape(-1)             # (Ny*R,)
        blocks = Gx_re[flat_idx].reshape(Ny, R, 2, 2)    # (Ny, R, 2, 2)

        # Average over y' and spins → (R,)
        C = np.sum(np.abs(blocks)**2, axis=(0, 2, 3)) / (2.0 * Ny)
        return C


    def plot_corr_y_profiles(self, dt=5e-2, max_steps=500, n_a=0.5,
                             x_positions=None, ry_max=None, filename=None):
        """
        Plot squared two-point correlation vs y-separation at several fixed x columns
        chosen to probe the domain-wall geometry:
          - deep in the topological bulk (alpha=1 slab center),
          - deep in each trivial bulk (alpha=3) on left and right,
          - at the left and right domain walls (interfaces).

        If x_positions is provided, it should be a list of (x0, label) pairs to plot.

        Parameters
        ----------
        dt, max_steps, n_a : evolution parameters passed to RK4 (if we evolve here).
        x_positions : list[(int,str)] or None
            Custom columns and labels. If None, choose sensible positions from DW geometry.
        ry_max : int or None
            Max y-separation r_y to plot; default Ny//2.
        filename : str or None
            PDF basename to save. If None, a descriptive name is generated.

        Returns
        -------
        fullpath : str
            Full path to the saved PDF.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)

        # Ensure Lindbladian ingredients exist
        self._ensure_OW_ready(alpha_hint=1.0)

        # Use cached steady-state
        self._ensure_evolved()
        G = self.G_last

        # Default y-range
        if ry_max is None:
            ry_max = Ny // 2
        ry_vals = np.arange(0, int(ry_max) + 1, dtype=int)

        # Default x-positions from the domain-wall construction used in construct_OW_functions:
        # central slab: x in [x0, x1) has alpha=1; outside is alpha=3
        if x_positions is None:
            #half = Nx // 2
            #w = int(np.floor(0.2 * Nx))
            #x0 = max(0, half - w)
            #x1 = min(Nx, half + w + 1)  # end-exclusive

            ## Representative columns
            #x_topo   = (x0 + x1) // 2                        # deep inside topo slab
            #x_triv_L = x0 // 2                               # middle of left trivial bulk
            #x_triv_R = (x1 + Nx) // 2                        # middle of right trivial bulk
            #x_wall_L = (x0 - 1) % Nx                         # just left of slab (interface)
            #x_wall_R = x1 % Nx                               # just right of slab (interface)

            #x_positions = [
            #    (x_topo,   "topo bulk"),
            #    (x_triv_L, "trivial (L)"),
            #    (x_triv_R, "trivial (R)"),
            #    (x_wall_L, "wall (L)"),
            #    (x_wall_R, "wall (R)"),
            #]
            x_positions = np.arange(4, 27)

        # Plot
        outdir = self._ensure_outdir('figs/corr_y_profiles')
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # --- Add Chern marker inset ---
        # Compute local Chern marker for inset
        #C_inset = self.local_chern_marker(G)
        # Add an inset showing tanh C(r)
        #axins = inset_axes(ax, width="32%", height="40%", loc="upper right", borderpad=1.0)
        #im_in = axins.imshow(C_inset, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
        # Show ticks and labels on the inset
        #axins.set_xlabel('y', fontsize=9)
        #axins.set_ylabel('x', fontsize=9)
        #axins.tick_params(axis='both', labelsize=8)
        # Draw a light grid at lattice spacing to make the torus lattice visible
        #Nx, Ny = int(self.Nx), int(self.Ny)
        #axins.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        #axins.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        #axins.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)
        # Colorbar with legend text
        #cbar_in = fig.colorbar(im_in, ax=axins, fraction=0.046, pad=0.04)
        #cbar_in.set_label(r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=9)
        #cbar_in.ax.tick_params(labelsize=8)

        # Plot correlation profiles with inline labels to the right of each curve
        for x0 in x_positions:
            C_vec = self.squared_two_point_corr_xslice(G, x0=int(x0), ry=ry_vals).real
            line, = ax.plot(ry_vals, C_vec, marker='o', ms=3, lw=1, label=f"$x_0={int(x0)%Nx}$")
            
            # Determine a good y value at the right edge (use last finite value)
            yvals = C_vec
            # fallback if there are NaNs/inf at the end
            finite = np.isfinite(yvals)
            if np.any(finite):
                y_right = yvals[finite][-1]
            else:
                y_right = yvals[-1]

            # Place label slightly to the right of the last x value
            x_right = ry_vals[-1] * 1.02 if ry_vals[-1] > 0 else ry_vals[-1] + 0.5
            ax.annotate(f"{int(x0)%Nx}", xy=(ry_vals[-1], y_right), xytext=(x_right, y_right),
                        textcoords='data', ha='left', va='center', fontsize=9,
                        color=line.get_color())

        ax.set_xlabel(r"$r_y$")
        ax.set_ylabel(r"$C_G(x_0; r_y)$")
        ax.set_title(f"Squared correlator vs $r_y$ at fixed $x_0$ (N={Nx}, steps={int(max_steps)})")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        # Inset legend inside the axes (lower-left corner), multi‑column, semi‑transparent box
        leg = ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.02, 0.02),   # within axes: small margin from bottom-left
            ncol=4,                        # adjust columns as desired
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            borderpad=0.4,
            handlelength=1.5,
            handletextpad=0.6,
            columnspacing=0.9,
            labelspacing=0.3
        )
        # Ensure tight layout now that the legend is inside the axes
        fig.tight_layout()
        # Inline labels used instead of legend
        # ax.legend()

        # Robust filename encoding for x_positions
        if filename is None:
            # encode x positions briefly (support both list of ints and list of (x,label))
            try:
                # list of (x,label)
                xlist = [int(x) for (x, *_) in x_positions]
            except Exception:
                # list/array of ints
                xlist = [int(x) for x in x_positions]
            xdesc = "-".join(str(int(x) % Nx) for x in xlist)
            decoh_tag = "_decoh_on" if self.decoh else ""
            filename = f"corr2_y_profiles_N{Nx}_xs_{xdesc}_steps{int(max_steps)}{decoh_tag}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath)
        plt.close(fig)
        return fullpath

    def G_CI(self, alpha=1.0, norm='backward', k_is_centered=False):
        """
        Real-space two-point function G = P_minus^*(r,r') for uniform alpha.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
    
        # (Re)build k-space projectors if alpha differs
        if (self.alpha is None) or (float(alpha) != float(self.alpha)):
            self.construct_OW_functions(float(alpha))
    
        def _k_to_r_rel_full(Pk):  # Pk: (Nx,Ny,2,2)
            if k_is_centered:
                Pk = np.fft.ifftshift(Pk, axes=(0,1))
            PR = np.fft.ifft2(Pk, axes=(0,1), norm=norm)  # (Nx,Ny,2,2)
            PR = np.real_if_close(PR, tol=1e3)
    
            x = np.arange(Nx); y = np.arange(Ny)
            X  = x[:, None, None, None]
            Xp = x[None, None, :, None]
            Y  = y[None, :, None, None]
            Yp = y[None, None, None, :]
            dX = (X - Xp) % Nx
            dY = (Y - Yp) % Ny
            return PR[dX, dY, :, :]                         # (Nx,Ny,Nx,Ny,2,2)
    
        Pminus_rel = _k_to_r_rel_full(self.Pminus_k)        # 6D
        G = np.moveaxis(Pminus_rel.conj(), 4, 2)            # (Nx,Ny,2,Nx,Ny,2)
        return G

    def plot_squared_corr_vs_alpha(self, direction='x', alpha_list=None,
                                   dt=5e-2, max_steps=500, filename=None):
        """
        Plot 1D squared correlator C(r) vs r along a chosen path, overlaying
        curves for multiple alpha values.

        Direction options
        -----------------
        direction='x'   : use separations (r, 0) with r = 0..Nx//2  → C(r) = C_G(r, 0)
        direction='y'   : use separations (0, r) with r = 0..Ny//2  → C(r) = C_G(0, r)
        direction='diag': use separations (r, r) with r = 0..min(Nx,Ny)//2 → C(r) = C_G(r, r)

        Parameters
        ----------
        direction : {'x','y','diag'}
            Which path in (r_x,r_y) to plot along.
        alpha_list : iterable of float, optional
            Sequence of alpha values. If None, defaults to linspace(-4,4,161).
        dt : float
            RK4 time step.
        max_steps : int
            Number of steps for steady-state approximation.
        filename : str or None
            Basename of the PDF to save. If None, a descriptive name is generated
            that reflects the chosen direction and r-range.

        Returns
        -------
        fullpath : str
            Full path to the saved PDF.
        """
        # Alphas to compare
        if alpha_list is None:
            alphas = np.linspace(-4.0, 4.0, 161)
        else:
            alphas = np.asarray(alpha_list, dtype=float)

        if direction not in ('x', 'y', 'diag'):
            raise ValueError("direction must be 'x', 'y', or 'diag'")

        Nx, Ny = int(self.Nx), int(self.Ny)
        if direction == 'x':
            r_vals = np.arange(0, Nx//2 + 1, dtype=int)  # r = |rx|
            rx_arr = r_vals
            ry_arr = np.array([0], dtype=int)
            ylabel = r"$C_G(r, 0)$"
            title_dir = 'x'
        elif direction == 'y':
            r_vals = np.arange(0, Ny//2 + 1, dtype=int)  # r = |ry|
            rx_arr = np.array([0], dtype=int)
            ry_arr = r_vals
            ylabel = r"$C_G(0, r)$"
            title_dir = 'y'
        else:  # 'diag'
            rmax = min(Nx, Ny)//2
            r_vals = np.arange(0, rmax + 1, dtype=int)  # r = |rx| = |ry|
            rx_arr = r_vals
            ry_arr = r_vals
            ylabel = r"$C_G(r, r)$"
            title_dir = 'diag'

        outdir = self._ensure_outdir('figs/corr_vs_r')

        # If alphas contains a single value and we already have a cached evolution, reuse it
        if np.size(alphas) == 1:
            self._ensure_evolved()
            G_final = self.G_last
            a = float(alphas.reshape(-1)[0])
            C_grid = self.squared_two_point_corr(G_final, rx=rx_arr, ry=ry_arr)
            if direction == 'diag':
                L = r_vals.size
                C_vec = C_grid[np.arange(L), np.arange(L)].astype(float)
            else:
                C_vec = C_grid.reshape(-1).astype(float)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(r_vals, C_vec, marker='o', ms=3, lw=1, label=f"alpha={a:g}")
            ax.set_xlabel(r"$r$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Steady-state |G|$^2$ vs r along {title_dir}-path")
            ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend()
            plt.tight_layout()
            outdir = self._ensure_outdir('figs/corr_vs_r')
            if filename is None:
                r_desc = f"r0-{r_vals[-1]}"
                filename = f"corr2_1D_vs_r_dir{title_dir}_N{Nx}_{r_desc}_alpha_{a:g}.pdf"
            fullpath = os.path.join(outdir, filename)
            plt.show(); fig.savefig(fullpath, bbox_inches='tight'); plt.close(fig)
            return fullpath
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for a in alphas:
            solver = CI_Lindblad(Nx=Nx, Ny=Ny, decoh=self.decoh, alpha_init=a)
            G_final, _ = solver.G_evolution(max_steps=int(max_steps), dt=dt, keep_history=False)
            C_grid = solver.squared_two_point_corr(G_final, rx=rx_arr, ry=ry_arr)

            if direction == 'diag':
                # C_grid has shape (len(r), len(r)); take the diagonal entries
                L = r_vals.size
                C_vec = C_grid[np.arange(L), np.arange(L)].astype(float)
            else:
                # C_grid is (len(r),1) or (1,len(r)); flatten to vector of length len(r)
                C_vec = C_grid.reshape(-1).astype(float)

            ax.plot(r_vals, C_vec, marker='o', ms=3, lw=1, label=f"alpha={a:g}")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Steady-state |G|$^2$ vs r along {title_dir}-path (steps={int(max_steps)})")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        # Build descriptive filename
        if filename is None:
            # Encode alpha list succinctly
            if alphas.size <= 6:
                a_desc = '-'.join(f"{v:g}" for v in alphas)
            else:
                a_desc = f"{alphas.size}vals_{alphas.min():g}to{alphas.max():g}"
            r_desc = f"r0-{r_vals[-1]}"
            if self.decoh:
                filename = f"corr2_1D_vs_r_dir{title_dir}_N{Nx}_{r_desc}_alphas_{a_desc}_steps{int(max_steps)}_decoh_on.pdf"
            else:
                filename = f"corr2_1D_vs_r_dir{title_dir}_N{Nx}_{r_desc}_alphas_{a_desc}_steps{int(max_steps)}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath
    
    def local_chern_marker(self, G=None, mask_outside=False):
        """
        Local Chern marker C(r) (Bianco–Resta) using *non-modular* coordinates.
        Here (x,y) run 1..Nx and 1..Ny (1-based), i.e. X=1..Nx, Y=1..Ny multiply
        the right-hand real-space indices directly.

        Parameters
        ----------
        G : ndarray or None
            Two-point function with shape (Nx, Ny, 2, Nx, Ny, 2). If None, uses
            the cached steady-state `self.G_last`.
        mask_outside : bool
            If True, zero out values outside `self.inside_mask` (for visualization
            inside the tri-partition disk).

        Returns
        -------
        C_tanh : ndarray, shape (Nx, Ny)
            tanh of the local Chern marker.
        """
        if G is None:
            self._ensure_evolved()
            G = self.G_last

        P = np.asarray(G.conj())
        Nx, Ny, s1, Nx2, Ny2, s2 = P.shape
        if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
            raise ValueError("G must have shape (Nx, Ny, 2, Nx, Ny, 2).")

        # --- Non-modular 1-based coordinates: X=1..Nx, Y=1..Ny ---
        X = np.arange(1, Nx + 1, dtype=float)
        Y = np.arange(1, Ny + 1, dtype=float)
        Xgrid, Ygrid = np.meshgrid(X, Y, indexing='ij')

        # Multiply by X or Y on the right real-space indices (x',y') of the kernel
        def right_X(A):  return A * Xgrid[None, None, None, :, :, None]
        def right_Y(A):  return A * Ygrid[None, None, None, :, :, None]

        # Contraction over shared (x',y',s')
        def mm(A, B):    return np.einsum('ijslmn,lmnopr->ijsopr', A, B, optimize=True)

        # Ordered products
        T = right_X(P); T = mm(T, P); T = right_Y(T); T = mm(T, P)  # G X G Y G
        U = right_Y(P); U = mm(U, P); U = right_X(U); U = mm(U, P)  # G Y G X G
        M = (2.0 * np.pi * 1j) * (T - U)

        # Diagonal (x,y,μ; x,y,μ), sum over μ
        ix = np.arange(Nx)[:, None, None]
        iy = np.arange(Ny)[None, :, None]
        ispin = np.arange(2)[None, None, :]
        diag_vals = M[ix, iy, ispin, ix, iy, ispin]  # (Nx,Ny,2)
        C = diag_vals.sum(axis=2)                    # (Nx,Ny)

        C = np.tanh(np.real_if_close(C, tol=1e-9))
        if mask_outside:
            C = np.where(self.inside_mask, C, 0.0)
        return C    


    def chern_marker_dynamics(self, dt=5e-2, max_steps=250, n_a=0.5,
                              G_init=None, fps=12, cmap='RdBu_r',
                              outbasename=None, vmin=-1.0, vmax=1.0):
        """
        Animate the local Chern marker C(r) over time using the cached evolution history
        if available; otherwise, perform a short evolution to generate the frames.
        Saves a GIF and also saves a static final frame as a PDF image.
        Returns the paths to both the GIF and the final image, as well as the final arrays.

        Parameters
        ----------
        dt : float
            RK4 time step.
        max_steps : int
            Number of RK4 steps.
        n_a : float
            Parameter forwarded to Lcycle.
        G_init : ndarray or None
            Optional initial state for G; otherwise zeros.
        fps : int
            Frames per second of the output GIF.
        cmap : str
            Matplotlib colormap for the heatmap.
        outbasename : str or None
            Base file name (without extension). If None, a descriptive one is used.
        vmin, vmax : float
            Color scale limits for the marker plot.

        Returns
        -------
        gif_path : str
            Full path to the saved GIF.
        final_path : str
            Full path to the saved static final frame (PDF).
        C_last : ndarray, shape (Nx, Ny)
            Chern marker of the final step.
        G : ndarray
            Final G after evolution.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)

        # Ensure Lindbladian ingredients exist
        self._ensure_OW_ready(alpha_hint=1.0)


        # Where to save
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            decoh_tag = 'decoh_on' if self.decoh else 'decoh_off'
            outbasename = f"chern_marker_dynamics_N{Nx}_steps{int(max_steps)}_{decoh_tag}"
        gif_path = os.path.join(outdir, outbasename + ".gif")

        # --- Set up compact figure/axes like the reference frame ---
        # Small square panel with room for a horizontal colorbar above.
        fig = plt.figure(figsize=(3.2, 3.8))
        # Main image axes
        ax = fig.add_axes([0.12, 0.10, 0.78, 0.78])  # [left, bottom, width, height]
        im = ax.imshow(np.zeros((Nx, Ny)), cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')

        # Style: thick black border; no axis labels or ticks
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        # Show labeled ticks and a light lattice grid
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        # Major ticks every 5 (adjust step if you like)
        ax.set_xticks(np.arange(0, Ny, max(1, Ny//10)))
        ax.set_yticks(np.arange(0, Nx, max(1, Nx//10)))
        ax.tick_params(axis='both', labelsize=8)
        # Minor ticks at every cell boundary to draw the lattice grid
        ax.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        ax.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)

        # Horizontal colorbar above the panel with ticks at -1, 0, 1 and text on the left
        cax = fig.add_axes([0.32, 0.90, 0.36, 0.06])  # a small bar above the image
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
        # Put a LaTeX-like label to the left of the colorbar
        fig.text(0.70, 0.91, r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=12)


        # Initialize with zeros to guarantee a first frame
        C = np.zeros((Nx, Ny))
        im.set_data(C)

        writer = animation.PillowWriter(fps=fps)

        # Timing
        t_start = time.time()
        # Use precomputed history if available; otherwise fall back to a quick evolution
        if hasattr(self, "G_history") and len(self.G_history) > 0:
            history = self.G_history
            G = self.G_last
        else:
            # Fallback: perform a short evolution to get frames
            G_tmp, G_hist_tmp = self.G_evolution(max_steps=int(max_steps), dt=dt, keep_history=True)
            history = G_hist_tmp
            G = history[-1]

        with writer.saving(fig, gif_path, dpi=120):
            writer.grab_frame()
            for step, G_frame in enumerate(history, start=1):
                t0 = time.time()
                C = self.local_chern_marker(G_frame)
                im.set_data(C)
                writer.grab_frame()

                iter_time = time.time() - t0
                clear_output(wait=True)
                print(f"[chern_marker_dynamics] frame {step}/{len(history)} | "
                      f"proc={iter_time:.3e}s | total={time.time()-t_start:.3e}s | "
                      f"N=({Nx},{Ny}) | decoh={self.decoh}")

        plt.close(fig)

        # Final arrays to return
        C_last = np.round(C, decimals = 2)

        # Save static final frame (same styling as GIF frame)
        final_path = os.path.join(outdir, outbasename + "_final.pdf")
        fig2 = plt.figure(figsize=(3.2, 3.8))
        ax2 = fig2.add_axes([0.12, 0.10, 0.78, 0.78])
        im2 = ax2.imshow(C_last, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        ax2.set_xlabel("y")
        ax2.set_ylabel("x")
        ax2.set_xticks(np.arange(0, Ny, max(1, Ny//10)))
        ax2.set_yticks(np.arange(0, Nx, max(1, Nx//10)))
        ax2.tick_params(axis='both', labelsize=8)
        ax2.set_xticks(np.arange(-0.5, Ny, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, Nx, 1), minor=True)
        ax2.grid(which='minor', color='k', linewidth=0.2, alpha=0.25)

        cax2 = fig2.add_axes([0.32, 0.90, 0.36, 0.06])
        cbar2 = fig2.colorbar(im2, cax=cax2, orientation='horizontal', ticks=[-1, 0, 1])
        fig2.text(0.70, 0.91, r"$\tanh\mathcal{C}(\mathbf{r})$", fontsize=12)

        fig2.savefig(final_path)
        plt.show()
        plt.close(fig2)

        # --- plot and save steady-state line cut: tanh C(x, y=0) vs x ---
        final_cut_path = os.path.join(outdir, outbasename + "_final_y0_linecut.pdf")
        fig3 = plt.figure(figsize=(3.2, 3.0))
        ax3 = fig3.add_axes([0.15, 0.18, 0.80, 0.72])

        x_vals = np.arange(Nx)
        y0_cut = C_last[:, 0]  # y = 0 column; first index is x (rows), second is y (cols)
        ax3.plot(x_vals, y0_cut, marker='o', ms=4, lw=1)

        ax3.set_xlabel("x")
        ax3.set_ylabel(r"$\tanh\,\mathcal{C}(x,y\!=\!0)$")
        ax3.grid(True, alpha=0.3)

        fig3.savefig(final_cut_path)
        plt.show()
        plt.close(fig3)

        return gif_path, final_path, C_last, G


    def plot_spectrum_vs_nshell(self, nshell_list=(None, 1, 2, 3), alpha=None,
                                 dt=5e-2, max_steps=250, filename=None):
        """
        Compute and plot the spectrum (eigenvalues) of the steady-state two-point
        function G for a fixed system size (self.Nx,self.Ny) and fixed alpha,
        comparing different truncations nshell \in {1,2,3,\infty} (\infty ≡ None).

        For each nshell, we build a CI_Lindblad with the same (Nx,Ny,decoh,alpha),
        evolve to `max_steps`, then diagonalize the Hermitian matrix
            G_H = (G + G^\dagger)/2
        where G is reshaped to a (2*Nx*Ny, 2*Nx*Ny) operator. We plot the sorted
        eigenvalues versus their index, using distinct markers for different nshell.

        Parameters
        ----------
        nshell_list : iterable of int or None
            Sequence of truncation radii to compare. Use None to denote no truncation (∞).
        alpha : float or None
            Mass parameter. If None, use self.alpha.
        dt : float
            RK4 time step used in evolution.
        max_steps : int
            Number of RK4 steps.
        filename : str or None
            If None, a descriptive filename is generated under figs/spectrum_nshell/.

        Returns
        -------
        fullpath : str
            Full path to the saved PDF.
        """
        Nx, Ny = int(self.Nx), int(self.Ny)
        alpha_val = float(self.alpha if alpha is None else alpha)

        outdir = self._ensure_outdir('figs/spectrum_nshell')
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        marker_cycle = ['o', 's', 'x', '^', 'D', '*', 'P']

        for i, ns in enumerate(nshell_list):
            # Build solver with given truncation
            #solver = CI_Lindblad(Nx=Nx, Ny=Ny, decoh=self.decoh, alpha_init=alpha_val, nshell=ns)
            # Evolve once; use cached result inside the instance
            #solver._ensure_evolved()
            self.construct_OW_functions(alpha = alpha_val, nshell = ns)
            G, _ = self.G_evolution(max_steps = max_steps, dt = dt)

            # Flatten G to matrix and symmetrize to ensure Hermiticity
            Ntot = 2*Nx*Ny
            Gm = G.reshape(Ntot, Ntot)

            # Diagonalize
            evals, _ = np.linalg.eigh(Gm)
            evals.sort()

            idx = np.arange(Ntot)
            label_ns = '∞' if ns is None else str(int(ns))
            ax.plot(idx, evals.real,
                    marker=marker_cycle[i % len(marker_cycle)],
                    markevery=max(1, Ntot // 64),
                    lw=1.0, ms=3,
                    label=f"nshell={label_ns}", linestyle = '')

        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("eigenvalue of G")
        ax.set_title(f"Spectrum of G vs nshell (N={Nx}, alpha={alpha_val:g}, steps={int(max_steps)})")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(title="truncation", loc='best', ncol=1)
        plt.tight_layout()

        if filename is None:
            def _ns_desc(vals):
                out = []
                for v in vals:
                    out.append('inf' if v is None else str(int(v)))
                return '-'.join(out)
            ns_desc = _ns_desc(nshell_list)
            decoh_tag = '_decoh_on' if self.decoh else '_decoh_off'
            filename = f"spectrum_G_vs_nshell_N{Nx}_alpha{alpha_val:g}_nshells_{ns_desc}_steps{int(max_steps)}{decoh_tag}.pdf"

        fullpath = os.path.join(outdir, filename)
        plt.show()
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)
        return fullpath
   