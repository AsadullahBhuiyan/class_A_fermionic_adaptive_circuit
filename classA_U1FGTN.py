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
from contextlib import nullcontext
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib 


class classA_U1FGTN:
# ==================== EDIT: __init__ (load-or-build histories) ====================

    def __init__(self, Nx, Ny, DW=True, cycles=None, samples=None, nshell=None,
                 filling_frac=1/2, G0=None, *,
                 init_kind="default",            # "default" or "maxmix_top"
                 backend="loky",                 # for generation, if needed
                 n_jobs=None,                    # for generation, if needed
                 prompt_on_miss=True):           # prompt before generating if cache miss
    
        self.time_init = time.time()
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.DW = bool(DW)
        self.Ntot = 4 * self.Nx * self.Ny
        self.nshell = nshell
    
        self.cycles  = 5 if cycles  is None else int(cycles)
        self.samples = 5 if samples is None else int(samples)
        self.filling_frac = filling_frac
    
        # ---------------- initial G0 ----------------
        if G0 is None:
            Ntot   = self.Ntot
            Nlayer = Ntot // 2
            Nfill = int(round(filling_frac * Ntot))
            if Nfill < 0:
                Nfill = 0
            if Nfill > Ntot:
                Nfill = Ntot
            diag = np.concatenate([np.ones(Nfill, dtype=np.complex128),
                                   -np.ones(Ntot - Nfill, dtype=np.complex128)])
            rng = np.random.default_rng()
            rng.shuffle(diag)
            D = np.diag(diag)
            U_top  = self.random_unitary(Nlayer)
            I_bot  = np.eye(Nlayer, dtype=np.complex128)
            U_tot  = self._block_diag2(U_top, I_bot)
            self.G0 = U_tot.conj().T @ D @ U_tot
        else:
            self.G0 = np.asarray(G0, dtype=np.complex128)
    
        # Histories container (filled by load or generate)
        self.G_history_samples = None
        self._loaded_top_only = False  # set True if we loaded an old top-only cache
    
        # -------- try to load existing data for current parameters --------
        hit = self.load_G_history_samples(
            samples=self.samples,
            cycles=self.cycles,
            nshell=self.nshell,
            init_kind=init_kind
        )
    
        if hit is None:
            if prompt_on_miss:
                print("Cached histories not found for current parameters. Press ENTER to generate, or Ctrl-C to abort.")
                try:
                    _ = input()
                except Exception:
                    pass
                
            # Need OW only if we are going to generate
            self.construct_OW_projectors(nshell=self.nshell, DW=self.DW)
    
            # Generate with full G saved
            if n_jobs is None:
                if self.samples <= 1:
                    n_jobs = 1
                else:
                    n_jobs = min(self.samples, os.cpu_count() or 1)
    
            # Map init_kind -> run() init_mode
            if init_kind == "default":
                init_mode = "default"
            else:
                init_mode = "maxmix_top"  # maximally mixed top layer
    
            result = self.run_adaptive_circuit(
                G_history=True,
                cycles=self.cycles,
                samples=self.samples,
                n_jobs=n_jobs,
                backend=backend,
                parallelize_samples=True,
                store="full",         # save full G histories
                init_mode=init_mode,
                progress=True
            )
            # Convert returned packed history (S,T,dim,dim) to list[S][T] full G
            S = int(result["G_hist"].shape[0])
            T = int(result["G_hist"].shape[1])
            full_hist_list = []
            for s in range(S):
                traj = []
                for t in range(T):
                    traj.append(np.array(result["G_hist"][s, t], copy=True))
                full_hist_list.append(traj)
    
            # Save to cache (full)
            self.save_G_history_samples(
                full_hist_list,
                samples=self.samples,
                cycles=self.cycles,
                nshell=self.nshell,
                init_kind=init_kind
            )
    
        print("------------------------- classA_U1FGTN Initialized -------------------------")

    # ------------ Centralized cache helpers (FULL + convenience TOP) ------------

    def _cache_dir_Ghist(self):
        return self._ensure_outdir("cache/G_history")

    def _cache_key(self, samples=None, cycles=None, nshell=None, init_kind="default"):
        """
        No seed info. Keep DW in the key so different physics don't collide.
        """
        Nx, Ny = self.Nx, self.Ny
        C  = int(self.cycles  if cycles  is None else cycles)
        S  = int(self.samples if samples is None else samples)
        nsh = "None" if nshell is None else str(int(nshell))
        kind = str(init_kind)  # "default" | "maxmix_top"
        DW_tag = f"DW{int(bool(self.DW))}"
        return f"N{Nx}x{Ny}_C{C}_S{S}_nsh{nsh}_{DW_tag}_init-{kind}"

    def _cache_path_Ghist(self, **kw):
        return os.path.join(self._cache_dir_Ghist(), self._cache_key(**kw) + ".npz")

    def load_G_history_samples(self, **kw):
        """
        Try to load list[S][T] of FULL G (4N x 4N).
        If only a legacy top-only file exists, load that and mark self._loaded_top_only=True.
        Returns None if not found.
        """
        path = self._cache_path_Ghist(**kw)
        if not os.path.isfile(path):
            return None

        data = np.load(path, allow_pickle=True)
        full_key = "G_history_full_objs"
        top_key  = "G_history_top_objs"

        if full_key in data.files:
            arr = data[full_key]            # (S,T) object array, each (Ntot,Ntot)
            S, T = arr.shape
            out = [[arr[s, t] for t in range(T)] for s in range(S)]
            self.G_history_samples = out
            self._loaded_top_only = False
            return out

        if top_key in data.files:
            # Backward-compatibility: top-only histories exist
            arr = data[top_key]             # (S,T) object array, each (Nlayer,Nlayer)
            S, T = arr.shape
            out = [[arr[s, t] for t in range(T)] for s in range(S)]
            self.G_history_samples = out
            self._loaded_top_only = True
            return out

        return None

    def save_G_history_samples(self, histories, **kw):
        """
        Save list[S][T] of FULL G (4N x 4N).
        Also writes a convenience top-layer copy for fast/legacy reads.
        """
        S = len(histories)
        if S == 0:
            raise ValueError("save_G_history_samples: empty histories list.")
        T = len(histories[0])
        Ntot   = int(self.Ntot)
        Nlayer = Ntot // 2

        full_obj = np.empty((S, T), dtype=object)
        top_obj  = np.empty((S, T), dtype=object)

        for s in range(S):
            if len(histories[s]) != T:
                raise ValueError("save_G_history_samples: all trajectories must have the same T.")
            for t in range(T):
                G = np.asarray(histories[s][t], dtype=np.complex128)
                if G.shape != (Ntot, Ntot):
                    raise ValueError(f"save_G_history_samples expects FULL G of shape ({Ntot},{Ntot}).")
                full_obj[s, t] = G
                top_obj[s, t]  = G[:Nlayer, :Nlayer]

        path = self._cache_path_Ghist(**kw)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        meta = {
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "Ntot": Ntot,
            "cycles": int(self.cycles),
            "samples": int(S),
            "nshell": ("None" if self.nshell is None else int(self.nshell)),
            "DW": bool(self.DW),
            "DW_loc": getattr(self, "DW_loc", None),
            "filling_frac": float(getattr(self, "filling_frac", 0.5)),
            "init_kind": str(kw.get("init_kind", "default")),
        }

        np.savez_compressed(
            path,
            G_history_full_objs=full_obj,
            G_history_top_objs=top_obj,              # convenience/legacy
            meta_json=np.array([str(meta)], dtype=object)
        )

        # keep in-memory mirror (use FULL so downstream can slice as needed)
        self.G_history_samples = [[full_obj[s, t] for t in range(T)] for s in range(S)]
        self._loaded_top_only = False
        return path
    
    # ------------ History accessors (lean) ------------

    def _top_layer_from_full(self, Gfull):
        """Slice the top layer from a full 4N×4N covariance."""
        Nlayer = self.Ntot // 2
        return np.asarray(Gfull, dtype=np.complex128)[:Nlayer, :Nlayer]

    def get_full_histories(self):
        """
        Return list[S][T] of FULL G (4N×4N), as loaded by load_G_history_samples().
        Does not generate. Raises if nothing is loaded or if only a legacy top-only cache is present.
        """
        if self.G_history_samples is None:
            raise RuntimeError("Histories not loaded yet.")
        if getattr(self, "_loaded_top_only", False):
            raise RuntimeError("Only a legacy top-only cache is loaded; full histories are unavailable.")
        # hand out copies to avoid accidental mutation of the in-memory cache
        return [[np.array(Gfull, copy=True) for Gfull in traj] for traj in self.G_history_samples]

    def get_top_histories(self):
        """
        Return list[S][T] of top-layer G_tt.
        Works for both full caches (preferred) and legacy top-only files.
        """
        if self.G_history_samples is None:
            raise RuntimeError("Histories not loaded yet.")

        S = len(self.G_history_samples)
        if S == 0:
            return []

        if getattr(self, "_loaded_top_only", False):
            # Already top-only: coerce dtype and copy for safety
            return [[np.array(Gt, dtype=np.complex128, copy=True) for Gt in traj]
                    for traj in self.G_history_samples]

        # Full in memory -> slice top layer
        out = []
        for traj in self.G_history_samples:
            out.append([self._top_layer_from_full(Gfull) for Gfull in traj])
        return out

    def get_history_bundle(self):
        """
        Convenience: stacks TOP-layer histories into arrays:
        - "G_hist"     : (S, T, Nlayer, Nlayer)
        - "G_hist_avg" : (T, Nlayer, Nlayer)
        - "samples"    : S
        - "T"          : T
        """
        top_histories = self.get_top_histories()
        S = len(top_histories)
        if S == 0:
            return {"G_hist": np.empty((0, 0, 0, 0)),
                    "G_hist_avg": np.empty((0, 0, 0)),
                    "samples": 0, "T": 0}

        T = len(top_histories[0])
        Nlayer = self.Ntot // 2
        G_hist = np.empty((S, T, Nlayer, Nlayer), dtype=np.complex128)
        for s in range(S):
            for t in range(T):
                G_hist[s, t] = np.asarray(top_histories[s][t], dtype=np.complex128)
        return {
            "G_hist": G_hist,
            "G_hist_avg": np.mean(G_hist, axis=0),
            "samples": S,
            "T": T,
        }

    def current_final_G(self, sample_index=0, averaged=False):
        """
        Return final-step TOP-layer G.
        - averaged=False: G_tt from sample_index
        - averaged=True : average over samples of final G_tt
        """
        tops = self.get_top_histories()
        if len(tops) == 0:
            raise RuntimeError("No histories loaded.")
        if averaged:
            arr = [np.asarray(tops[s][-1], dtype=np.complex128) for s in range(len(tops))]
            return np.mean(arr, axis=0)
        s = int(np.clip(sample_index, 0, len(tops) - 1))
        return np.asarray(tops[s][-1], dtype=np.complex128)

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
    

    # ------------ Centralized cache helpers (FULL/TOP) ------------
    def _cache_dir_Ghist(self):
        return self._ensure_outdir("cache/G_history")

    def _cache_key(self, *, samples=None, cycles=None, nshell=None, init_kind="default"):
        Nx, Ny = self.Nx, self.Ny
        C  = int(self.cycles if cycles  is None else cycles)
        S  = int(self.samples if samples is None else samples)
        nsh = "None" if nshell is None else str(int(nshell))
        kind = str(init_kind)  # "default" or "maxmix_top"
        DW_tag = f"DW{int(bool(self.DW))}"
        return f"N{Nx}x{Ny}_C{C}_S{S}_nsh{nsh}_{DW_tag}_init-{kind}"

    def _cache_path_Ghist(self, **kw):
        return os.path.join(self._cache_dir_Ghist(), self._cache_key(**kw) + ".npz")

    def load_G_history_samples(self, *, samples=None, cycles=None, nshell=None, init_kind="default"):
        """
        Try to load list[S][T] of histories.
        Supports either:
        - G_history_full_objs: (S,T) object array of (Ntot,Ntot) full G
        - G_history_top_objs : (S,T) object array of (Nlayer,Nlayer) top-layer G_tt
        Returns None if not found.
        """
        path = self._cache_path_Ghist(samples=samples, cycles=cycles, nshell=nshell, init_kind=init_kind)
        if not os.path.isfile(path):
            return None

        data = np.load(path, allow_pickle=True)
        if "G_history_full_objs" in data.files:
            arr = data["G_history_full_objs"]  # (S,T) object -> (Ntot,Ntot)
            S, T = arr.shape
            out = [[arr[s, t] for t in range(T)] for s in range(S)]
            self.G_history_samples = out
            self._loaded_top_only = False
            return out

        if "G_history_top_objs" in data.files:
            # legacy top-only cache
            arr = data["G_history_top_objs"]  # (S,T) object -> (Nlayer,Nlayer)
            S, T = arr.shape
            out = [[arr[s, t] for t in range(T)] for s in range(S)]
            self.G_history_samples = out
            self._loaded_top_only = True
            return out

        return None

    def save_G_history_samples(self, histories, *, samples=None, cycles=None, nshell=None, init_kind="default"):
        """
        Save histories to a single .npz, storing both FULL and TOP for convenience.
        'histories' is list[S][T] of (Ntot,Ntot) arrays (FULL G).
        """
        S = len(histories)
        if S == 0:
            raise ValueError("save_G_history_samples: empty histories list.")
        T = len(histories[0])
        Ntot = self.Ntot
        Nlayer = Ntot // 2

        full_obj = np.empty((S, T), dtype=object)
        top_obj  = np.empty((S, T), dtype=object)

        for s in range(S):
            if len(histories[s]) != T:
                raise ValueError("save_G_history_samples: all trajectories must have the same length T.")
            for t in range(T):
                G = np.asarray(histories[s][t], dtype=np.complex128)
                if G.shape != (Ntot, Ntot):
                    raise ValueError(f"Expected FULL G of shape {(Ntot, Ntot)}, got {G.shape}.")
                full_obj[s, t] = G
                top_obj[s, t]  = G[:Nlayer, :Nlayer]

        path = self._cache_path_Ghist(samples=samples, cycles=cycles, nshell=nshell, init_kind=init_kind)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # light metadata (no seed)
        meta = {
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "Ntot": int(self.Ntot),
            "cycles": int(self.cycles if cycles is None else cycles),
            "samples": int(S),
            "nshell": ("None" if nshell is None else int(nshell)),
            "DW": bool(self.DW),
            "DW_loc": getattr(self, "DW_loc", None),
            "filling_frac": float(getattr(self, "filling_frac", 0.5)),
            "init_kind": str(init_kind),
        }

        np.savez_compressed(
            path,
            G_history_full_objs=full_obj,
            G_history_top_objs=top_obj,
            meta_json=np.array([str(meta)], dtype=object),
        )

        self.G_history_samples = [[full_obj[s, t] for t in range(T)] for s in range(S)]
        self._loaded_top_only = False
        return path

    def ensure_G_history_samples(self, *, samples=None, cycles=None, n_jobs=None, backend="loky", init_kind="default"):
        """
        Load cache or, if absent, generate histories (FULL G), save, and return list[S][T].
        """
        S = int(self.samples if samples is None else samples)
        C = int(self.cycles  if cycles  is None else cycles)

        hit = self.load_G_history_samples(samples=S, cycles=C, nshell=self.nshell, init_kind=init_kind)
        if hit is not None:
            return hit

        print("Cached histories not found. Press ENTER to generate (Ctrl-C to abort).")
        try:
            _ = input()
        except Exception:
            pass

        if n_jobs is None:
            n_jobs = 1 if S <= 1 else min(S, os.cpu_count() or 1)

        # Need OW to generate
        self.construct_OW_projectors(nshell=self.nshell, DW=self.DW)

        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()

        def _make_G0():
            return self._reset_G0_for_entanglement_contour() if init_kind == "maxmix_top" else np.array(self.G0, copy=True)

        def _worker(seed_u32):
            np.random.seed(int(seed_u32) & 0xFFFFFFFF)
            child = self._spawn_for_parallel()
            child.G0 = _make_G0()
            child.run_adaptive_circuit(
                cycles=C, G_history=True, progress=False,
                parallelize_samples=False, store="none", init_mode="default"
            )
            return [g.copy() for g in child.G_list]  # FULL G per cycle

        with self._joblib_tqdm_ctx(S, "samples"):
            histories = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_worker)(s) for s in seeds
            )

        # sync state
        self.cycles = C
        self.samples = S

        self.save_G_history_samples(histories, samples=S, cycles=C, nshell=self.nshell, init_kind=init_kind)
        return self.G_history_samples

    def assert_hist_kind(self, expected_init_kind):
        """
        Assert that the on-disk cache for current params was generated with `expected_init_kind`.
        """
        path = self._cache_path_Ghist(
            samples=self.samples,
            cycles=self.cycles,
            nshell=self.nshell,
            init_kind=expected_init_kind,
        )
        if os.path.isfile(path):
            # Optional sniff: confirm stored init_kind
            try:
                meta = np.load(path, allow_pickle=True)
                file_kind = str(meta.get("init_kind", expected_init_kind))
                if file_kind != expected_init_kind:
                    raise RuntimeError(
                        f"Cache kind mismatch: expected init_kind='{expected_init_kind}', "
                        f"but file says '{file_kind}'. Delete/regenerate."
                    )
            except Exception:
                pass
            return

        other = "maxmix_top" if expected_init_kind == "default" else "default"
        alt_path = self._cache_path_Ghist(
            samples=self.samples, cycles=self.cycles, nshell=self.nshell, init_kind=other
        )
        if os.path.isfile(alt_path):
            raise RuntimeError(
                f"No cache for init_kind='{expected_init_kind}' but found one for '{other}'. "
                "Regenerate with the desired kind."
            )
        raise RuntimeError("No cached histories found for these parameters.")
    
    def get_history_bundle(self):
        """
        Build a lightweight bundle from the cache for convenience:
          {
            "G_hist": np.array shape (S,T,Nlayer,Nlayer),
            "G_hist_avg": np.array shape (T,Nlayer,Nlayer),
            "samples": S,
            "T": T
          }
        """
        histories = self.ensure_G_history_samples(samples=self.samples, cycles=self.cycles)
        S, T = len(histories), len(histories[0])
        Nlayer = self.Ntot // 2
        G_hist = np.empty((S, T, Nlayer, Nlayer), dtype=np.complex128)
        for s in range(S):
            for t in range(T):
                Gt = np.asarray(histories[s][t])
                G_hist[s, t] = Gt[:Nlayer, :Nlayer] if Gt.shape[0] == self.Ntot else Gt
        G_hist_avg = np.mean(G_hist, axis=0)
        return {"G_hist": G_hist, "G_hist_avg": G_hist_avg, "samples": S, "T": T}

    def current_final_G(self, sample_index=0, averaged=False):
        histories = self.ensure_G_history_samples(samples=self.samples, cycles=self.cycles)
        Nlayer = self.Ntot // 2
        if averaged:
            Gbar_T = np.mean([np.asarray(histories[s][-1])[:Nlayer, :Nlayer] for s in range(len(histories))], axis=0)
            return Gbar_T
        s = int(np.clip(sample_index, 0, len(histories)-1))
        Gt = np.asarray(histories[s][-1])
        return Gt[:Nlayer, :Nlayer] if Gt.shape[0] == self.Ntot else Gt


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

    def _proj_from_WF(self, WF, Rx, Ry):
        """Build rank-1 projector χχ† on-the-fly from a stored Wannier spinor WF[:, Rx, Ry]."""
        chi = np.asarray(WF[:, Rx, Ry], dtype=np.complex128)
        # make writable (avoid read-only view issues under loky/shared arrays)
        chi = np.array(chi, copy=True)
        return np.outer(chi, chi.conj())
    
    #============================== Circuit Operations ====================================
    
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

    def top_layer_meas_feedback(self, G, Rx, Ry):
        """
        Measurement-feedback at a given center (Rx,Ry).
        """
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
        """
        Directly impose the four outcomes (no feedback unitary).
        """
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
        store="none",           # "none" (no return), "top", or "full"
        init_mode="default"     # "default" uses self.G0; "maxmix_top" uses zero top-layer via helper
    ):
        """
        Runs the adaptive circuit. Internally ALWAYS records full G per cycle when G_history=True.
        Return packaging is controlled by `store` ("none" | "top" | "full").
        Parallel mode returns stacked arrays across samples.

        Single-trajectory mode:
            - Mutates self.G and self.G_list (full G each cycle).
            - If store == "none": returns None (back-compat).
            - Else returns {"G_hist": ..., "G_hist_avg": ..., "samples": S, "T": T}
        Parallel multi-sample mode:
            - No mutation of parent instance state; returns packaged result.
        """
        # ---------------- single-trajectory (back-compat) ----------------
        if not parallelize_samples or (samples is None or int(samples) <= 1):
            # choose initial G
            if init_mode == "default":
                self.G = np.array(self.G0, copy=True)
            else:
                if init_mode == "maxmix_top":
                    self.G = np.array(self._reset_G0_for_entanglement_contour(), copy=True)
                else:
                    raise ValueError("init_mode must be 'default' or 'mm'.")

            # storage
            if G_history:
                self.G_list = []  # full G each cycle
            self.g2_flags = []

            # dimensions / loop extents
            Nx, Ny = int(self.Nx), int(self.Ny)
            D = int(self.Ntot)
            I = np.eye(D, dtype=np.complex128)

            if cycles is None:
                cycles = self.cycles
            else:
                self.cycles = int(cycles)

            total_sites = self.cycles * Nx * Ny
            use_bar = bool(progress)
            pbar = tqdm(total=total_sites, desc="RAC (sites)", unit="site", leave=True) if use_bar else None

            # main RAC loop
            for _c in range(self.cycles):
                for Rx in range(Nx):
                    for Ry in range(Ny):
                        if not postselect:
                            self.G = self.top_layer_meas_feedback(self.G, Rx, Ry)
                        else:
                            self.G = self.post_selection_top_layer(self.G, Rx, Ry)
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

            # old behavior: nothing to return unless asked
            if store == "none":
                return None

            # package a single-trajectory history (S=1)
            Ntot = self.Ntot
            Nlayer = Ntot // 2
            if G_history and len(self.G_list) > 0:
                per_cycle = self.G_list
            else:
                per_cycle = [self.G]  # final snapshot only

            if store == "top":
                hist = [np.asarray(Gk)[:Nlayer, :Nlayer] for Gk in per_cycle]
            elif store == "full":
                hist = [np.asarray(Gk) for Gk in per_cycle]
            else:
                raise ValueError("store must be 'none', 'top', or 'full'")

            G_hist = np.expand_dims(np.stack(hist, axis=0), axis=0)  # (1,T,dim,dim)
            return {
                "G_hist": G_hist,
                "G_hist_avg": np.mean(G_hist, axis=0),
                "samples": 1,
                "T": G_hist.shape[1],
            }

        # ---------------- parallel multi-sample mode ----------------
        S = int(samples)
        if n_jobs is None:
            n_jobs = min(S, os.cpu_count() or 1)
        n_jobs = max(1, int(n_jobs))

        if backend not in ("loky", "threading"):
            raise ValueError("backend must be 'loky' or 'threading'.")

        Ntot = self.Ntot
        Nlayer = Ntot // 2

        # seeds safe for process-based execution
        ss = np.random.SeedSequence()
        seeds = ss.generate_state(S, dtype=np.uint32).tolist()

        def _make_G0():
            if init_mode == "default":
                return np.array(self.G0, copy=True)
            if init_mode == "maxmix_top":
                return self._reset_G0_for_entanglement_contour()
            raise ValueError("init_mode must be 'default' or 'maxmix_top'.")

        def _worker(seed_u32):
            np.random.seed(int(seed_u32) & 0xFFFFFFFF)
            child = self._spawn_for_parallel()
            child.G0 = _make_G0()

            # run single trajectory; child will accumulate full G in child.G_list
            child.run_adaptive_circuit(
                G_history=True, tol=tol, progress=False, cycles=cycles, postselect=postselect,
                parallelize_samples=False, store="none", init_mode="default"  # child's G0 is already chosen
            )

            # collect per-cycle FULL G then downcast if needed
            full_hist = [np.asarray(Gk) for Gk in child.G_list]
            if store == "full":
                return np.stack(full_hist, axis=0)  # (T,Ntot,Ntot)
            elif store == "top":
                top_hist = [Gk[:Nlayer, :Nlayer] for Gk in full_hist]
                return np.stack(top_hist, axis=0)   # (T,Nlayer,Nlayer)
            else:
                # For parallel mode, returning nothing is not supported:
                # You asked to run multiple samples; choose 'top' or 'full'.
                raise ValueError("When parallelizing samples, set store='top' or 'full'.")

        with self._joblib_tqdm_ctx(S, "samples"):
            G_hist_list = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_worker)(seeds[i]) for i in range(S)
            )

        # stack across samples: (S,T,dim,dim)
        G_hist = np.stack(G_hist_list, axis=0)
        G_hist_avg = np.mean(G_hist, axis=0)
        return {"G_hist": G_hist, "G_hist_avg": G_hist_avg, "samples": S, "T": G_hist.shape[1]}
    
    
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

    # ------------------------- Plotting Methods -------------------------

    def plot_real_space_chern_history(self, filename=None, traj_avg=False, samples=None, n_jobs=None, backend="loky"):
        """
        Plot the real-space Chern number across time.

        Uses histories already loaded/generated at __init__ time.
        This method does NOT generate data.

        If traj_avg is False (default):
            - uses a single trajectory (first sample).
        If traj_avg is True:
            - LEFT  subplot: traj-resolved average  \overline{C_G}(t) = (1/S) sum_s C(G^{(s)}(t))
            - RIGHT subplot: traj-averaged curve   C_{Ḡ}(t) = C( (1/S) sum_s G^{(s)}(t) )

        Figures are saved if filename is provided.
        """
        self.assert_hist_kind("default")

        # top-layer histories only
        top_histories = self.get_top_histories()
        S = len(top_histories)
        if S == 0:
            raise RuntimeError("No histories available.")
        T = len(top_histories[0])

        def _top_layer(G):
            return np.asarray(G, dtype=np.complex128)

        # Compute Chern as requested
        x = np.arange(1, T + 1)
        if not traj_avg:
            cherns = np.empty(T, dtype=float)
            for t in range(T):
                cherns[t] = float(np.real(self.real_space_chern_number(_top_layer(top_histories[0][t]))))
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
        Nlayer = self.Ntot // 2
        for t in range(T):
            Cs = [float(np.real(self.real_space_chern_number(_top_layer(top_histories[s][t])))) for s in range(S)]
            C_traj_res[t] = float(np.mean(Cs))

            Gbar_t = np.mean([_top_layer(top_histories[s][t]) for s in range(S)], axis=0)
            G_full = self._block_diag2(Gbar_t, np.eye(Nlayer, dtype=np.complex128))
            C_traj_avg[t] = float(np.real(self.real_space_chern_number(G_full)))

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.0), constrained_layout=True)
        axL.plot(x, C_traj_res, marker="o", lw=1.25)
        axL.set_title(r"$\overline{C_{G}}(t)$ (traj-resolved)")
        axL.set_xlabel("Cycles"); axL.set_ylabel("Chern"); axL.grid(True, alpha=0.3)

        axR.plot(x, C_traj_avg, marker="o", lw=1.25)
        axR.set_title(r"$C_{\overline{G}}(t)$ (traj-averaged)")
        axR.set_xlabel("Cycles"); axR.set_ylabel("Chern"); axR.grid(True, alpha=0.3)

        fig.suptitle(f"Real-space Chern history (cycles={self.cycles}, samples={S})")
        if filename is not None:
            outdir = self._ensure_outdir(os.path.dirname(filename) or "figs/chern_history")
            fig.savefig(os.path.join(outdir, os.path.basename(filename)), bbox_inches="tight")
        return fig, (axL, axR), (C_traj_res, C_traj_avg)

    def chern_marker_dynamics(self, outbasename=None, vmin=-1.0, vmax=1.0, cmap='RdBu_r', traj_avg=False):
        """
        Animate local Chern marker over cached multi-sample history (read-only).

        traj_avg = False:
            - single panel using first trajectory.
        traj_avg = True:
            - two-panel animation:
                LEFT  = average over samples of marker maps: \overline{tanh C(G)} (avg of f(G))
                RIGHT = marker of per-cycle averaged G: tanh C(\overline{G})      (f of avg G)

        Saves one GIF and a final PNG frame (with cycles in the title).
        """
        self.assert_hist_kind("default")

        Nx, Ny = self.Nx, self.Ny
        outdir = self._ensure_outdir('figs/chern_marker')
        if outbasename is None:
            nshell_str = getattr(self, "nshell", None)
            nshell_str = "None" if nshell_str is None else str(nshell_str)
            outbasename = f"chern_marker_dynamics_N={Nx}_nshell={nshell_str}_cycles={getattr(self,'cycles','NA')}_DWis{getattr(self,'DW','NA')}"
        gif_path   = os.path.join(outdir, outbasename + ".gif")
        final_path = os.path.join(outdir, outbasename + "_final.png")

        # read-only top histories
        top_histories = self.get_top_histories()
        S = len(top_histories)
        if S == 0:
            raise RuntimeError("No histories available.")
        T = len(top_histories[0])

        def _top_layer(G):
            return np.asarray(G, dtype=np.complex128)

        if not traj_avg:
            frames = []
            for t in range(T):
                Cmap = self.local_chern_marker_flat(_top_layer(top_histories[0][t]))
                frames.append(Cmap)

            fig = plt.figure(figsize=(3.6, 4.0))
            ax  = fig.add_subplot(111)
            im  = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
            for sp in ax.spines.values():
                sp.set_linewidth(1.5); sp.set_color('black')
            ax.set_xlabel("y"); ax.set_ylabel("x")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Local Chern marker (cycles={self.cycles})")

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
            ax2.set_title(f"Local Chern marker — final (cycles={self.cycles})")
            fig2.savefig(final_path, bbox_inches='tight', dpi=140); plt.close(fig2)
            return gif_path, final_path, final, top_histories[0][-1]

        # traj_avg=True -> two-panel animation
        frames_L, frames_R = [], []
        Nlayer = self.Ntot // 2
        for t in range(T):
            maps = [self.local_chern_marker_flat(_top_layer(top_histories[s][t])) for s in range(S)]
            frames_L.append(np.mean(maps, axis=0))
            Gbar_t = np.mean([_top_layer(top_histories[s][t]) for s in range(S)], axis=0)
            G_full = self._block_diag2(Gbar_t, np.eye(Nlayer, dtype=np.complex128))
            frames_R.append(self.local_chern_marker_flat(G_full))

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
        fig.suptitle(f"Local Chern marker (cycles={self.cycles}, samples={S})")

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
        fig2.suptitle(f"Local Chern marker — final (cycles={self.cycles}, samples={S})")
        fig2.savefig(final_path, bbox_inches='tight', dpi=140); plt.close(fig2)
        return gif_path, final_path, (frames_L[-1], frames_R[-1]), (_top_layer(top_histories[0][-1]),)

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
        child.cycles = self.cycles
        child.samples = self.samples
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
        child.G0 = np.array(self.G0, copy=True)
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

    def plot_corr_y_profiles_v2(self, samples=None, n_jobs=None, filename=None, ry_max=None, backend="loky"):
        """
        Read-only: uses already saved histories.
        ALWAYS plots both:
          - \overline{C}_G   (traj-resolved average of profiles)
          - C_{\overline{G}} (profile of the per-cycle averaged G)
        Also produces a separate 2x2 heatmap figure vs cycle t.
        Only figures are saved (no npz).
        """
        Nx, Ny = self.Nx, self.Ny
        Ntot   = self.Ntot
        Nlayer = Ntot // 2
        cycles = int(getattr(self, "cycles", 1))
        
        self.assert_hist_kind("default")

        # top histories
        top_histories = self.get_top_histories()
        S = len(top_histories)
        if S == 0:
            raise RuntimeError("No histories available.")
        T = len(top_histories[0])

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

        x_positions = _pick_x_positions()

        # --- DW positions for heatmaps ---
        if not (hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2):
            raise RuntimeError("plot_corr_y_profiles_v2: DW_loc (two positions) is required for the heatmaps.")
        xDW1, xDW2 = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx

        # -------------- helpers --------------
        def _coerce_to_two_point_kernel_top(G_in):
            Gin = np.asarray(G_in, dtype=np.complex128)
            if Gin.ndim == 6:
                return Gin
            elif Gin.ndim == 2:
                if Gin.shape == (Nlayer, Nlayer):
                    Gtt = Gin
                elif Gin.shape == (Ntot, Ntot):
                    Gtt = Gin[:Nlayer, :Nlayer]
                else:
                    raise ValueError(f"G has incompatible shape {Gin.shape}.")
                G2 = 0.5 * (Gtt + np.eye(Nlayer, dtype=np.complex128))
                G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order='F')
                return np.transpose(G6, (1, 2, 0, 4, 5, 3))
            else:
                raise ValueError("G must be (4N,4N), (2N,2N), or (Nx,Ny,2,Nx,Ny,2).")

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
        Gsum_fin = np.zeros((Nlayer, Nlayer), dtype=np.complex128)
        last_Gtt = None

        for s in range(S):
            Gtt_final = np.asarray(top_histories[s][-1], dtype=np.complex128)
            last_Gtt = Gtt_final
            Gker_fin = _coerce_to_two_point_kernel_top(Gtt_final)
            for x0, _ in x_positions:
                C_accum[x0] += _C_xslice_from_kernel(Gker_fin, x0, ry_vals).real
            Gsum_fin += Gtt_final

        C_resolved = {x0: C_accum[x0] / S for x0, _ in x_positions}
        Gavg_fin   = Gsum_fin / S
        C_avg      = {x0: _C_xslice_from_kernel(_coerce_to_two_point_kernel_top(Gavg_fin), x0, ry_vals).real
                      for x0, _ in x_positions}

        # spectra & Chern (final step)
        evals_last = np.linalg.eigvalsh(last_Gtt)
        evals_avg  = np.linalg.eigvalsh(Gavg_fin)
        Chern_last = _chern_from_Gtop(last_Gtt)
        Chern_avg  = _chern_from_Gtop(Gavg_fin)

        # ---------------- plotting: 3 x 2 ----------------
        outdir = self._ensure_outdir('figs/corr_y_profiles')
        if filename is None:
            xdesc = "-".join(f"{x}" for x, _ in x_positions)
            filename = f"corr2_y_profiles_v2_N{Nx}_xs_{xdesc}_S{S}.pdf"
        fullpath = os.path.join(outdir, filename)

        mpl.rcParams['text.usetex'] = False

        fig = plt.figure(figsize=(12.5, 12.0), constrained_layout=True)
        gs  = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, 1.04])

        axC1 = fig.add_subplot(gs[0, 0])
        axC2 = fig.add_subplot(gs[0, 1])
        axE1 = fig.add_subplot(gs[1, 0])
        axE2 = fig.add_subplot(gs[1, 1])
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
        ymin = min(axC1.get_ylim()[0], axC2.get_ylim()[0])
        ymax = max(axC1.get_ylim()[1], axC2.get_ylim()[1])
        axC1.set_ylim(ymin, ymax)
        axC2.set_ylim(ymin, ymax)

        # Middle row: eigenvalue spectra
        axE1.plot(np.arange(len(evals_last)), np.sort(evals_last), '.', ms=3)
        axE1.set_title(r"eigvals($G_{\mathrm{final}}$)")
        axE1.set_xlabel("index"); axE1.set_ylabel("eigenvalue"); axE1.grid(True, alpha=0.3)

        axE2.plot(np.arange(len(evals_avg)),  np.sort(evals_avg),  '.', ms=3)
        axE2.set_title(r"eigvals($\overline{G}_{\mathrm{final}}$)")
        axE2.set_xlabel("index"); axE2.set_ylabel("eigenvalue"); axE2.grid(True, alpha=0.3)

        # Bottom row: Chern maps
        im1 = axM1.imshow(Chern_last, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
        axM1.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for final $G$")
        axM1.set_xlabel("y"); axM1.set_ylabel("x"); axM1.grid(False)
        fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)

        im2 = axM2.imshow(Chern_avg,  cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
        axM2.set_title(r"$\tanh\mathcal{C}(\mathbf{r})$ for $\overline{G}$")
        axM2.set_xlabel("y"); axM2.set_ylabel("x"); axM2.grid(False)
        fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04)

        fig.suptitle(f"Correlation profiles & spectra (cycles={cycles}, samples={S})")
        fig.savefig(fullpath, bbox_inches='tight'); plt.close(fig)

        # ---------------- 2x2 heatmaps vs cycle t ----------------
        def _C_time_traj_resolved(x0):
            out = np.zeros((T, len(ry_vals)), dtype=float)
            for t in range(T):
                vals = []
                for s in range(S):
                    Gker = _coerce_to_two_point_kernel_top(top_histories[s][t])
                    vals.append(_C_xslice_from_kernel(Gker, x0, ry_vals).real)
                out[t, :] = np.mean(vals, axis=0)
            return out

        def _C_time_traj_averaged(x0):
            out = np.zeros((T, len(ry_vals)), dtype=float)
            for t in range(T):
                Gavg_t = np.mean([np.asarray(top_histories[s][t]) for s in range(S)], axis=0)
                Gker = _coerce_to_two_point_kernel_top(Gavg_t)
                out[t, :] = _C_xslice_from_kernel(Gker, x0, ry_vals).real
            return out

        H_trajres_1 = _C_time_traj_resolved(xDW1)
        H_trajres_2 = _C_time_traj_resolved(xDW2)
        H_trajavg_1 = _C_time_traj_averaged(xDW1)
        H_trajavg_2 = _C_time_traj_averaged(xDW2)

        vmax_heat = max(np.max(H_trajres_1), np.max(H_trajres_2), np.max(H_trajavg_1), np.max(H_trajavg_2))
        vmin_heat = 0.0

        heatmaps_name = f"corr2_y_profiles_v2_HEATMAPS_N{Nx}_DW_{xDW1}_{xDW2}_S{S}.pdf"
        heatmaps_path = os.path.join(outdir, heatmaps_name)

        fig2 = plt.figure(figsize=(12.0, 8.8), constrained_layout=True)
        gs2 = fig2.add_gridspec(nrows=2, ncols=2)

        axH11 = fig2.add_subplot(gs2[0, 0])
        axH12 = fig2.add_subplot(gs2[0, 1])
        axH21 = fig2.add_subplot(gs2[1, 0])
        axH22 = fig2.add_subplot(gs2[1, 1])

        extent = [ry_vals.min(), ry_vals.max(), 1, T]  # x=r_y, y=t

        im11 = axH11.imshow(H_trajres_1, cmap="Blues", origin="upper", aspect="auto",
                            vmin=vmin_heat, vmax=vmax_heat, extent=extent)
        axH11.set_title(fr"traj-resolved $\overline{{C}}_G$ @ $x_0={xDW1}$")
        axH11.set_xlabel(r"$r_y$"); axH11.set_ylabel("cycle $t$")
        fig2.colorbar(im11, ax=axH11, fraction=0.046, pad=0.04)

        im12 = axH12.imshow(H_trajavg_1, cmap="Blues", origin="upper", aspect="auto",
                            vmin=vmin_heat, vmax=vmax_heat, extent=extent)
        axH12.set_title(fr"$C_{{\overline{{G}}}}$ @ $x_0={xDW1}$")
        axH12.set_xlabel(r"$r_y$"); axH12.set_ylabel("cycle $t$")
        fig2.colorbar(im12, ax=axH12, fraction=0.046, pad=0.04)

        im21 = axH21.imshow(H_trajres_2, cmap="Blues", origin="upper", aspect="auto",
                            vmin=vmin_heat, vmax=vmax_heat, extent=extent)
        axH21.set_title(fr"traj-resolved $\overline{{C}}_G$ @ $x_0={xDW2}$")
        axH21.set_xlabel(r"$r_y$"); axH21.set_ylabel("cycle $t$")
        fig2.colorbar(im21, ax=axH21, fraction=0.046, pad=0.04)

        im22 = axH22.imshow(H_trajavg_2, cmap="Blues", origin="upper", aspect="auto",
                            vmin=vmin_heat, vmax=vmax_heat, extent=extent)
        axH22.set_title(fr"$C_{{\overline{{G}}}}$ @ $x_0={xDW2}$")
        axH22.set_xlabel(r"$r_y$"); axH22.set_ylabel("cycle $t$")
        fig2.colorbar(im22, ax=axH22, fraction=0.046, pad=0.04)

        fig2.suptitle(f"Correlation y-profiles vs cycle (cycles={cycles}, samples={S})", y=1.02)
        fig2.savefig(heatmaps_path, bbox_inches="tight", dpi=140)
        plt.close(fig2)

        return fullpath
    
    # ------------------ Entanglement Contour Block ------------------

    def _reset_G0_for_entanglement_contour(self):
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
        return self._block_diag2(Gtt, Gbb)
       

    def entanglement_contour(self, Gtt):
        """
        Compute site-resolved entanglement contour s(r) for top layer covariance Gtt.
        Returns array of shape (Nx, Ny).
        """
        Nx, Ny = self.Nx, self.Ny
        Nlayer = Gtt.shape[0]

        I  = np.eye(Nlayer, dtype=np.complex128)
        G2 = 0.5 * (I + np.asarray(Gtt, dtype=np.complex128))

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

    def entanglement_contour_suite(self, samples=None, n_jobs=None, backend="loky",
                                filename_profiles=None, filename_prefix_dyn=None):
        """
        Entanglement-contour analysis using saved histories (read-only).

        Always shows:
        Row 1: time-profiles at smart x0 — LEFT = traj-resolved \overline{s_G}(t), RIGHT = s_{Ḡ}(t)
        Row 2: eigenvalue spectra (final-step) — traj vs Ḡ
        Row 3: final maps — s(G_final^{(traj)}) vs s(Ḡ_final)

        Also produces two GIFs (traj map & traj-avg map) with colorbars.
        """
        self.assert_hist_kind("maxmix_top")

        Nx, Ny = self.Nx, self.Ny
        Nlayer = self.Ntot // 2
        cycles = int(getattr(self, "cycles", 1))

        # top histories
        top_histories = self.get_top_histories()
        S = len(top_histories)
        if S == 0:
            raise RuntimeError("No histories available.")
        T = len(top_histories[0])

        # smart x-positions
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
        xs_only = [x for x, _ in x_positions]

        def _contour_from_G(Gtt):
            return self.entanglement_contour(np.asarray(Gtt, dtype=np.complex128))

        # Build time-profiles
        traj_profiles_mean = {
            x0: np.array([
                np.mean([float(np.sum(_contour_from_G(top_histories[s][t])[x0, :])) for s in range(S)])
                for t in range(T)
            ], dtype=float)
            for x0 in xs_only
        }

        Gavg_hist = [np.mean([np.asarray(top_histories[s][t], dtype=np.complex128) for s in range(S)], axis=0) for t in range(T)]
        avg_maps  = [_contour_from_G(Gavg_hist[t]) for t in range(T)]

        avg_profiles = {
            x0: np.array([float(np.sum(avg_maps[t][x0, :])) for t in range(T)], dtype=float)
            for x0 in xs_only
        }

        # spectra + final maps
        G_final_traj = np.asarray(top_histories[0][-1], dtype=np.complex128)
        G_final_avg  = Gavg_hist[-1]
        ev_traj = np.linalg.eigvalsh(G_final_traj)
        ev_avg  = np.linalg.eigvalsh(G_final_avg)
        final_traj_map = _contour_from_G(G_final_traj)
        final_avg_map  = _contour_from_G(G_final_avg)

        # --------------- profiles + spectra + final maps ---------------
        outdir_prof = self._ensure_outdir("figs/entanglement_contour")
        if filename_profiles is None:
            xdesc = "-".join(f"{x}" for x in xs_only)
            filename_profiles = f"entanglement_suite_yprofiles_N{Nx}_xs_{xdesc}_S{S}.pdf"
        profiles_pdf = os.path.join(outdir_prof, filename_profiles)

        fig = plt.figure(constrained_layout=True, figsize=(12.5, 12.0))
        gs  = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1.1, 1.0, 1.2])

        axP1 = fig.add_subplot(gs[0, 0]); axP2 = fig.add_subplot(gs[0, 1])
        t_vals = np.arange(1, T + 1)

        for x0, lbl in x_positions:
            axP1.plot(t_vals, traj_profiles_mean[x0], label=lbl, marker='o', ms=3)
        axP1.set_xlabel("cycle t"); axP1.set_ylabel(r"$\sum_y s(x_0,y)$")
        axP1.set_title(r"$\overline{s_{G}}$ (traj-resolved mean)")
        axP1.set_yscale("log"); axP1.grid(True, alpha=0.3); axP1.legend(fontsize=7, ncol=2)

        for x0, lbl in x_positions:
            axP2.plot(t_vals, avg_profiles[x0], label=lbl, marker='o', ms=3)
        axP2.set_xlabel("cycle t"); axP2.set_ylabel(r"$\sum_y s(x_0,y)$")
        axP2.set_title(r"$s_{\overline{G}}$"); axP2.set_yscale("log"); axP2.grid(True, alpha=0.3); axP2.legend(fontsize=7, ncol=2)

        y1 = axP1.get_ylim(); y2 = axP2.get_ylim()
        ymin = min(y1[0], y2[0]); ymax = max(y1[1], y2[1])
        axP1.set_ylim(ymin, ymax); axP2.set_ylim(ymin, ymax)

        axS1 = fig.add_subplot(gs[1, 0]); axS2 = fig.add_subplot(gs[1, 1])
        axS1.plot(np.arange(len(ev_traj)), np.sort(ev_traj), '.', ms=3); axS1.grid(True, alpha=0.3)
        axS2.plot(np.arange(len(ev_avg)),  np.sort(ev_avg),  '.', ms=3); axS2.grid(True, alpha=0.3)
        axS1.set_title(r"eigvals($G_{\mathrm{final}}$) (traj)"); axS2.set_title(r"eigvals($\overline{G}_{\mathrm{final}}$)")
        axS1.set_xlabel("index"); axS1.set_ylabel("eigenvalue")
        axS2.set_xlabel("index"); axS2.set_ylabel("eigenvalue")

        axM1 = fig.add_subplot(gs[2, 0]); axM2 = fig.add_subplot(gs[2, 1])
        im1 = axM1.imshow(final_traj_map, cmap="Blues", origin="upper", aspect="equal")
        im2 = axM2.imshow(final_avg_map,  cmap="Blues", origin="upper", aspect="equal")
        for ax in (axM1, axM2):
            ax.set_xlabel("y"); ax.set_ylabel("x")
        fig.colorbar(im1, ax=axM1, fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axM2, fraction=0.046, pad=0.04)
        axM1.set_title("Final $s_G$ (traj)"); axM2.set_title(r"Final $s_{\overline{G}}$")

        fig.suptitle(
            f"Entanglement contour (cycles={cycles}, samples={S}, backend={backend})\n"
            r"Initial top layer: maximally mixed ($G_{tt}=0$)",
            y=1.02
        )
        fig.savefig(profiles_pdf, bbox_inches="tight", dpi=140); plt.close(fig)

        # --------------- dynamics GIFs ---------------
        outdir_dyn = self._ensure_outdir("figs/entanglement_contour_dynamics")
        if filename_prefix_dyn is None:
            filename_prefix_dyn = f"entanglement_dyn_N{Nx}_S{S}_{backend}"

        traj_maps_for_gif = [self.entanglement_contour(np.asarray(top_histories[0][t], dtype=np.complex128)) for t in range(T)]

        dyn_final_png = os.path.join(outdir_dyn, f"{filename_prefix_dyn}_final.png")
        figF = plt.figure(constrained_layout=True, figsize=(10, 4))
        axsF = figF.subplots(1, 2, squeeze=True)
        imf1 = axsF[0].imshow(traj_maps_for_gif[-1], cmap="Blues", origin="upper", aspect="equal")
        imf2 = axsF[1].imshow(traj_maps_for_gif[-1] * 0 + avg_maps[-1], cmap="Blues", origin="upper", aspect="equal")  # reuse avg_maps
        for ax in axsF:
            ax.set_xlabel("y"); ax.set_ylabel("x")
        figF.colorbar(imf1, ax=axsF[0], fraction=0.046, pad=0.04)
        figF.colorbar(imf2, ax=axsF[1], fraction=0.046, pad=0.04)
        figF.suptitle(r"Dynamics final frames — initial top layer maximally mixed ($G_{tt}=0$)", y=1.02)
        figF.savefig(dyn_final_png, dpi=150, bbox_inches="tight"); plt.close(figF)

        def _make_gif(maps, fname, title):
            fig = plt.figure(constrained_layout=True, figsize=(5.6, 4.6))
            ax = fig.add_subplot(111)
            vmax = np.max(maps)
            im = ax.imshow(maps[0], cmap="Blues", origin="upper", aspect="equal", vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title); ax.set_xlabel("y"); ax.set_ylabel("x")
            def update(i):
                im.set_data(maps[i]); ax.set_title(f"{title}, cycle {i+1}")
                return [im]
            ani = animation.FuncAnimation(fig, update, frames=len(maps), interval=400, blit=True)
            ani.save(os.path.join(outdir_dyn, fname), writer="pillow", dpi=120)
            plt.close(fig)

        dyn_gif_traj = f"{filename_prefix_dyn}_traj.gif"
        dyn_gif_avg  = f"{filename_prefix_dyn}_avg.gif"
        _make_gif(traj_maps_for_gif, dyn_gif_traj, r"$s_G(r,t)$ (traj-resolved)")
        _make_gif(avg_maps,          dyn_gif_avg,  r"$s_{\overline{G}}(r,t)$ (traj-averaged)")

        return {
            "profiles_pdf": profiles_pdf,
            "dyn_dir": outdir_dyn,
            "dyn_gif_traj": os.path.join(outdir_dyn, dyn_gif_traj),
            "dyn_gif_avg":  os.path.join(outdir_dyn, dyn_gif_avg),
            "dyn_final_png": dyn_gif_traj.replace("_traj.gif", "_final.png"),
        }
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