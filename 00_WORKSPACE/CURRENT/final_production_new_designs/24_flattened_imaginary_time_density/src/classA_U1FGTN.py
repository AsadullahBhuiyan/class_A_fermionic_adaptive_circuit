import numpy as np
import os
import math
import time
import json
import copy
import hashlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
import joblib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from contextlib import contextmanager, nullcontext
from scipy.optimize import curve_fit
from contextlib import contextmanager

try:
    from .occupied_frame import (
        FRAME_REPRESENTATIONS,
        FRAME_ALGORITHM_VERSION,
        OccupiedFrameState,
        UpdateTimingCollector,
    )
except ImportError:  # Support historical direct imports from src/fgtn on PYTHONPATH.
    from occupied_frame import (  # type: ignore
        FRAME_REPRESENTATIONS,
        FRAME_ALGORITHM_VERSION,
        OccupiedFrameState,
        UpdateTimingCollector,
    )


class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
    """Update tqdm whenever a joblib batch finishes."""
    def __init__(self, tqdm_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object
    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        if getattr(self.tqdm_object, "_show_datetime", False):
            self.tqdm_object.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
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

    @staticmethod
    def _normalize_nshell(nshell):
        if nshell is None:
            return None
        value = float(nshell)
        if value < 0:
            raise ValueError("nshell must be None, a nonnegative integer, or 0.5.")
        if np.isclose(value, 0.5):
            return 0.5
        rounded = int(round(value))
        if np.isclose(value, rounded):
            return rounded
        raise ValueError("nshell must be None, a nonnegative integer, or 0.5.")

    @classmethod
    def _ow_support_stride(cls, nshell):
        nsh = cls._normalize_nshell(nshell)
        if nsh is None:
            return 1
        if np.isclose(nsh, 0.5):
            return 3
        return 2 * int(nsh) + 1

    @classmethod
    def _ow_support_mask(cls, dxw, dyw, nshell):
        nsh = cls._normalize_nshell(nshell)
        if nsh is None:
            return np.ones_like(dxw, dtype=bool)
        if np.isclose(nsh, 0.5):
            return (
                ((dxw == 0) & (dyw == 0))
                | ((np.abs(dxw) == 1) & (dyw == 0))
                | ((dxw == 0) & (np.abs(dyw) == 1))
            )
        return (np.abs(dxw) <= int(nsh)) & (np.abs(dyw) <= int(nsh))

    def __init__(
        self,
        Nx,
        Ny,
        DW=True,
        nshell=None,
        filling_frac=1 / 2,
        G0=None,
        alpha_1 = 1,
        alpha_2 = 30,
        trial_orbitals="X",
        dw_truncation=False,
        twist_y=0.0,
        dw_interval=None,
        *,
        twist_x=0.0,
    ):
        '''Initialize lattice dimensions, domain-wall option, and starting covariance.'''

        self.time_init = time.time()
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.Ntot = 4 * self.Nx * self.Ny
        self.Nlayer = self.Ntot // 2
        self.filling_frac = filling_frac
        self.nshell = self._normalize_nshell(nshell)
        self.DW = bool(DW)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_top = self.alpha_1
        self.alpha_triv = self.alpha_2
        self.trial_orbitals = str(trial_orbitals).upper()
        self.dw_truncation = bool(dw_truncation)
        self.dw_interval = self._normalize_dw_interval(dw_interval, self.Nx)
        self.twist_x = float(twist_x)
        self.twist_y = float(twist_y)
        if not np.isfinite(self.twist_x):
            raise ValueError("twist_x must be a finite scalar flux in radians.")
        if not np.isfinite(self.twist_y):
            raise ValueError("twist_y must be a finite scalar flux in radians.")
        if self.DW:
            self.create_domain_wall(
                alpha_1=self.alpha_1,
                alpha_2=self.alpha_2,
                dw_interval=self.dw_interval,
            )
        else:
            if self.dw_interval is not None:
                raise ValueError("dw_interval requires DW=True.")
            # Uniform mass profile when no DW is requested
            self.alpha_profile = self.alpha_1 * np.ones((self.Nx, self.Ny), dtype=np.complex128)
            self.alpha = self.alpha_profile

        self.G0 = None if G0 is None else np.array(G0, dtype=np.complex128, copy=True)
        self.G = None
        self.G_history_samples = None
        self._physical_covariance_update_mode = "rank1"

        print("------------------------- classA_U1FGTN Initialized -------------------------")

    @staticmethod
    def _normalize_dw_interval(dw_interval, Nx):
        """Validate an optional inclusive, non-wrapping domain-wall slab interval."""
        if dw_interval is None:
            return None
        if isinstance(dw_interval, (str, bytes)):
            raise ValueError("dw_interval must be a two-integer inclusive interval.")
        try:
            values = tuple(dw_interval)
        except TypeError as exc:
            raise ValueError("dw_interval must be a two-integer inclusive interval.") from exc
        if len(values) != 2:
            raise ValueError("dw_interval must contain exactly two endpoints.")
        normalized = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("dw_interval endpoints must be integers.")
            try:
                endpoint = int(value)
            except Exception as exc:
                raise ValueError("dw_interval endpoints must be integers.") from exc
            if isinstance(value, (float, np.floating)) and not float(value).is_integer():
                raise ValueError("dw_interval endpoints must be integers.")
            normalized.append(endpoint)
        x0, x1 = normalized
        if not (0 <= x0 <= x1 < int(Nx)):
            raise ValueError(
                f"dw_interval must satisfy 0 <= x0 <= x1 < Nx={int(Nx)}; "
                f"got {(x0, x1)}."
            )
        return (x0, x1)

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
    def _joblib_tqdm_ctx(self, total, desc, show_datetime=False):
        """
        Single outer tqdm bar for joblib.Parallel.
        Use as: with self._joblib_tqdm_ctx(samples, "samples"): Parallel(...).
        """
        if not (total and total > 1):
            return nullcontext()
        pbar = tqdm(total=total, desc=desc, unit="task")
        pbar._show_datetime = bool(show_datetime)
        if show_datetime:
            pbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
        return tqdm_joblib(pbar)
    
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

    def _g_history_outdir(self):
        size_key = f"N{int(self.Nx)}x{int(self.Ny)}"
        return self._ensure_outdir(os.path.join("cache", "G_history_samples", size_key))

    def _g_history_outdir_rel(self):
        size_key = f"N{int(self.Nx)}x{int(self.Ny)}"
        return os.path.join("cache", "G_history_samples", size_key)
    
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

    def _normalize_dw_exclude(self, val):
        """
        Treat dw_exclude=False as None so it is ignored; otherwise return val.
        """
        return None if val is False else val

    @staticmethod
    def _validate_probability(value, name):
        try:
            prob = float(value)
        except Exception as exc:
            raise ValueError(f"{name} must be a finite probability in [0, 1].") from exc
        if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            raise ValueError(f"{name} must be a finite probability in [0, 1].")
        return prob

    @classmethod
    def _resolve_feedback_probabilities(cls, n_a=0.5, p_gain=None, p_loss=None):
        """
        Resolve stochastic correction probabilities.

        Legacy n_a is the probability of drawing an occupied ancilla. Therefore
        it maps to gain success probability n_a and loss success probability
        1 - n_a when the new process-specific probabilities are omitted.
        """
        n_a_eff = cls._validate_probability(n_a, "n_a")
        p_gain_eff = n_a_eff if p_gain is None else cls._validate_probability(p_gain, "p_gain")
        p_loss_eff = (1.0 - n_a_eff) if p_loss is None else cls._validate_probability(p_loss, "p_loss")
        return n_a_eff, p_gain_eff, p_loss_eff

    def _block_diag2(self, A, B):
        """
        Minimal block_diag(A,B) without scipy. Returns [[A,0],[0,B]].
        """
        n, m = A.shape[0], B.shape[0]
        Z1 = np.zeros((n, m), dtype=A.dtype)
        Z2 = np.zeros((m, n), dtype=B.dtype)
        return np.block([[A, Z1],
                         [Z2, B]])

    def _get_top_eye(self):
        """
        Cache and return top-layer identity matrix to avoid repeated allocations
        in hot update paths.
        """
        Nlayer = self.Ntot // 2
        cache = getattr(self, "_I_top_cache", None)
        if cache is None or cache.shape != (Nlayer, Nlayer):
            cache = np.eye(Nlayer, dtype=np.complex128)
            self._I_top_cache = cache
        return cache

    def _eye_of_size(self, size):
        """Cache identities used by reduced-basis Choi contractions."""
        size = int(size)
        cache = getattr(self, "_eye_size_cache", None)
        if cache is None:
            cache = {}
            self._eye_size_cache = cache
        eye = cache.get(size)
        if eye is None:
            eye = np.eye(size, dtype=np.complex128)
            cache[size] = eye
        return eye

    def _meas_slab_only_effective(self, meas_slab_only):
        return bool(meas_slab_only and self.DW and self.dw_truncation)

    def active_top_layer_indices(self, meas_slab_only=True):
        """Return canonical top-layer indices included in adaptive cycle updates."""
        Nlayer = self.Ntot // 2
        if not self._meas_slab_only_effective(meas_slab_only):
            return np.arange(Nlayer, dtype=np.int64)
        x_min, x_max = sorted(int(x) for x in self.DW_loc)
        indices = [
            mu + 2 * x + 2 * self.Nx * y
            for y in range(self.Ny)
            for x in range(x_min, x_max + 1)
            for mu in (0, 1)
        ]
        return np.asarray(indices, dtype=np.int64)

    def restrict_ow_dynamics_to_slab(self):
        """Return an exact active-space OW model plus original physical indices.

        Only valid for support-terminated walls. The OW vectors are constructed
        on the ORIGINAL lattice and then restricted, never regenerated on a
        narrower periodic lattice. Both channel and Lindblad canonical entry
        points can evolve this block; a separately prepared exterior stays fixed.
        Initial states must be restricted separately by the caller. This helper
        does not perform a measurement or change the original model.
        """
        if not self._meas_slab_only_effective(True):
            raise ValueError('slab restriction requires DW=True and dw_truncation=True')
        names = ('WF_Ap', 'WF_Am', 'WF_Bp', 'WF_Bm')
        if not all(hasattr(self, name) for name in names):
            self.construct_OW_projectors(nshell=self.nshell, DW=True,
                trial_orbitals=self.trial_orbitals, dw_truncation=True,
                twist_y=self.twist_y)
        left, right = sorted(map(int, self.DW_loc))
        active = self.active_top_layer_indices(meas_slab_only=True)
        outside = np.setdiff1d(np.arange(self.Nlayer), active)
        child = classA_U1FGTN(right-left+1, self.Ny, DW=False,
            nshell=self.nshell, alpha_1=self.alpha_1, alpha_2=self.alpha_2,
            trial_orbitals=self.trial_orbitals, twist_y=self.twist_y)
        for name in names:
            source = np.asarray(getattr(self, name))[:, left:right+1, :]
            if outside.size and np.max(np.abs(source[outside])) > 1e-12:
                raise ValueError('active OW vectors leak into the exterior')
            selected = np.ascontiguousarray(source[active])
            if np.max(abs(np.sum(abs(selected)**2, axis=0)-1)) > 1e-10:
                raise ValueError('restricted OW vectors lost normalization')
            setattr(child, name, selected)
        child._ow_local_support_cache = {}
        child.restricted_ow_source = dict(Nx=self.Nx, Ny=self.Ny,
            wall_locations=[left, right], dw_truncation=True,
            meas_slab_only=True, construction='restriction_of_original_OW_arrays')
        return child, active

    def _exterior_site_coordinates(self):
        if not (self.DW and hasattr(self, "DW_loc") and len(self.DW_loc) == 2):
            return []
        x_min, x_max = sorted(int(x) for x in self.DW_loc)
        return [
            (x, y)
            for x in range(self.Nx)
            for y in range(self.Ny)
            if not (x_min <= x <= x_max)
        ]

    @staticmethod
    def _rng_random(rng=None):
        """Draw one U[0,1) variate while preserving legacy global-RNG behavior."""
        return float(np.random.rand()) if rng is None else float(rng.random())

    def _prepare_exterior_product_state(self, G, mode="born_conditioned", rng=None):
        """Prepare exterior canonical orbitals once without OW feedback.

        ``born_conditioned`` sequentially samples the two local orbitals in every
        exterior unit cell. ``forced_occupied`` is the fully postselected convention.
        """
        mode = str(mode).strip().lower()
        if mode not in ("born_conditioned", "forced_occupied"):
            raise ValueError(
                "exterior preparation mode must be 'born_conditioned' or "
                "'forced_occupied'."
            )
        if isinstance(G, OccupiedFrameState):
            chi_local = np.ones((1,), dtype=np.complex128)
            for Rx, Ry in self._exterior_site_coordinates():
                base_idx = 2 * int(Rx) + 2 * self.Nx * int(Ry)
                for idx in (base_idx, base_idx + 1):
                    support_idx = np.asarray([idx], dtype=np.int64)
                    if mode == "forced_occupied":
                        particle = True
                    else:
                        particle = bool(
                            self._rng_random(rng)
                            < G.occupation_probability_local(support_idx, chi_local)
                        )
                    if particle:
                        G.project_occupied_local(support_idx, chi_local)
                    else:
                        G.project_empty_local(support_idx, chi_local)
            return G
        G_top = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        chi_local = np.ones((1,), dtype=np.complex128)
        for Rx, Ry in self._exterior_site_coordinates():
            base_idx = 2 * int(Rx) + 2 * self.Nx * int(Ry)
            for idx in (base_idx, base_idx + 1):
                support_idx = np.asarray([idx], dtype=np.int64)
                comp_idx = self._complement_indices(support_idx, Nlayer)
                if mode == "forced_occupied":
                    particle = True
                else:
                    occ_prob = float(np.clip(0.5 * (1.0 + np.real(G_top[idx, idx])), 0.0, 1.0))
                    particle = bool(self._rng_random(rng) < occ_prob)
                G_top = self._measure_only_top_layer_local(
                    G_top, support_idx, comp_idx, chi_local, particle=particle
                )
        return G_top

    def _get_local_basis_vectors(self, Rx, Ry):
        """
        Cache canonical top-layer basis vectors at a site (Rx, Ry), used by local_mode.
        """
        key = (int(Rx) % self.Nx, int(Ry) % self.Ny)
        cache = getattr(self, "_local_basis_cache", None)
        if cache is None:
            cache = {}
            self._local_basis_cache = cache
        if key in cache:
            return cache[key]

        Nlayer = self.Ntot // 2
        idx_mu1 = 0 + 2 * key[0] + 2 * self.Nx * key[1]
        idx_mu2 = 1 + 2 * key[0] + 2 * self.Nx * key[1]
        e_mu1 = np.zeros(Nlayer, dtype=np.complex128)
        e_mu2 = np.zeros(Nlayer, dtype=np.complex128)
        e_mu1[idx_mu1] = 1.0
        e_mu2[idx_mu2] = 1.0
        cache[key] = (e_mu1, e_mu2)
        return cache[key]

    def _get_local_mode_ops(self, Rx, Ry):
        """
        Cache local-mode basis vectors and rank-1 projectors for canonical mu orbitals.
        """
        key = (int(Rx) % self.Nx, int(Ry) % self.Ny)
        cache = getattr(self, "_local_mode_ops_cache", None)
        if cache is None:
            cache = {}
            self._local_mode_ops_cache = cache
        if key in cache:
            return cache[key]

        e_mu1, e_mu2 = self._get_local_basis_vectors(*key)
        Il = self._get_top_eye()
        P_mu1 = np.outer(e_mu1, e_mu1.conj())
        P_mu2 = np.outer(e_mu2, e_mu2.conj())
        Q_mu1 = Il - P_mu1
        Q_mu2 = Il - P_mu2
        cache[key] = (e_mu1, e_mu2, P_mu1, P_mu2, Q_mu1, Q_mu2)
        return cache[key]

    def _get_ow_local_data(self, Rx, Ry):
        """
        Cache per-site OW spinors used in Markov updates.
        NOTE: Do not cache dense projector matrices per site; that scales as
        O(Nsite * Nlayer^2) memory and can OOM under parallel workers.
        """
        key = (int(Rx) % self.Nx, int(Ry) % self.Ny)
        cache = getattr(self, "_ow_local_cache", None)
        if cache is None:
            cache = {}
            self._ow_local_cache = cache
        if key in cache:
            return cache[key]

        have_ow = all(hasattr(self, attr) for attr in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm"))
        if not have_ow:
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
            )

        x, y = key
        chi_Ap = np.array(self.WF_Ap[:, x, y], dtype=np.complex128, copy=True)
        chi_Bp = np.array(self.WF_Bp[:, x, y], dtype=np.complex128, copy=True)
        chi_Am = np.array(self.WF_Am[:, x, y], dtype=np.complex128, copy=True)
        chi_Bm = np.array(self.WF_Bm[:, x, y], dtype=np.complex128, copy=True)

        def _normalize(v):
            nrm = np.linalg.norm(v)
            if np.isfinite(nrm) and nrm > 0:
                return v / nrm
            return v

        chi_Ap = _normalize(chi_Ap)
        chi_Bp = _normalize(chi_Bp)
        chi_Am = _normalize(chi_Am)
        chi_Bm = _normalize(chi_Bm)

        site_data = {
            "chi_Ap": chi_Ap, "chi_Bp": chi_Bp, "chi_Am": chi_Am, "chi_Bm": chi_Bm,
        }
        cache[key] = site_data
        return site_data

    def _occ_prob_from_vector(self, G, chi):
        """
        Occupation probability Tr[((I+G)/2) |chi><chi|] for normalized chi.
        """
        with self._update_timer("covariance_occupation_matvec", detailed=True):
            v = np.asarray(chi, dtype=np.complex128).reshape(-1)
            gv = G @ v
            occ = 0.5 * (1.0 + float(np.real(np.vdot(v, gv))))
        return float(np.clip(occ, 0.0, 1.0))

    def _projector_from_vector(self, chi):
        v = np.asarray(chi, dtype=np.complex128).reshape(-1)
        return np.outer(v, v.conj())

    def _rank_one_vector_from_projector(self, P):
        P = np.asarray(P, dtype=np.complex128)
        evals, evecs = np.linalg.eigh(0.5 * (P + P.conj().T))
        pos = int(np.argmax(evals))
        val = float(max(evals[pos], 0.0))
        return np.sqrt(val) * evecs[:, pos]

    def _physical_rank1_resolvent_action(self, G_block, chi, rhs, sign, eps=1e-9):
        """Apply (I + sign * G |chi><chi|)^(-1) to rhs."""
        with self._update_timer("rank1_resolvent_action", detailed=True):
            G_block = np.asarray(G_block, dtype=np.complex128)
            chi = np.asarray(chi, dtype=np.complex128).reshape(-1)
            rhs = np.asarray(rhs, dtype=np.complex128)
            sign = float(sign)
            v = G_block @ chi
            chi_rhs = chi.conj() @ rhs
            denom = 1.0 + sign * (chi.conj() @ v)
            self._update_observe_min("minimum_abs_rank1_denominator", abs(denom))
            if np.isfinite(denom) and abs(denom) >= 1e-14:
                return rhs - (sign / denom) * v[:, None] * chi_rhs[None, :]

            self._update_count("regularized_dense_fallback_count")
            with self._update_timer("regularized_dense_fallback", detailed=True):
                p = np.outer(chi, chi.conj())
                solve_mat = np.eye(G_block.shape[0], dtype=np.complex128) + sign * (G_block @ p)
                eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
                if not np.isfinite(eps_scale) or eps_scale < 1.0:
                    eps_scale = 1.0
                return self._solve_regularized(
                    solve_mat, rhs, eps=float(eps) * eps_scale
                )

    def _physical_rank1_measure_blocks(self, G_ss, G_sr, chi_local, particle):
        p = self._projector_from_vector(chi_local)
        q = np.eye(G_ss.shape[0], dtype=np.complex128) - p
        sign = 1.0 if bool(particle) else -1.0
        with self._update_timer("measurement_outer_update", detailed=True):
            if G_sr.shape[1] == 0:
                Y = np.empty((G_ss.shape[0], 0), dtype=np.complex128)
                pY = Y
                G_sr_new = Y
            else:
                Y = self._physical_rank1_resolvent_action(G_ss, chi_local, G_sr, sign)
                pY = p @ Y
                G_sr_new = q @ Y
            Z = self._physical_rank1_resolvent_action(G_ss, chi_local, G_ss @ q, sign)
            G_ss_new = sign * p + q @ Z
        with self._update_timer("hermitian_symmetrization", detailed=True):
            G_ss_new = 0.5 * (G_ss_new + G_ss_new.conj().T)
        if G_sr.shape[1] == 0:
            return G_ss_new, G_sr_new, np.empty((0, 0), dtype=np.complex128)
        with self._update_timer("measurement_outer_update", detailed=True):
            delta_rr = -sign * (G_sr.conj().T @ pY)
        with self._update_timer("hermitian_symmetrization", detailed=True):
            delta_rr = 0.5 * (delta_rr + delta_rr.conj().T)
        return G_ss_new, G_sr_new, delta_rr

    @staticmethod
    def _normalize_physical_covariance_update(mode):
        mode = str(mode).strip().lower()
        aliases = {
            "rank1": "rank1",
            "rank_1": "rank1",
            "rank-one": "rank1",
            "rank_one": "rank1",
            "rank1_resolvent": "rank1",
            "rank1_resolvent_v1": "rank1",
            "dense": "dense",
            "dense_solver": "dense",
            "dense_regularized": "dense",
            "dense_regularized_v1": "dense",
        }
        if mode not in aliases:
            raise ValueError("physical_covariance_update must be either 'rank1' or 'dense'.")
        return aliases[mode]

    @staticmethod
    def _physical_covariance_update_label(mode):
        mode = classA_U1FGTN._normalize_physical_covariance_update(mode)
        if mode == "rank1":
            return "rank1_resolvent_v1"
        return "dense_regularized_v1"

    @staticmethod
    def _normalize_state_representation(representation):
        value = str(representation).strip().lower()
        aliases = {
            "auto": "auto",
            "covariance": "covariance",
            "physical_frame": "physical_frame",
            "pure_frame": "physical_frame",
            "purification_frame": "purification_frame",
            "doubled_frame": "purification_frame",
        }
        if value not in aliases:
            raise ValueError(
                "state_representation must be 'auto', 'covariance', "
                "'physical_frame', or 'purification_frame'."
            )
        return aliases[value]

    @staticmethod
    def _centered_purity_defect(centered_covariance):
        centered = np.asarray(centered_covariance, dtype=np.complex128)
        if centered.ndim < 2 or centered.shape[-1] != centered.shape[-2]:
            raise ValueError("G_init must contain square centered covariance matrices.")
        flat = centered.reshape((-1,) + centered.shape[-2:])
        defects = []
        for matrix in flat:
            hermitian = 0.5 * (matrix + matrix.conj().T)
            dimension = hermitian.shape[0]
            correlation = 0.5 * (
                hermitian + np.eye(dimension, dtype=np.complex128)
            )
            occupations = np.linalg.eigvalsh(correlation)
            if not np.all(np.isfinite(occupations)):
                raise ValueError("G_init has non-finite occupation eigenvalues.")
            range_defect = max(
                0.0,
                float(-np.min(occupations)),
                float(np.max(occupations) - 1.0),
            )
            idempotency_defect = float(
                np.max(np.minimum(np.abs(occupations), np.abs(1.0 - occupations)))
            )
            defects.append(max(range_defect, idempotency_defect))
        return np.asarray(defects, dtype=np.float64)

    @staticmethod
    def _normalize_timing_level(level):
        value = str(level).strip().lower()
        if value not in UpdateTimingCollector.LEVELS:
            raise ValueError(
                f"timing_level must be one of {UpdateTimingCollector.LEVELS}."
            )
        return value

    def _update_timer(self, name, *, detailed=False):
        collector = getattr(self, "_update_timing_collector", None)
        if collector is None:
            return nullcontext()
        return collector.measure(name, detailed=detailed)

    def _update_count(self, name, amount=1):
        collector = getattr(self, "_update_timing_collector", None)
        if collector is not None:
            collector.increment(name, amount)

    def _update_observe_min(self, name, value):
        collector = getattr(self, "_update_timing_collector", None)
        if collector is not None:
            collector.observe_min(name, value)

    def _physical_dense_measure_blocks(self, G_ss, G_sr, chi_local, particle):
        p = self._projector_from_vector(chi_local)
        q = np.eye(G_ss.shape[0], dtype=np.complex128) - p
        sign = 1.0 if bool(particle) else -1.0
        solve_mat = np.eye(G_ss.shape[0], dtype=np.complex128) + sign * (G_ss @ p)
        eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
        if not np.isfinite(eps_scale) or eps_scale < 1.0:
            eps_scale = 1.0
        if G_sr.shape[1] == 0:
            y = np.empty((G_ss.shape[0], 0), dtype=np.complex128)
            pY = y
            G_sr_new = y
        else:
            y = self._solve_regularized(solve_mat, G_sr, eps=1e-9 * eps_scale)
            pY = p @ y
            G_sr_new = q @ y
        z = self._solve_regularized(solve_mat, G_ss @ q, eps=1e-9 * eps_scale)
        G_ss_new = sign * p + q @ z
        G_ss_new = 0.5 * (G_ss_new + G_ss_new.conj().T)
        if G_sr.shape[1] == 0:
            return G_ss_new, G_sr_new, np.empty((0, 0), dtype=np.complex128)
        delta_rr = -sign * (G_sr.conj().T @ pY)
        delta_rr = 0.5 * (delta_rr + delta_rr.conj().T)
        return G_ss_new, G_sr_new, delta_rr

    def _physical_measure_blocks(self, G_ss, G_sr, chi_local, particle):
        mode = getattr(self, "_physical_covariance_update_mode", "rank1")
        mode = self._normalize_physical_covariance_update(mode)
        if mode == "dense":
            return self._physical_dense_measure_blocks(G_ss, G_sr, chi_local, particle)
        return self._physical_rank1_measure_blocks(G_ss, G_sr, chi_local, particle)

    def _complement_indices(self, support_idx, Nlayer):
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        mask = np.ones(int(Nlayer), dtype=bool)
        mask[support_idx] = False
        return np.flatnonzero(mask)

    @staticmethod
    def _normalize_choi_observer_cycles(observer_cycles, cycles):
        if observer_cycles is None:
            return None
        values = [int(cyc) for cyc in observer_cycles]
        if not values:
            raise ValueError("choi_observer_cycles must contain at least one cycle.")
        if values != sorted(set(values)):
            raise ValueError("choi_observer_cycles must be sorted and unique.")
        if values[0] <= 0 or values[-1] > int(cycles):
            raise ValueError("choi_observer_cycles entries must be in the range 1..cycles.")
        return values

    def _init_choi_state(self, batch_count, singular_tol, basis_idx=None, failure_mode="raise"):
        batch_count = int(batch_count)
        Nlayer = self.Ntot // 2
        basis_idx = (
            np.arange(Nlayer, dtype=np.int64)
            if basis_idx is None
            else np.asarray(basis_idx, dtype=np.int64).reshape(-1)
        )
        dim = int(basis_idx.size)
        zero = np.zeros((batch_count, dim, dim), dtype=np.complex128)
        identity = np.broadcast_to(self._eye_of_size(dim), (batch_count, dim, dim)).copy()
        full_to_basis = np.full((Nlayer,), -1, dtype=np.int64)
        full_to_basis[basis_idx] = np.arange(dim, dtype=np.int64)
        return {
            "LL": zero.copy(),
            "LR": identity,
            "RR": zero,
            "basis_idx": basis_idx,
            "full_to_basis": full_to_basis,
            "singular_tol": float(singular_tol),
            "failure_mode": str(failure_mode),
            "active": np.ones((batch_count,), dtype=bool),
            "failure_records": [],
            "min_abs_d": float("inf"),
            "min_abs_d_context": None,
        }

    @staticmethod
    def _choi_outer(left, right):
        return left[:, :, None] * right.conj()[:, None, :]

    def _choi_resolvent_action(self, rhs, v, chi, d, eta1, regularized, eps=1e-9):
        chi_rhs = np.einsum("bi,bij->bj", chi.conj(), rhs)
        alpha_real = np.ones((rhs.shape[0],), dtype=np.float64)
        if np.any(regularized):
            eta1_complex = eta1.astype(np.complex128, copy=False)
            diag_abs = np.abs(1.0 - eta1_complex[:, None] * v * chi.conj())
            row_abs_sum = diag_abs + np.abs(v) * (np.sum(np.abs(chi), axis=-1)[:, None] - np.abs(chi))
            eps_scale = np.maximum(np.amax(row_abs_sum, axis=-1), 1.0)
            alpha_real[regularized] = 1.0 + float(eps) * eps_scale[regularized]

        alpha = alpha_real.astype(np.complex128)
        eta1_complex = eta1.astype(np.complex128, copy=False)
        r = d + eta1_complex
        denom = alpha - eta1_complex * r
        coeff = eta1_complex / (alpha * denom)
        return rhs / alpha[:, None, None] + coeff[:, None, None] * v[:, :, None] * chi_rhs[:, None, :]

    def _choi_apply_rank_one(
        self,
        choi_state,
        sample_offsets,
        chi,
        eta1,
        eta2,
        *,
        support_idx=None,
        context=None,
    ):
        if choi_state is None:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        if sample_offsets.size == 0:
            return
        active = choi_state["active"][sample_offsets]
        if not np.any(active):
            return
        sample_offsets = sample_offsets[active]
        ll = choi_state["LL"][sample_offsets]
        lr = choi_state["LR"][sample_offsets]
        rr = choi_state["RR"][sample_offsets]
        count = int(sample_offsets.size)

        chi = np.asarray(chi, dtype=np.complex128)
        basis_idx = choi_state.get("basis_idx")
        Nlayer = self.Ntot // 2
        if basis_idx is not None and int(basis_idx.size) != Nlayer:
            if support_idx is None:
                chi = chi[..., basis_idx]
            else:
                support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
                mapped = choi_state["full_to_basis"][support_idx]
                if np.any(mapped < 0):
                    raise ValueError("Choi update received mode support outside the active slab basis.")
                support_idx = mapped
        if chi.ndim == 1:
            chi = np.broadcast_to(chi.reshape(1, -1), (count, chi.size)).copy()
        if support_idx is None:
            chi_basis = chi
        else:
            support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
            chi_basis = np.zeros((count, rr.shape[-1]), dtype=np.complex128)
            chi_basis[:, support_idx] = chi

        u = np.einsum("bij,bj->bi", lr, chi_basis)
        v = np.einsum("bij,bj->bi", rr, chi_basis)
        r = np.sum(chi_basis.conj() * v, axis=-1)

        eta1 = np.asarray(eta1, dtype=np.float64).reshape(-1)
        eta2 = np.asarray(eta2, dtype=np.float64).reshape(-1)
        if eta1.size == 1:
            eta1 = np.full((count,), float(eta1[0]), dtype=np.float64)
        if eta2.size == 1:
            eta2 = np.full((count,), float(eta2[0]), dtype=np.float64)
        d = r - eta1.astype(np.complex128)
        abs_d = np.abs(d).astype(np.float64)
        local_pos = int(np.argmin(abs_d))
        local_min = float(abs_d[local_pos])
        if local_min < choi_state["min_abs_d"]:
            diagnostic = {} if context is None else dict(context)
            diagnostic.update(
                {
                    "sample_offset": int(sample_offsets[local_pos]),
                    "eta1": float(eta1[local_pos]),
                    "eta2": float(eta2[local_pos]),
                    "d_real": float(np.real(d[local_pos])),
                    "d_imag": float(np.imag(d[local_pos])),
                }
            )
            if diagnostic.get("batch_start") is not None:
                diagnostic["sample_index"] = int(diagnostic["batch_start"]) + diagnostic["sample_offset"]
            choi_state["min_abs_d"] = local_min
            choi_state["min_abs_d_context"] = diagnostic

        invalid = abs_d < choi_state["singular_tol"]
        if np.any(invalid):
            if choi_state["failure_mode"] == "raise":
                raise FloatingPointError(
                    "Cumulative Choi contraction encountered abs(d) below choi_singular_tol: "
                    f"abs(d)={local_min:.6e}, tol={choi_state['singular_tol']:.6e}, "
                    f"context={choi_state['min_abs_d_context']}"
                )
            for pos in np.flatnonzero(invalid):
                record = {} if context is None else dict(context)
                record.update(
                    {
                        "stage": "denominator_regularized",
                        "sample_offset": int(sample_offsets[pos]),
                        "eta1": float(eta1[pos]),
                        "eta2": float(eta2[pos]),
                        "abs_d": float(abs_d[pos]),
                        "d_real": float(np.real(d[pos])),
                        "d_imag": float(np.imag(d[pos])),
                    }
                )
                if record.get("batch_start") is not None:
                    record["sample_index"] = int(record["batch_start"]) + int(sample_offsets[pos])
                choi_state["failure_records"].append(record)

        sigma_rl = np.swapaxes(lr.conj(), -2, -1)
        rq = rr - v[:, :, None] * chi_basis.conj()[:, None, :]
        x = self._choi_resolvent_action(sigma_rl, v, chi_basis, d, eta1, invalid, eps=1e-9)
        y = self._choi_resolvent_action(rq, v, chi_basis, d, eta1, invalid, eps=1e-9)
        chi_x = np.einsum("bi,bij->bj", chi_basis.conj(), x)
        chi_y = np.einsum("bi,bij->bj", chi_basis.conj(), y)
        eta1_complex = eta1.astype(np.complex128)
        eta2_complex = eta2.astype(np.complex128)
        lrq = lr - u[:, :, None] * chi_basis.conj()[:, None, :]
        ll_new = ll + eta1_complex[:, None, None] * u[:, :, None] * chi_x[:, None, :]
        lr_new = lrq + eta1_complex[:, None, None] * u[:, :, None] * chi_y[:, None, :]
        rr_new = (
            eta2_complex[:, None, None] * self._choi_outer(chi_basis, chi_basis)
            + y
            - chi_basis[:, :, None] * chi_y[:, None, :]
        )

        choi_state["LL"][sample_offsets] = ll_new
        choi_state["LR"][sample_offsets] = lr_new
        choi_state["RR"][sample_offsets] = rr_new

    def _init_lyapunov_state(
        self,
        batch_count=1,
        n_vec=None,
        basis_idx=None,
        initial_frame=None,
        sample_start=0,
        track_restricted_core=False,
        track_record_fisher=False,
        singular_tol=1e-12,
        failure_mode="raise",
    ):
        basis_idx = (
            np.arange(self.Ntot // 2, dtype=np.int64)
            if basis_idx is None
            else np.asarray(basis_idx, dtype=np.int64).reshape(-1)
        )
        basis_dim = int(basis_idx.size)
        batch_count = int(batch_count)
        sample_start = int(sample_start)
        Nlayer = self.Ntot // 2

        if initial_frame is None:
            n_vec = basis_dim if n_vec is None else int(n_vec)
            frame = np.zeros((batch_count, Nlayer, n_vec), dtype=np.complex128)
            frame[:, basis_idx[:n_vec], np.arange(n_vec, dtype=np.int64)] = 1.0
        else:
            raw = np.asarray(initial_frame, dtype=np.complex128)
            if raw.ndim == 2:
                raw = np.broadcast_to(raw[None, ...], (batch_count,) + raw.shape).copy()
            elif raw.ndim == 3:
                if raw.shape[0] == batch_count:
                    raw = np.array(raw, copy=True)
                elif raw.shape[0] >= sample_start + batch_count:
                    raw = np.array(raw[sample_start : sample_start + batch_count], copy=True)
                else:
                    raise ValueError(
                        "A batched lyapunov_initial_frame must have either batch_count rows "
                        "or enough rows for the requested sample slice."
                    )
            else:
                raise ValueError(
                    "lyapunov_initial_frame must have shape (dimension,nvec) "
                    "or (samples,dimension,nvec)."
                )
            inferred_nvec = int(raw.shape[-1])
            if n_vec is None:
                n_vec = inferred_nvec
            elif int(n_vec) != inferred_nvec:
                raise ValueError(
                    f"lyapunov_nvec={int(n_vec)} disagrees with the initial frame's "
                    f"{inferred_nvec} columns."
                )
            n_vec = inferred_nvec
            if raw.shape[1] == Nlayer:
                frame = raw
            elif raw.shape[1] == basis_dim:
                frame = np.zeros((batch_count, Nlayer, n_vec), dtype=np.complex128)
                frame[:, basis_idx, :] = raw
            else:
                raise ValueError(
                    "lyapunov_initial_frame row dimension must equal the full top-layer "
                    "dimension or the active-basis dimension."
                )
            if not np.all(np.isfinite(frame)):
                raise ValueError("lyapunov_initial_frame contains non-finite entries.")
            gram = np.matmul(frame.conj().transpose(0, 2, 1), frame)
            target = np.broadcast_to(np.eye(n_vec, dtype=np.complex128), gram.shape)
            if not np.allclose(gram, target, atol=1e-9, rtol=1e-9):
                err = float(np.max(np.abs(gram - target)))
                raise ValueError(
                    "lyapunov_initial_frame columns must be orthonormal; "
                    f"maximum Gram error is {err:.3e}."
                )
        if n_vec <= 0 or n_vec > basis_dim:
            raise ValueError(f"lyapunov_nvec must satisfy 1 <= n_vec <= {basis_dim}; got {n_vec}")

        failure_mode = str(failure_mode).strip().lower()
        if failure_mode not in ("raise", "censor"):
            raise ValueError("lyapunov_failure_mode must be either 'raise' or 'censor'.")
        singular_tol = float(singular_tol)
        if not np.isfinite(singular_tol) or singular_tol <= 0.0:
            raise ValueError("lyapunov_singular_tol must be a positive finite scalar.")
        track_restricted_core = bool(track_restricted_core or track_record_fisher)
        track_record_fisher = bool(track_record_fisher)
        if track_record_fisher and n_vec != 2:
            raise ValueError("lyapunov_track_record_fisher requires exactly two tangent-frame columns.")

        state = {
            "frame": frame,
            "log_diag": np.zeros((batch_count, n_vec), dtype=np.float64),
            "n_vec": n_vec,
            "basis_idx": basis_idx,
            "null_counts": np.zeros((batch_count,), dtype=np.int64),
            "active": np.ones((batch_count,), dtype=bool),
            "singular_tol": singular_tol,
            "failure_mode": failure_mode,
            "failure_records": [],
            "min_branch_probability": np.full((batch_count,), np.inf, dtype=np.float64),
            "min_abs_born_denominator": np.full((batch_count,), np.inf, dtype=np.float64),
            "invalid_branch_count": np.zeros((batch_count,), dtype=np.int64),
            "track_restricted_core": track_restricted_core,
            "track_record_fisher": track_record_fisher,
            "last_r": None,
            "last_cycle_null_mask": np.zeros((batch_count, n_vec), dtype=bool),
        }
        if track_restricted_core:
            state["core_hat"] = np.broadcast_to(
                np.eye(n_vec, dtype=np.complex128), (batch_count, n_vec, n_vec)
            ).copy()
            state["core_log_scale"] = np.zeros((batch_count,), dtype=np.float64)
            state["core_null_count"] = np.zeros((batch_count,), dtype=np.int64)
        if track_record_fisher:
            state["record_fisher_hat"] = np.zeros((batch_count, 3, 3), dtype=np.float64)
            state["record_fisher_log_scale"] = np.full((batch_count,), -np.inf, dtype=np.float64)
            state["record_fisher_endpoint_count"] = np.zeros((batch_count,), dtype=np.int64)
            state["record_fisher_infinite"] = np.zeros((batch_count,), dtype=bool)
        return state

    def _init_pure_occupied_empty_lyapunov_state(
        self,
        physical_state,
        *,
        basis_idx=None,
        singular_tol=1e-12,
        failure_mode="raise",
        purity_tolerance=1e-9,
    ):
        """Initialize exact occupied/empty ambient half-Jacobian blocks.

        The pure physical state supplies a complete cycle-zero decomposition.
        Tangent updates act on the concatenated frame, while QR factors and
        cumulative cores are stabilized independently in the occupied and
        empty column blocks.  This is the NumPy/CPU counterpart of the
        canonical GPU ``pure_occupied_empty`` path.
        """
        if not isinstance(physical_state, OccupiedFrameState):
            raise TypeError(
                "lyapunov_basis_mode='pure_occupied_empty' requires "
                "state_representation='physical_frame'."
            )
        if physical_state.representation != "physical_frame":
            raise ValueError(
                "lyapunov_basis_mode='pure_occupied_empty' requires a pure "
                "physical-frame trajectory."
            )
        basis_idx = (
            np.arange(self.Ntot // 2, dtype=np.int64)
            if basis_idx is None
            else np.asarray(basis_idx, dtype=np.int64).reshape(-1)
        )
        active_dim = int(basis_idx.size)
        occupied_restricted = physical_state.frame[basis_idx, :]
        correlation = occupied_restricted @ occupied_restricted.conj().T
        correlation = 0.5 * (correlation + correlation.conj().T)
        occupations, eigenvectors = np.linalg.eigh(correlation)
        purity_defect = float(
            np.max(np.minimum(np.abs(occupations), np.abs(1.0 - occupations)))
        )
        if not np.isfinite(purity_defect) or purity_defect > float(purity_tolerance):
            raise ValueError(
                "The active cycle-zero state is not pure enough to define exact "
                "occupied/empty tangent blocks; maximum occupation defect is "
                f"{purity_defect:.3e}."
            )
        occupied_mask = occupations > 0.5
        occupied_basis = eigenvectors[:, occupied_mask]
        empty_basis = eigenvectors[:, ~occupied_mask]
        occupied_dim = int(occupied_basis.shape[1])
        empty_dim = int(empty_basis.shape[1])
        if occupied_dim + empty_dim != active_dim:
            raise RuntimeError("Occupied/empty basis did not span the active space.")
        initial_frame = np.concatenate((occupied_basis, empty_basis), axis=1)
        state = self._init_lyapunov_state(
            batch_count=1,
            n_vec=active_dim,
            basis_idx=basis_idx,
            initial_frame=initial_frame,
            sample_start=0,
            track_restricted_core=False,
            track_record_fisher=False,
            singular_tol=singular_tol,
            failure_mode=failure_mode,
        )
        state.update(
            {
                "basis_mode": "pure_occupied_empty",
                "block_sizes": (occupied_dim, empty_dim),
                "block_labels": ("occupied", "empty"),
                "block_core_hat": (
                    np.eye(occupied_dim, dtype=np.complex128)[None, ...],
                    np.eye(empty_dim, dtype=np.complex128)[None, ...],
                ),
                "block_core_log_scale": (
                    np.zeros((1,), dtype=np.float64),
                    np.zeros((1,), dtype=np.float64),
                ),
                "block_core_null_count": (
                    np.zeros((1,), dtype=np.int64),
                    np.zeros((1,), dtype=np.int64),
                ),
                "last_r_blocks": (None, None),
                "initial_block_basis": (
                    np.array(state["frame"][:, :, :occupied_dim], copy=True),
                    np.array(state["frame"][:, :, occupied_dim:], copy=True),
                ),
                "initial_active_occupations": occupations,
                "initial_active_purity_defect": purity_defect,
                "track_restricted_core": True,
            }
        )
        return state

    def _lyapunov_register_branch(self, lyapunov_state, G_sel, sample_offsets, P, particle):
        """Record fixed-branch Born denominators and return the valid-row mask."""
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        G_batch = np.asarray(G_sel, dtype=np.complex128)
        if G_batch.ndim == 2:
            G_batch = G_batch[None, ...]
        P_batch = np.asarray(P, dtype=np.complex128)
        if P_batch.ndim == 2:
            P_batch = np.broadcast_to(P_batch[None, ...], G_batch.shape)
        sign = 1.0 if bool(particle) else -1.0
        g = np.real(np.einsum("bij,bji->b", G_batch, P_batch))
        denominator = np.abs(1.0 + sign * g)
        probability = 0.5 * denominator
        lyapunov_state["min_abs_born_denominator"][sample_offsets] = np.minimum(
            lyapunov_state["min_abs_born_denominator"][sample_offsets], denominator
        )
        lyapunov_state["min_branch_probability"][sample_offsets] = np.minimum(
            lyapunov_state["min_branch_probability"][sample_offsets], probability
        )
        valid = np.isfinite(denominator) & (
            denominator > float(lyapunov_state["singular_tol"])
        )
        if np.all(valid):
            return valid, g

        bad_local = np.flatnonzero(~valid)
        bad_offsets = sample_offsets[bad_local]
        lyapunov_state["invalid_branch_count"][bad_offsets] += 1
        for local_index, offset in zip(bad_local.tolist(), bad_offsets.tolist()):
            lyapunov_state["failure_records"].append(
                {
                    "sample_offset": int(offset),
                    "particle": bool(particle),
                    "born_denominator": float(denominator[local_index]),
                    "branch_probability": float(probability[local_index]),
                }
            )
        if lyapunov_state["failure_mode"] == "raise":
            raise FloatingPointError(
                "The requested fixed measurement branch has zero or non-finite Born "
                f"probability; sample offsets={bad_offsets.tolist()}, "
                f"denominators={denominator[bad_local].tolist()}."
            )
        lyapunov_state["active"][bad_offsets] = False
        lyapunov_state["frame"][bad_offsets] = 0.0
        return valid, g

    def _lyapunov_accumulate_record_fisher(
        self,
        lyapunov_state,
        G_sel,
        sample_offsets,
        P,
        *,
        support_idx=None,
        valid_mask=None,
    ):
        """Accumulate predictable record Fisher information for a two-mode tangent plane."""
        if lyapunov_state is None or not lyapunov_state.get("track_record_fisher", False):
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        G_batch = np.asarray(G_sel, dtype=np.complex128)
        if G_batch.ndim == 2:
            G_batch = G_batch[None, ...]
        P_batch = np.asarray(P, dtype=np.complex128)
        if P_batch.ndim == 2:
            P_batch = np.broadcast_to(P_batch[None, ...], G_batch.shape)
        if valid_mask is None:
            valid_mask = np.ones((sample_offsets.size,), dtype=bool)
        support_idx_arr = (
            None
            if support_idx is None
            else np.asarray(support_idx, dtype=np.int64).reshape(-1)
        )
        tol = float(lyapunov_state["singular_tol"])

        for row, offset in enumerate(sample_offsets):
            if not valid_mask[row] or not lyapunov_state["active"][offset]:
                continue
            gamma = float(lyapunov_state["core_log_scale"][offset])
            if not np.isfinite(gamma):
                continue
            frame = lyapunov_state["frame"][offset]
            if support_idx_arr is not None:
                frame = frame[support_idx_arr]
            image_hat = frame @ lyapunov_state["core_hat"][offset]
            overlap = image_hat.conj().T @ P_batch[row] @ image_hat
            d = np.asarray(
                [
                    2.0 * np.real(overlap[0, 1]),
                    -2.0 * np.imag(overlap[0, 1]),
                    np.real(overlap[0, 0] - overlap[1, 1]),
                ],
                dtype=np.float64,
            )
            g = float(np.real(np.trace(G_batch[row] @ P_batch[row])))
            conditional_variance = max(0.0, 1.0 - g * g)
            if conditional_variance <= tol:
                lyapunov_state["record_fisher_endpoint_count"][offset] += 1
                if np.linalg.norm(d) > np.sqrt(tol):
                    lyapunov_state["record_fisher_infinite"][offset] = True
                continue
            if not np.any(d):
                continue

            increment_hat = np.outer(d, d)
            increment_log_scale = 4.0 * gamma - np.log(conditional_variance)
            old_log_scale = float(
                lyapunov_state["record_fisher_log_scale"][offset]
            )
            new_log_scale = max(old_log_scale, increment_log_scale)
            old_weight = (
                0.0
                if not np.isfinite(old_log_scale)
                else np.exp(old_log_scale - new_log_scale)
            )
            new_weight = np.exp(increment_log_scale - new_log_scale)
            lyapunov_state["record_fisher_hat"][offset] = (
                old_weight * lyapunov_state["record_fisher_hat"][offset]
                + new_weight * increment_hat
            )
            lyapunov_state["record_fisher_log_scale"][offset] = new_log_scale

    def _lyapunov_apply_local_selected(
        self, lyapunov_state, G_sel, sample_offsets, support_idx, comp_idx, chi_local, particle
    ):
        if lyapunov_state is None or np.asarray(G_sel).shape[0] == 0:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
        G_ss, G_sr = self._local_support_block(
            np.asarray(G_sel, dtype=np.complex128), support_idx, comp_idx
        )
        p = self._projector_from_vector(chi_local)
        q = np.eye(support_idx.size, dtype=np.complex128) - p
        valid, _ = self._lyapunov_register_branch(
            lyapunov_state, G_ss, sample_offsets, p, particle
        )
        self._lyapunov_accumulate_record_fisher(
            lyapunov_state,
            G_ss,
            sample_offsets,
            p,
            support_idx=support_idx,
            valid_mask=valid,
        )
        if not valid[0]:
            return

        sign = 1.0 if bool(particle) else -1.0
        V = lyapunov_state["frame"][sample_offsets[0]]
        V_s = V[support_idx, :]
        V_r = (
            V[comp_idx, :]
            if comp_idx.size
            else np.empty((0, V.shape[-1]), dtype=V.dtype)
        )
        solve_mat = (
            np.eye(support_idx.size, dtype=np.complex128) + sign * (G_ss @ p)
        )
        X_s = np.linalg.solve(solve_mat, V_s)
        pX_s = p @ X_s
        X_r = (
            V_r - sign * (G_sr.conj().T @ pX_s)
            if comp_idx.size
            else V_r
        )
        V_new = V.copy()
        V_new[support_idx, :] = q @ X_s
        if comp_idx.size:
            V_new[comp_idx, :] = X_r
        lyapunov_state["frame"][sample_offsets[0]] = V_new

    def _lyapunov_apply_dense_selected(self, lyapunov_state, G_sel, sample_offsets, P, particle):
        if lyapunov_state is None:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        P = np.asarray(P, dtype=np.complex128)
        Q = self._get_top_eye() - P
        sign = 1.0 if bool(particle) else -1.0
        valid, _ = self._lyapunov_register_branch(
            lyapunov_state, G_sel, sample_offsets, P, particle
        )
        self._lyapunov_accumulate_record_fisher(
            lyapunov_state, G_sel, sample_offsets, P, valid_mask=valid
        )
        if not valid[0]:
            return

        V = lyapunov_state["frame"][sample_offsets[0]]
        solve_mat = self._get_top_eye() + sign * (
            np.asarray(G_sel, dtype=np.complex128) @ P
        )
        X = np.linalg.solve(solve_mat, V)
        lyapunov_state["frame"][sample_offsets[0]] = Q @ X

    def _lyapunov_apply_occupied_frame_channel(
        self,
        lyapunov_state,
        state,
        sample_offsets,
        support_idx,
        orbital_local,
        *,
        particle,
    ):
        """Propagate the legacy tangent cocycle using frame contractions only."""

        if lyapunov_state is None:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        if sample_offsets.size != 1:
            raise ValueError("CPU occupied-frame tangent updates are serial.")
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        chi = np.zeros(state.physical_dimension, dtype=np.complex128)
        chi[support_idx] = np.asarray(orbital_local, dtype=np.complex128)
        occupied_coefficients = state.physical_frame.conj().T @ chi
        g_chi = 2.0 * (state.physical_frame @ occupied_coefficients) - chi
        g = float(np.real(np.vdot(chi, g_chi)))
        sign = 1.0 if bool(particle) else -1.0
        denominator = abs(1.0 + sign * g)
        probability = 0.5 * denominator
        offset = int(sample_offsets[0])
        lyapunov_state["min_abs_born_denominator"][offset] = min(
            lyapunov_state["min_abs_born_denominator"][offset], denominator
        )
        lyapunov_state["min_branch_probability"][offset] = min(
            lyapunov_state["min_branch_probability"][offset], probability
        )
        if not np.isfinite(denominator) or denominator <= float(
            lyapunov_state["singular_tol"]
        ):
            lyapunov_state["invalid_branch_count"][offset] += 1
            lyapunov_state["active"][offset] = False
            lyapunov_state["frame"][offset] = 0.0
            if lyapunov_state["failure_mode"] == "raise":
                raise FloatingPointError(
                    "Occupied-frame tangent branch has zero or non-finite Born probability."
                )
            return
        tangent = lyapunov_state["frame"][offset]
        solved = tangent - (
            sign / (1.0 + sign * g)
        ) * np.outer(g_chi, chi.conj() @ tangent)
        lyapunov_state["frame"][offset] = solved - np.outer(
            chi, chi.conj() @ solved
        )

        if lyapunov_state.get("track_record_fisher", False):
            projector = np.outer(chi, chi.conj())
            self._lyapunov_accumulate_record_fisher(
                lyapunov_state,
                np.asarray([g * projector]),
                sample_offsets,
                projector,
            )

    def _lyapunov_reset_occupied_frame_channel(
        self, lyapunov_state, sample_offsets, support_idx, orbital_local
    ):
        if lyapunov_state is None:
            return
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        chi = np.zeros(self.Ntot // 2, dtype=np.complex128)
        chi[support_idx] = np.asarray(orbital_local, dtype=np.complex128)
        for offset in np.asarray(sample_offsets, dtype=np.int64).reshape(-1):
            tangent = lyapunov_state["frame"][offset]
            lyapunov_state["frame"][offset] = tangent - np.outer(
                chi, chi.conj() @ tangent
            )

    def _lyapunov_apply_local_channel(
        self, lyapunov_state, G, sample_offsets, support_idx, comp_idx, chi_local, particle
    ):
        self._lyapunov_apply_local_selected(
            lyapunov_state, G, sample_offsets, support_idx, comp_idx, chi_local, particle=particle
        )

    def _lyapunov_apply_dense_channel(self, lyapunov_state, G, sample_offsets, P, particle):
        self._lyapunov_apply_dense_selected(lyapunov_state, G, sample_offsets, P, particle=particle)

    def _lyapunov_apply_local_reset(self, lyapunov_state, sample_offsets, support_idx, comp_idx, chi_local):
        if lyapunov_state is None:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        V = lyapunov_state["frame"][sample_offsets[0]].copy()
        q = np.eye(support_idx.size, dtype=np.complex128) - self._projector_from_vector(chi_local)
        V[support_idx, :] = q @ V[support_idx, :]
        lyapunov_state["frame"][sample_offsets[0]] = V

    def _lyapunov_apply_dense_reset(self, lyapunov_state, sample_offsets, P):
        if lyapunov_state is None:
            return
        sample_offsets = np.asarray(sample_offsets, dtype=np.int64).reshape(-1)
        Q = self._get_top_eye() - np.asarray(P, dtype=np.complex128)
        lyapunov_state["frame"][sample_offsets[0]] = Q @ lyapunov_state["frame"][sample_offsets[0]]

    def _lyapunov_end_cycle(self, lyapunov_state, cycle):
        if lyapunov_state is None:
            return None
        if lyapunov_state.get("basis_mode") == "pure_occupied_empty":
            return self._lyapunov_end_cycle_pure_blocks(lyapunov_state, cycle)
        frame = lyapunov_state["frame"]
        spectra = []
        r_payload = np.zeros(
            (frame.shape[0], frame.shape[-1], frame.shape[-1]),
            dtype=np.complex128,
        )
        cycle_null_mask = np.zeros(
            (frame.shape[0], frame.shape[-1]), dtype=bool
        )
        floor = np.finfo(np.float64).tiny
        for offset in range(frame.shape[0]):
            if not lyapunov_state["active"][offset]:
                lyapunov_state["log_diag"][offset] = -np.inf
                lyapunov_state["frame"][offset] = 0.0
                cycle_null_mask[offset] = True
                spectra.append(
                    np.full((frame.shape[-1],), -np.inf, dtype=np.float64)
                )
                continue

            q, r = np.linalg.qr(frame[offset], mode="reduced")
            diag = np.diag(r)
            abs_diag = np.abs(diag).astype(np.float64)
            rank_tol = (
                np.finfo(np.float64).eps
                * max(frame.shape[1], frame.shape[2])
                * max(float(np.max(abs_diag)), 1.0)
            )
            null_mask = abs_diag <= rank_tol
            cycle_null_mask[offset] = null_mask
            lyapunov_state["null_counts"][offset] += int(
                np.count_nonzero(null_mask)
            )
            lyapunov_state["log_diag"][offset] += np.log(
                np.clip(abs_diag, floor, None)
            )
            phase = np.ones_like(diag)
            mask = np.abs(diag) > 1e-30
            phase[mask] = diag[mask] / np.abs(diag[mask])
            q_gauge = q * phase[None, :]
            r_gauge = phase.conj()[:, None] * r
            lyapunov_state["frame"][offset] = q_gauge
            r_payload[offset] = r_gauge

            if lyapunov_state.get("track_restricted_core", False):
                core_raw = r_gauge @ lyapunov_state["core_hat"][offset]
                core_norm = float(np.linalg.norm(core_raw, ord="fro"))
                if not np.isfinite(core_norm) or core_norm <= floor:
                    lyapunov_state["core_hat"][offset] = 0.0
                    lyapunov_state["core_log_scale"][offset] = -np.inf
                    lyapunov_state["core_null_count"][offset] += 1
                else:
                    lyapunov_state["core_hat"][offset] = core_raw / core_norm
                    lyapunov_state["core_log_scale"][offset] += np.log(core_norm)
            spectra.append(
                np.sort(lyapunov_state["log_diag"][offset] / float(cycle))
            )
        lyapunov_state["last_r"] = r_payload
        lyapunov_state["last_cycle_null_mask"] = cycle_null_mask
        return np.asarray(spectra, dtype=np.float64)

    def _lyapunov_end_cycle_pure_blocks(self, lyapunov_state, cycle):
        """QR-stabilize occupied and empty ambient blocks independently."""
        frame = lyapunov_state["frame"]
        floor = np.finfo(np.float64).tiny
        q_blocks = []
        r_blocks = []
        null_masks = []
        core_hats = list(lyapunov_state["block_core_hat"])
        core_scales = list(lyapunov_state["block_core_log_scale"])
        core_null_counts = list(lyapunov_state["block_core_null_count"])
        start = 0
        for block_index, block_size in enumerate(lyapunov_state["block_sizes"]):
            stop = start + int(block_size)
            raw_block = frame[:, :, start:stop]
            q_block = np.zeros_like(raw_block)
            r_block = np.zeros(
                (raw_block.shape[0], int(block_size), int(block_size)),
                dtype=np.complex128,
            )
            block_null = np.zeros(
                (raw_block.shape[0], int(block_size)), dtype=bool
            )
            for offset in range(raw_block.shape[0]):
                if not lyapunov_state["active"][offset]:
                    lyapunov_state["log_diag"][offset, start:stop] = -np.inf
                    block_null[offset] = True
                    continue
                q, r = np.linalg.qr(raw_block[offset], mode="reduced")
                diag = np.diag(r)
                abs_diag = np.abs(diag).astype(np.float64)
                scale = max(float(np.max(abs_diag)), 1.0)
                rank_tol = (
                    np.finfo(np.float64).eps
                    * max(raw_block.shape[1], raw_block.shape[2])
                    * scale
                )
                null_mask = abs_diag <= rank_tol
                block_null[offset] = null_mask
                lyapunov_state["null_counts"][offset] += int(
                    np.count_nonzero(null_mask)
                )
                lyapunov_state["log_diag"][offset, start:stop] += np.log(
                    np.clip(abs_diag, floor, None)
                )
                phase = np.ones_like(diag)
                nonzero = np.abs(diag) > 1e-30
                phase[nonzero] = diag[nonzero] / np.abs(diag[nonzero])
                q_gauge = q * phase[None, :]
                r_gauge = phase.conj()[:, None] * r
                q_block[offset] = q_gauge
                r_block[offset] = r_gauge

                core_raw = r_gauge @ core_hats[block_index][offset]
                core_norm = float(np.linalg.norm(core_raw, ord="fro"))
                if not np.isfinite(core_norm) or core_norm <= floor:
                    core_hats[block_index][offset] = 0.0
                    core_scales[block_index][offset] = -np.inf
                    core_null_counts[block_index][offset] += 1
                else:
                    core_hats[block_index][offset] = core_raw / core_norm
                    core_scales[block_index][offset] += np.log(core_norm)
            q_blocks.append(q_block)
            r_blocks.append(r_block)
            null_masks.append(block_null)
            start = stop

        lyapunov_state["frame"] = np.concatenate(q_blocks, axis=-1)
        lyapunov_state["last_r_blocks"] = tuple(r_blocks)
        last_r = np.zeros(
            (frame.shape[0], frame.shape[-1], frame.shape[-1]),
            dtype=np.complex128,
        )
        start = 0
        for block_size, r_block in zip(lyapunov_state["block_sizes"], r_blocks):
            stop = start + int(block_size)
            last_r[:, start:stop, start:stop] = r_block
            start = stop
        lyapunov_state["last_r"] = last_r
        lyapunov_state["last_cycle_null_mask"] = np.concatenate(
            null_masks, axis=-1
        )
        lyapunov_state["block_core_hat"] = tuple(core_hats)
        lyapunov_state["block_core_log_scale"] = tuple(core_scales)
        lyapunov_state["block_core_null_count"] = tuple(core_null_counts)
        return np.sort(lyapunov_state["log_diag"] / float(cycle), axis=-1)

    def _lyapunov_frame_payload(self, lyapunov_state):
        if lyapunov_state is None:
            return {}
        payload = {
            "lyapunov_frame": lyapunov_state["frame"],
            "lyapunov_qr_r": lyapunov_state["last_r"],
            "lyapunov_log_diag": lyapunov_state["log_diag"],
            "lyapunov_cycle_null_mask": lyapunov_state["last_cycle_null_mask"],
            "lyapunov_null_counts": lyapunov_state["null_counts"],
            "lyapunov_active_mask": lyapunov_state["active"],
            "lyapunov_min_branch_probability": lyapunov_state[
                "min_branch_probability"
            ],
            "lyapunov_min_abs_born_denominator": lyapunov_state[
                "min_abs_born_denominator"
            ],
            "lyapunov_invalid_branch_count": lyapunov_state[
                "invalid_branch_count"
            ],
            "lyapunov_failure_records": tuple(
                lyapunov_state["failure_records"]
            ),
        }
        if lyapunov_state.get("basis_mode") == "pure_occupied_empty":
            payload.update(
                {
                    "lyapunov_basis_mode": "pure_occupied_empty",
                    "lyapunov_block_sizes": lyapunov_state["block_sizes"],
                    "lyapunov_block_labels": lyapunov_state["block_labels"],
                    "lyapunov_block_core_hat": lyapunov_state[
                        "block_core_hat"
                    ],
                    "lyapunov_block_core_log_scale": lyapunov_state[
                        "block_core_log_scale"
                    ],
                    "lyapunov_block_core_null_count": lyapunov_state[
                        "block_core_null_count"
                    ],
                    "lyapunov_initial_block_basis": lyapunov_state[
                        "initial_block_basis"
                    ],
                    "lyapunov_initial_active_occupations": lyapunov_state[
                        "initial_active_occupations"
                    ],
                    "lyapunov_initial_active_purity_defect": lyapunov_state[
                        "initial_active_purity_defect"
                    ],
                }
            )
            return payload
        if lyapunov_state.get("track_restricted_core", False):
            payload.update(
                {
                    "lyapunov_core_hat": lyapunov_state["core_hat"],
                    "lyapunov_core_log_scale": lyapunov_state[
                        "core_log_scale"
                    ],
                    "lyapunov_core_null_count": lyapunov_state[
                        "core_null_count"
                    ],
                }
            )
        if lyapunov_state.get("track_record_fisher", False):
            payload.update(
                {
                    "lyapunov_record_fisher_hat": lyapunov_state[
                        "record_fisher_hat"
                    ],
                    "lyapunov_record_fisher_log_scale": lyapunov_state[
                        "record_fisher_log_scale"
                    ],
                    "lyapunov_record_fisher_endpoint_count": lyapunov_state[
                        "record_fisher_endpoint_count"
                    ],
                    "lyapunov_record_fisher_infinite": lyapunov_state[
                        "record_fisher_infinite"
                    ],
                }
            )
        return payload

    def _lyapunov_min_abs_vector_payload(self, lyapunov_state, cycle):
        if lyapunov_state is None:
            return {}
        spectrum = lyapunov_state["log_diag"] / float(cycle)
        min_abs_index = np.argmin(np.abs(spectrum), axis=1)
        vectors = []
        for offset, idx in enumerate(min_abs_index):
            vector = lyapunov_state["frame"][offset, :, int(idx)]
            basis_idx = lyapunov_state.get("basis_idx")
            if basis_idx is not None and int(basis_idx.size) != self.Ntot // 2:
                vector = vector[basis_idx]
            vectors.append(vector)
        return {
            "lyapunov_min_abs_vector": np.asarray(vectors, dtype=np.complex128),
            "lyapunov_min_abs_value": spectrum[np.arange(spectrum.shape[0]), min_abs_index],
            "lyapunov_min_abs_index": min_abs_index,
            "lyapunov_null_counts": lyapunov_state["null_counts"].copy(),
        }

    def _get_ow_local_support_data(self, Rx, Ry):
        """
        Cache the exact finite-support payload for OW channels at one site.

        For finite ``nshell``, the OW modes are compactly supported after masking.
        This cache stores the union support of Ap/Bp/Am/Bm and the corresponding
        restricted local channel vectors so the Markov update can act on the small
        support block instead of solving on the full top layer.
        """
        key = (int(Rx) % self.Nx, int(Ry) % self.Ny)
        cache = getattr(self, "_ow_local_support_cache", None)
        if cache is None:
            cache = {}
            self._ow_local_support_cache = cache
        if key in cache:
            return cache[key]

        site_data = self._get_ow_local_data(*key)
        Nlayer = self.Ntot // 2
        tol = 1e-15
        support_mask = (
            (np.abs(site_data["chi_Ap"]) > tol)
            | (np.abs(site_data["chi_Bp"]) > tol)
            | (np.abs(site_data["chi_Am"]) > tol)
            | (np.abs(site_data["chi_Bm"]) > tol)
        )
        support_idx = np.flatnonzero(support_mask)
        if support_idx.size == 0:
            raise ValueError(f"Empty OW local support at site {(Rx, Ry)}.")
        comp_idx = self._complement_indices(support_idx, Nlayer)
        payload = {
            "idx": support_idx,
            "comp": comp_idx,
            "Ap": np.asarray(site_data["chi_Ap"][support_idx], dtype=np.complex128),
            "Bp": np.asarray(site_data["chi_Bp"][support_idx], dtype=np.complex128),
            "Am": np.asarray(site_data["chi_Am"][support_idx], dtype=np.complex128),
            "Bm": np.asarray(site_data["chi_Bm"][support_idx], dtype=np.complex128),
        }
        cache[key] = payload
        return payload

    def _local_support_block(self, G, support_idx, comp_idx):
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
        G_ss = G[np.ix_(support_idx, support_idx)]
        if comp_idx.size == 0:
            G_sr = np.empty((support_idx.size, 0), dtype=np.complex128)
        else:
            G_sr = G[np.ix_(support_idx, comp_idx)]
        return G_ss, G_sr

    def _measure_only_top_layer_local(self, G, support_idx, comp_idx, chi_local, particle=True):
        """
        Exact top-layer measurement update restricted to the local support block.

        This is algebraically equivalent to the dense projector update for a
        finite-support mode, but it only solves on the support-support block.
        """
        G = np.asarray(G, dtype=np.complex128)
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
        G_ss, G_sr = self._local_support_block(G, support_idx, comp_idx)
        G_ss_new, G_sr_new, delta_rr = self._physical_measure_blocks(
            G_ss,
            G_sr,
            np.asarray(chi_local, dtype=np.complex128).reshape(-1),
            particle=particle,
        )
        Gnew = np.array(G, copy=True)
        Gnew[np.ix_(support_idx, support_idx)] = G_ss_new

        if comp_idx.size > 0:
            Gnew[np.ix_(comp_idx, comp_idx)] += delta_rr
            Gnew[np.ix_(support_idx, comp_idx)] = G_sr_new
            Gnew[np.ix_(comp_idx, support_idx)] = G_sr_new.conj().T

        return 0.5 * (Gnew + Gnew.conj().T)

    def _ancilla_swap_top_local(self, Gtop, support_idx, comp_idx, chi_local, n_a, target_occupied=None):
        """
        Exact local reset/pump update on the finite support block.
        """
        Gtop = np.asarray(Gtop, dtype=np.complex128)
        support_idx = np.asarray(support_idx, dtype=np.int64).reshape(-1)
        comp_idx = np.asarray(comp_idx, dtype=np.int64).reshape(-1)
        G_ss, G_sr = self._local_support_block(Gtop, support_idx, comp_idx)
        p = self._projector_from_vector(chi_local)
        Ik = np.eye(support_idx.size, dtype=np.complex128)
        q = Ik - p

        if target_occupied is None:
            ancilla_cov = 1 if np.random.rand() < n_a else -1
        else:
            ancilla_cov = 1 if bool(target_occupied) else -1

        with self._update_timer("reset_outer_update", detailed=True):
            Gnew = np.array(Gtop, copy=True)
            G_ss_new = q @ G_ss @ q + ancilla_cov * p
            Gnew[np.ix_(support_idx, support_idx)] = G_ss_new
            if comp_idx.size > 0:
                G_sr_new = q @ G_sr
                Gnew[np.ix_(support_idx, comp_idx)] = G_sr_new
                Gnew[np.ix_(comp_idx, support_idx)] = G_sr_new.conj().T
        with self._update_timer("hermitian_symmetrization", detailed=True):
            Gnew[np.ix_(support_idx, support_idx)] = 0.5 * (
                G_ss_new + G_ss_new.conj().T
            )
            Gnew = 0.5 * (Gnew + Gnew.conj().T)
        if not np.all(np.isfinite(Gnew)):
            raise FloatingPointError(f"Non-finite local ancilla swap result (n_a={n_a}, occ={ancilla_cov})")
        return Gnew

    # ------------------ Overcomplete Wannier (OW) projectors ------------------

    def create_domain_wall(self, alpha_1, alpha_2, dw_interval=None):
        '''Construct and store the default domain-wall mass profile for the lattice.

        ``alpha_1`` is the topological-region mass and ``alpha_2`` is the trivial-region mass.
        ``dw_interval`` optionally supplies the inclusive topological slab endpoints.
        Omitting it uses the GPU-production ``max(1, Nx // 4)`` rule.
        '''
        Nx, Ny = self.Nx, self.Ny
        alpha = np.full((Nx, Ny), float(alpha_2), dtype=np.complex128)
        interval = self._normalize_dw_interval(dw_interval, Nx)
        if interval is None:
            half = Nx // 2
            slab_half_width_rule = "max(1, Nx // 4)"
            w = max(1, Nx // 4)
            x0 = max(0, half - w)
            x1 = min(Nx, half + w + 1)  # inclusive slab -> slice end-exclusive
        else:
            x0, x1_inclusive = interval
            x1 = x1_inclusive + 1
            w = (x1_inclusive - x0) // 2
            slab_half_width_rule = "explicit_inclusive_interval"
        alpha[x0:x1, :] = float(alpha_1)        # topological region
        print(f"DWs at x=({int(x0)}, {int(x1-1)})")
        self.DW_loc = [int(x0), int(x1-1)]
        self.dw_interval = interval
        self.DW_slab_half_width_rule = slab_half_width_rule
        self.DW_slab_half_width = int(w)
        self.DW_slab_width_sites = int(x1 - x0)
        self.alpha_profile = alpha
        # Backward-compat alias used by some notebooks/plots
        self.alpha = self.alpha_profile

    def construct_OW_projectors(
        self,
        nshell,
        DW,
        trial_orbitals='X',
        dw_truncation=False,
        twist_y=None,
        *,
        twist_x=None,
    ):
        '''Build overcomplete Wannier projectors used for adaptive measurements.

        trial_orbitals : 'X', 'Y', or 'Z'
            Pauli matrix whose eigenstates are used as the two trial spinors τ_A (+ eigenstate)
            and τ_B (− eigenstate).  Default is 'X' (legacy behaviour).
        dw_truncation : bool
            If True and ``DW`` is active, further truncate each Wannier mode so that it has
            support only inside the region (topological or trivial) that contains its center.
            This mask is applied after any ``nshell`` cutoff and before the mode is renormalized.
        '''
        Nx, Ny = self.Nx, self.Ny
        trial_orbitals = str(trial_orbitals).upper()
        self.nshell = self._normalize_nshell(nshell)
        self.DW = bool(DW)
        self.trial_orbitals = trial_orbitals
        self.dw_truncation = bool(dw_truncation)
        if twist_x is not None:
            self.twist_x = float(twist_x)
        elif not hasattr(self, "twist_x"):
            self.twist_x = 0.0
        if twist_y is not None:
            self.twist_y = float(twist_y)
        elif not hasattr(self, "twist_y"):
            self.twist_y = 0.0
        if not np.isfinite(self.twist_x):
            raise ValueError("twist_x must be a finite scalar flux in radians.")
        if not np.isfinite(self.twist_y):
            raise ValueError("twist_y must be a finite scalar flux in radians.")

        # Mass profile alpha(x) for the Dirac-CI model
        if not DW:
            self.alpha_profile = self.alpha_1*np.ones((Nx, Ny), dtype=np.complex128)
        # Keep legacy alias in sync
        self.alpha = self.alpha_profile

        alpha = self.alpha_profile

        # k-grid (FFT order)
        # Uniform Peierls gauges for c(x+Nx,y)=exp(i*twist_x)c(x,y) and
        # c(x,y+Ny)=exp(i*twist_y)c(x,y).
        # Keeping the unwrapped flux is useful for the explicit 0/2pi closure check.
        kx = 2*np.pi * np.fft.fftfreq(Nx) + self.twist_x / Nx
        ky = 2*np.pi * np.fft.fftfreq(Ny) + self.twist_y / Ny
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

        # local 2-spinors: ± eigenstates of the chosen Pauli matrix
        if trial_orbitals == 'X':
            tauA = (1/np.sqrt(2)) * np.array([[1], [1]],   dtype=np.complex128)
            tauB = (1/np.sqrt(2)) * np.array([[1], [-1]],  dtype=np.complex128)
        elif trial_orbitals == 'Y':
            tauA = (1/np.sqrt(2)) * np.array([[1], [1j]],  dtype=np.complex128)
            tauB = (1/np.sqrt(2)) * np.array([[1], [-1j]], dtype=np.complex128)
        elif trial_orbitals == 'Z':
            tauA = np.array([[1], [0]], dtype=np.complex128)
            tauB = np.array([[0], [1]], dtype=np.complex128)
        else:
            raise ValueError(f"trial_orbitals must be 'X', 'Y', or 'Z', got {trial_orbitals!r}")

        # phases for centers R=(Rx,Ry)
        Rx_grid = np.arange(Nx)
        Ry_grid = np.arange(Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])  # (Nx,Ny,Rx,1)
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])  # (Nx,Ny,1,Ry)
        phase   = phase_x * phase_y

        dw_region_mask = None
        dw_center_mask = None
        if dw_truncation and not DW:
            import warnings
            warnings.warn(
                "dw_truncation=True has no effect when DW=False: no domain-wall region mask is defined.",
                UserWarning, stacklevel=2,
            )
        if DW and dw_truncation:
            if not (hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2):
                raise ValueError("dw_truncation requires DW=True with valid DW_loc.")
            xL = int(self.DW_loc[0]) % Nx
            xR = int(self.DW_loc[1]) % Nx
            topo_x = np.zeros(Nx, dtype=bool)
            if xL <= xR:
                topo_x[xL:xR + 1] = True
            else:
                topo_x[xL:] = True
                topo_x[:xR + 1] = True
            topo_region = np.broadcast_to(topo_x[:, None], (Nx, Ny))
            topo_center = np.broadcast_to(topo_x[:, None], (Nx, Ny))
            dw_region_mask = topo_region[:, :, None, None, None]
            dw_center_mask = topo_center[None, None, None, :, :]

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
                mask = self._ow_support_mask(dxw, dyw, nshell)[:, :, None, :, :]
                W = W * mask
            if dw_region_mask is not None:
                # Keep support only in the same DW sector as the Wannier center.
                region_mask = np.where(dw_center_mask, dw_region_mask, ~dw_region_mask)
                W = W * region_mask
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
        # OW-dependent per-site caches are invalid once projectors are rebuilt.
        self._ow_local_cache = {}
        self._ow_local_support_cache = {}

    def _compute_ow_projectors_for_alpha(self, alpha_value, nshell):
        """Compute OW projectors for a uniform mass without mutating instance state."""
        Nx, Ny = self.Nx, self.Ny
        alpha_profile = float(alpha_value) * np.ones((Nx, Ny), dtype=np.complex128)

        kx = 2 * np.pi * np.fft.fftfreq(Nx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        nx = np.sin(KX)[:, :, None, None]
        ny = np.sin(KY)[:, :, None, None]
        nz = alpha_profile[None, None, :, :] - np.cos(KX)[:, :, None, None] - np.cos(KY)[:, :, None, None]
        nmag = np.sqrt(nx**2 + ny**2 + nz**2)
        nmag = np.where(nmag == 0, 1e-15, nmag)

        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)

        hk = (nx[..., None, None] * sx +
              ny[..., None, None] * sy +
              nz[..., None, None] * sz) / nmag[..., None, None]

        Pminus = 0.5 * (I2 - hk)
        Pplus = 0.5 * (I2 + hk)

        tauA = (1 / np.sqrt(2)) * np.array([[1], [1]], dtype=np.complex128)
        tauB = (1 / np.sqrt(2)) * np.array([[1], [-1]], dtype=np.complex128)

        Rx_grid = np.arange(Nx)
        Ry_grid = np.arange(Ny)
        phase_x = np.exp(1j * KX[..., None, None] * Rx_grid[None, None, :, None])
        phase_y = np.exp(1j * KY[..., None, None] * Ry_grid[None, None, None, :])
        phase = phase_x * phase_y

        def k2_to_r2(Ak):
            return np.fft.fft2(Ak, axes=(0, 1))

        def make_W(Pband, tau, phase):
            tau_dag = tau[:, 0].conj()
            psi_k = np.einsum("m,...mn->...n", tau_dag, Pband, optimize=True)

            F0 = phase * psi_k[..., 0]
            F1 = phase * psi_k[..., 1]
            W0 = k2_to_r2(F0)
            W1 = k2_to_r2(F1)

            W = np.moveaxis(np.stack([W0, W1], axis=-1), -1, 2)

            if nshell is not None:
                x = np.arange(Nx)[:, None, None, None]
                y = np.arange(Ny)[None, :, None, None]
                Rx = np.arange(Nx)[None, None, :, None]
                Ry = np.arange(Ny)[None, None, None, :]
                dxw = ((x - Rx + Nx // 2) % Nx) - Nx // 2
                dyw = ((y - Ry + Ny // 2) % Ny) - Ny // 2
                mask = self._ow_support_mask(dxw, dyw, nshell)[:, :, None, :, :]
                W = W * mask
            denom = np.sqrt(np.sum(np.abs(W)**2, axis=(0, 1, 2), keepdims=True)) + 1e-15
            return W / denom

        W_Ap = make_W(Pplus, tauA, phase)
        W_Bp = make_W(Pplus, tauB, phase)
        W_Am = make_W(Pminus, tauA, phase)
        W_Bm = make_W(Pminus, tauB, phase)

        def flatten_centers(W):
            W_mu_xy = np.transpose(W, (2, 0, 1, 3, 4))
            return W_mu_xy.reshape(2 * Nx * Ny, Nx, Ny, order="F")

        return (
            flatten_centers(W_Ap),
            flatten_centers(W_Bp),
            flatten_centers(W_Am),
            flatten_centers(W_Bm),
        )

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
    
    def _domain_wall_hamiltonian(self, periodic=True, alpha=None, triv_region_local_mode=False):
        """
        Build the real-space Dirac/Chern Hamiltonian with spatially varying mass alpha(x,y).

        Here ``alpha_1`` is the topological-region mass and ``alpha_2`` is the trivial-region
        mass.  When ``triv_region_local_mode=True``, the DW slab defined by ``DW_loc`` keeps
        the real-space CI Hamiltonian with onsite mass ``alpha_1``, while the complementary
        trivial region is replaced by decoupled local modes with onsite block ``-I_2``.
        """
        if alpha is None:
            if not hasattr(self, "alpha_profile"):
                self.create_domain_wall(
                    alpha_1=self.alpha_1,
                    alpha_2=self.alpha_2,
                    dw_interval=getattr(self, "dw_interval", None),
                )
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

        topo_mask = None
        if triv_region_local_mode:
            topo_mask = np.zeros((Nx, Ny), dtype=bool)
            if hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2:
                xL = int(self.DW_loc[0]) % Nx
                xR = int(self.DW_loc[1]) % Nx
                if xL <= xR:
                    topo_mask[xL:xR + 1, :] = True
                else:
                    topo_mask[xL:, :] = True
                    topo_mask[:xR + 1, :] = True
            else:
                topo_mask = np.isclose(np.real(alpha), float(self.alpha_1))

            if not np.any(topo_mask):
                raise ValueError("triv_region_local_mode=True requires a non-empty topological region.")

        # onsite mass terms
        for x in range(Nx):
            for y in range(Ny):
                if triv_region_local_mode:
                    onsite = float(self.alpha_1) * sigma_z if topo_mask[x, y] else -np.eye(2, dtype=np.complex128)
                else:
                    onsite = alpha[x, y] * sigma_z
                add_block(x, y, x, y, onsite)

        # nearest-neighbour hoppings
        hop_x = -0.5 * sigma_z - 0.5j * sigma_x
        hop_y = -0.5 * sigma_z - 0.5j * sigma_y  # sign matches sin(k_y)= (e^{ik_y}-e^{-ik_y})/(2i)

        for x in range(Nx):
            for y in range(Ny):
                xp = x + 1
                if xp < Nx:
                    if (not triv_region_local_mode) or (topo_mask[x, y] and topo_mask[xp, y]):
                        add_block(x, y, xp, y, hop_x)
                        add_block(xp, y, x, y, hop_x.conj().T)
                elif periodic:
                    xp = 0
                    if (not triv_region_local_mode) or (topo_mask[x, y] and topo_mask[xp, y]):
                        add_block(x, y, xp, y, hop_x)
                        add_block(xp, y, x, y, hop_x.conj().T)

                yp = y + 1
                if yp < Ny:
                    if (not triv_region_local_mode) or (topo_mask[x, y] and topo_mask[x, yp]):
                        add_block(x, y, x, yp, hop_y)
                        add_block(x, yp, x, y, hop_y.conj().T)
                elif periodic:
                    yp = 0
                    if (not triv_region_local_mode) or (topo_mask[x, y] and topo_mask[x, yp]):
                        add_block(x, y, x, yp, hop_y)
                        add_block(x, yp, x, y, hop_y.conj().T)
        return H

    def G_CI_domain_wall(self, periodic=True, alpha=None, tol=1e-9, triv_region_local_mode=False):
        """
        Build the complex covariance by diagonalizing the real-space DW Hamiltonian.
        """
        H = self._domain_wall_hamiltonian(
            periodic=periodic,
            alpha=alpha,
            triv_region_local_mode=triv_region_local_mode,
        )
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
    
    def measure_only_top_layer(self, G, P, particle=True, symmetrize=True, chi=None):
        '''Apply a charge-conserving measurement update on the top layer without coupling to bottom layer.
            Expected input size: Nlayer x Nlayer
        '''
        Ntot = self.Ntot
        Nlayer = Ntot // 2

        G = np.asarray(G, dtype=np.complex128)
        P = np.asarray(P, dtype=np.complex128)

        Il = np.eye(Nlayer, dtype=np.complex128)
        Gtt = G[:Nlayer, :Nlayer]
        chi_vec = (
            self._rank_one_vector_from_projector(P)
            if chi is None
            else np.asarray(chi, dtype=np.complex128).reshape(-1)
        )
        Q = Il - P
        sign = 1.0 if bool(particle) else -1.0
        mode = self._normalize_physical_covariance_update(
            getattr(self, "_physical_covariance_update_mode", "rank1")
        )
        if mode == "dense":
            solve_mat = Il + sign * (Gtt @ P)
            eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
            if not np.isfinite(eps_scale) or eps_scale < 1.0:
                eps_scale = 1.0
            Z = self._solve_regularized(solve_mat, Gtt @ Q, eps=1e-9 * eps_scale)
        else:
            Z = self._physical_rank1_resolvent_action(Gtt, chi_vec, Gtt @ Q, sign)
        Gp = sign * P + Q @ Z

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
    def local_markov_channel(self, G, Rx, Ry, n_a=0.5, p=1):
        '''
        Applies exact local channel at (Rx, Ry) to the physical (top) layer within the Markov Approximation
        '''

        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        if G.shape != (Nlayer, Nlayer):
            raise ValueError(f"markov_meas_feedback expects top-layer covariance of shape ({Nlayer},{Nlayer}); got {G.shape}")

        Il = self._get_top_eye()
        G_2pt = 0.5 * (G + Il)

        # On-the-fly projectors at (Rx, Ry)
        P_Ap = self._proj_from_WF(self.WF_Ap, Rx, Ry)
        P_Bp = self._proj_from_WF(self.WF_Bp, Rx, Ry)
        P_Am = self._proj_from_WF(self.WF_Am, Rx, Ry)
        P_Bm = self._proj_from_WF(self.WF_Bm, Rx, Ry)

        # Chi spinors — make writable copies
        chi_Ap = np.array(self.WF_Ap[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Bp = np.array(self.WF_Bp[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Am = np.array(self.WF_Am[:, Rx, Ry], dtype=np.complex128, copy=True)
        chi_Bm = np.array(self.WF_Bm[:, Rx, Ry], dtype=np.complex128, copy=True)

        def L_depletion(G, chi, P, n_a):
            return -(P @ G + G @ P) + (1 + n_a) * P * (chi.conj().T @ G @ chi)
        
        def L_fill(G, chi, P, n_a):
            return n_a * P - (P @ G + G @ P) + (2-n_a) * P * (chi.conj().T @ G @ chi)
        
        # Upper band A: Deplete
        G_2pt += p * L_depletion(G_2pt, chi_Ap, P_Ap, n_a)
        # Upper band B: Deplete
        G_2pt += p * L_depletion(G_2pt, chi_Bp, P_Bp, n_a)
        # Lower band A: Fill
        G_2pt += p * L_fill(G_2pt, chi_Am, P_Am, n_a)   
        # Lower band B: Fill
        G_2pt += p * L_fill(G_2pt, chi_Bm, P_Bm, n_a)

        return 2 * G_2pt - Il
    
    def _markov_meas_feedback_frame(
        self,
        state,
        Rx,
        Ry,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        local_mode=False,
        perfect_correction=False,
        return_weight_summary=False,
        rng=None,
        branch_replay_events=None,
        replay_probability_tol=1e-14,
        state_event_observer=None,
        event_context=None,
        sample_offsets=None,
        choi_state=None,
        choi_context=None,
        lyapunov_state=None,
    ):
        """Frame-native version of one canonical Markov site update.

        Schedule construction and cycle traversal remain in ``run_markov_circuit``.
        This routine only replaces the physical state adapter used by the four
        ordered OW channels at a visited site.
        """
        if not isinstance(state, OccupiedFrameState):
            raise TypeError("state must be an OccupiedFrameState.")
        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        replay_events = (
            None
            if branch_replay_events is None
            else tuple(dict(event) for event in branch_replay_events)
        )
        replay_position = 0
        replay_probability_tol = float(replay_probability_tol)
        if not np.isfinite(replay_probability_tol) or replay_probability_tol < 0.0:
            raise ValueError("replay_probability_tol must be a finite nonnegative scalar.")
        measurement_log_weight = 0.0
        correction_log_weight = 0.0
        branch_events = []
        timing = state.timing
        if sample_offsets is None:
            sample_offsets = np.asarray([0], dtype=np.int64)

        def _channel_context(channel):
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = str(channel)
            return payload

        def _consume_replay_event(kind, channel):
            nonlocal replay_position
            if replay_events is None:
                return None
            if replay_position >= len(replay_events):
                raise ValueError(
                    f"Trajectory replay ended before {kind} event {channel!r}."
                )
            event = replay_events[replay_position]
            replay_position += 1
            if str(event.get("kind")) != str(kind) or str(event.get("channel")) != str(channel):
                raise ValueError(
                    "Trajectory replay event mismatch at site "
                    f"({int(Rx)},{int(Ry)}): expected ({kind},{channel}), got "
                    f"({event.get('kind')},{event.get('channel')})."
                )
            return event

        def _selected_probability(probability, occurred):
            return float(probability if bool(occurred) else 1.0 - probability)

        def _validate_selected(probability, occurred, *, kind, channel):
            selected = _selected_probability(probability, occurred)
            if not np.isfinite(selected) or selected <= replay_probability_tol:
                raise FloatingPointError(
                    "The forced frame trajectory branch has zero, non-finite, or "
                    f"sub-tolerance probability at site ({int(Rx)},{int(Ry)}), "
                    f"kind={kind!r}, channel={channel!r}, selected_probability={selected!r}, "
                    f"tolerance={replay_probability_tol:.3e}."
                )
            return selected

        def _sample_measurement(channel, p_occ):
            with timing.measure("branch_sample_or_validate", detailed=True):
                replay_event = _consume_replay_event("measurement", channel)
                if replay_event is None:
                    return bool(self._rng_random(rng) < p_occ), None
                outcome = bool(replay_event["outcome_occupied"])
                _validate_selected(
                    p_occ, outcome, kind="measurement", channel=channel
                )
                return outcome, replay_event

        def _event_log(probability):
            clipped = float(
                np.clip(probability, np.finfo(np.float64).tiny, 1.0)
            )
            return float(np.log(clipped))

        def _record_measurement(channel, p_occ, outcome, replay_event):
            nonlocal measurement_log_weight
            selected = _selected_probability(p_occ, outcome)
            logp = _event_log(selected)
            measurement_log_weight += logp
            # Compact replay records only need to pin the realized branch.  The
            # selected probability is recomputed from the current state above,
            # so a saved probability is optional diagnostic provenance rather
            # than part of the physical replay contract.
            reference_value = (
                None if replay_event is None else replay_event.get("probability")
            )
            reference = (
                None if reference_value is None else float(reference_value)
            )
            with timing.measure("branch_record_bookkeeping", detailed=True):
                branch_events.append(
                    {
                        "channel": channel,
                        "kind": "measurement",
                        "probability": float(p_occ),
                        "outcome_occupied": bool(outcome),
                        "log_weight": logp,
                        "replay_reference_probability": reference,
                        "replay_probability_error": (
                            None if reference is None else float(abs(p_occ - reference))
                        ),
                    }
                )

        def _sample_feedback_target(expected_occupied, channel):
            with timing.measure("branch_sample_or_validate", detailed=True):
                replay_event = _consume_replay_event("correction", channel)
                if replay_event is not None:
                    if bool(replay_event.get("expected_occupied")) != bool(expected_occupied):
                        raise ValueError(
                            f"Trajectory replay correction target mismatch for channel {channel!r}."
                        )
                    target = bool(replay_event["target_occupied"])
                    if perfect_correction:
                        probability = 1.0 if target == bool(expected_occupied) else 0.0
                        occurred = True
                    elif bool(expected_occupied):
                        probability = p_gain_eff
                        occurred = target
                    else:
                        probability = p_loss_eff
                        occurred = not target
                    _validate_selected(
                        probability, occurred, kind="correction", channel=channel
                    )
                    return target, replay_event
                if perfect_correction:
                    return bool(expected_occupied), None
                if bool(expected_occupied):
                    return bool(self._rng_random(rng) < p_gain_eff), None
                return not bool(self._rng_random(rng) < p_loss_eff), None

        def _record_correction(channel, expected_occupied, target, replay_event):
            nonlocal correction_log_weight
            if perfect_correction:
                probability = 1.0
                logp = 0.0
            elif bool(expected_occupied):
                probability = float(p_gain_eff)
                logp = _event_log(probability if target else 1.0 - probability)
            else:
                probability = float(p_loss_eff)
                logp = _event_log(probability if not target else 1.0 - probability)
            correction_log_weight += logp
            with timing.measure("branch_record_bookkeeping", detailed=True):
                branch_events.append(
                    {
                        "channel": channel,
                        "kind": "correction",
                        "expected_occupied": bool(expected_occupied),
                        "target_occupied": bool(target),
                        "probability": probability,
                        "perfect_correction": bool(perfect_correction),
                        "log_weight": float(logp),
                        "replay_reference_probability": (
                            None
                            if replay_event is None
                            else float(replay_event.get("probability", probability))
                        ),
                    }
                )

        if local_mode:
            e_mu1, e_mu2, _, _, _, _ = self._get_local_mode_ops(Rx, Ry)
            channels = (
                ("A", False, np.arange(e_mu1.size, dtype=np.int64), e_mu1),
                ("B", True, np.arange(e_mu2.size, dtype=np.int64), e_mu2),
            )
        elif self.nshell is not None:
            with timing.measure("orbital_fetch", detailed=True):
                payload = self._get_ow_local_support_data(Rx, Ry)
            channels = tuple(
                (channel, expected, payload["idx"], payload[channel])
                for channel, expected in (
                    ("Ap", False),
                    ("Am", True),
                    ("Bp", False),
                    ("Bm", True),
                )
            )
        else:
            with timing.measure("orbital_fetch", detailed=True):
                payload = self._get_ow_local_data(Rx, Ry)
            all_idx = np.arange(state.physical_dimension, dtype=np.int64)
            channels = (
                ("Ap", False, all_idx, payload["chi_Ap"]),
                ("Am", True, all_idx, payload["chi_Am"]),
                ("Bp", False, all_idx, payload["chi_Bp"]),
                ("Bm", True, all_idx, payload["chi_Bm"]),
            )

        for channel, expected_occupied, support_idx, orbital_local in channels:
            with timing.measure("channel_total", detailed=True):
                p_occ = state.occupation_probability_local(support_idx, orbital_local)
                outcome, measurement_replay = _sample_measurement(channel, p_occ)
                _record_measurement(channel, p_occ, outcome, measurement_replay)
                self._lyapunov_apply_occupied_frame_channel(
                    lyapunov_state,
                    state,
                    sample_offsets,
                    support_idx,
                    orbital_local,
                    particle=outcome,
                )
                mismatch = bool(outcome) != bool(expected_occupied)
                target = bool(outcome)
                if mismatch:
                    target, correction_replay = _sample_feedback_target(
                        expected_occupied, channel
                    )
                    _record_correction(
                        channel, expected_occupied, target, correction_replay
                    )
                    self._lyapunov_reset_occupied_frame_channel(
                        lyapunov_state,
                        sample_offsets,
                        support_idx,
                        orbital_local,
                    )

                selected_probability = _selected_probability(p_occ, outcome)
                if mismatch and target != bool(outcome):
                    timing.increment("simplified_mismatch_update_count")
                    with timing.measure("feedback_total", detailed=True):
                        if target:
                            operation_probability = state.gain_local(
                                support_idx, orbital_local
                            ).probability
                        else:
                            operation_probability = state.loss_local(
                                support_idx, orbital_local
                            ).probability
                else:
                    with timing.measure("measurement_total", detailed=True):
                        if outcome:
                            operation_probability = state.project_occupied_local(
                                support_idx, orbital_local
                            )
                        else:
                            operation_probability = state.project_empty_local(
                                support_idx, orbital_local
                            )
                if abs(operation_probability - selected_probability) > max(
                    1e-10, 50.0 * replay_probability_tol
                ):
                    raise FloatingPointError(
                        "Frame letter probability disagrees with its Born overlap at "
                        f"site ({int(Rx)},{int(Ry)}), channel={channel!r}: "
                        f"letter={operation_probability:.16g}, selected={selected_probability:.16g}."
                    )
                if choi_state is not None:
                    self._choi_apply_rank_one(
                        choi_state,
                        sample_offsets,
                        orbital_local,
                        eta1=-(1.0 if outcome else -1.0),
                        eta2=(1.0 if target else -1.0),
                        support_idx=(
                            support_idx
                            if np.asarray(support_idx).size != state.physical_dimension
                            else None
                        ),
                        context=_channel_context(channel),
                    )
                timing.increment(
                    f"channel.{channel}.expected_{int(expected_occupied)}."
                    f"outcome_{int(outcome)}.corrected_{int(mismatch)}"
                )
                if state_event_observer is not None:
                    payload = {} if event_context is None else dict(event_context)
                    state_event_observer(
                        **payload,
                        channel=str(channel),
                        expected_occupied=bool(expected_occupied),
                        outcome_occupied=bool(outcome),
                        correction_applied=bool(mismatch),
                        target_occupied=bool(target),
                        state=state,
                    )

        if replay_events is not None and replay_position != len(replay_events):
            raise ValueError(
                "Trajectory replay site contained unused branch events: "
                f"consumed {replay_position} of {len(replay_events)}."
            )
        branch_log_weight = float(measurement_log_weight + correction_log_weight)
        state.log_weight += branch_log_weight
        if not return_weight_summary:
            return state
        return state, {
            "branch_log_weight": branch_log_weight,
            "measurement_log_weight": float(measurement_log_weight),
            "correction_log_weight": float(correction_log_weight),
            "forced_postselect": False,
            "branch_events": tuple(branch_events),
        }

    def markov_meas_feedback(
        self,
        G,
        Rx,
        Ry,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        local_mode=False,
        perfect_correction=False,
        sample_offsets=None,
        choi_state=None,
        choi_context=None,
        lyapunov_state=None,
        return_weight_summary=False,
        rng=None,
        branch_replay_events=None,
        replay_probability_tol=1e-14,
        state_event_observer=None,
        event_context=None,
    ):
        '''Perform Markovian measurement plus feedback at site (Rx, Ry) on top-layer covariance.
        
        If local_mode is True, bypass the overcomplete Wannier feedback and measure only the
        canonical μ=1 (expected occupied) and μ=2 (expected unoccupied) orbitals at (Rx,Ry).
        If perfect_correction is True, corrective ancilla swaps are deterministic:
        pump-out enforces unoccupied and pump-in enforces occupied, independent of n_a.
        '''
        if isinstance(G, OccupiedFrameState):
            return self._markov_meas_feedback_frame(
                G,
                Rx,
                Ry,
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                local_mode=local_mode,
                perfect_correction=perfect_correction,
                return_weight_summary=return_weight_summary,
                rng=rng,
                branch_replay_events=branch_replay_events,
                replay_probability_tol=replay_probability_tol,
                state_event_observer=state_event_observer,
                event_context=event_context,
                sample_offsets=sample_offsets,
                choi_state=choi_state,
                choi_context=choi_context,
                lyapunov_state=lyapunov_state,
            )

        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        if G.shape != (Nlayer, Nlayer):
            raise ValueError(f"markov_meas_feedback expects top-layer covariance of shape ({Nlayer},{Nlayer}); got {G.shape}")

        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        Il = self._get_top_eye()
        use_local_ow = (not local_mode) and (self.nshell is not None)
        if (choi_state is not None or lyapunov_state is not None) and sample_offsets is None:
            sample_offsets = np.asarray([0], dtype=np.int64)
        if sample_offsets is None:
            sample_offsets = np.asarray([0], dtype=np.int64)

        measurement_log_weight = 0.0
        correction_log_weight = 0.0
        branch_events = []
        replay_events = (
            None
            if branch_replay_events is None
            else tuple(dict(event) for event in branch_replay_events)
        )
        replay_position = 0
        replay_probability_tol = float(replay_probability_tol)
        if not np.isfinite(replay_probability_tol) or replay_probability_tol < 0.0:
            raise ValueError("replay_probability_tol must be a finite nonnegative scalar.")

        def _consume_replay_event(kind, channel):
            nonlocal replay_position
            if replay_events is None:
                return None
            if replay_position >= len(replay_events):
                raise ValueError(
                    f"Trajectory replay ended before {kind} event {channel!r}."
                )
            event = replay_events[replay_position]
            replay_position += 1
            if str(event.get("kind")) != str(kind) or str(event.get("channel")) != str(channel):
                raise ValueError(
                    "Trajectory replay event mismatch at site "
                    f"({int(Rx)},{int(Ry)}): expected ({kind},{channel}), got "
                    f"({event.get('kind')},{event.get('channel')})."
                )
            return event

        def _validate_replayed_probability(probability, occurred, *, kind, channel):
            selected = float(probability if bool(occurred) else 1.0 - probability)
            if not np.isfinite(selected) or selected <= replay_probability_tol:
                raise FloatingPointError(
                    "The forced trajectory branch has zero, non-finite, or sub-tolerance "
                    f"probability at site ({int(Rx)},{int(Ry)}), kind={kind!r}, "
                    f"channel={channel!r}, selected_probability={selected!r}, "
                    f"tolerance={replay_probability_tol:.3e}."
                )

        def _sample_measurement(channel, p_occ):
            with self._update_timer("branch_sample_or_validate", detailed=True):
                replay_event = _consume_replay_event("measurement", channel)
                if replay_event is None:
                    return bool(self._rng_random(rng) < p_occ)
                outcome = bool(replay_event["outcome_occupied"])
                _validate_replayed_probability(
                    p_occ, outcome, kind="measurement", channel=channel
                )
                return outcome

        def _event_log(prob, occurred=True):
            p = float(np.clip(prob if occurred else 1.0 - prob, np.finfo(np.float64).tiny, 1.0))
            return float(np.log(p))

        def _channel_context(channel):
            if choi_state is None:
                return None
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = channel
            return payload

        def _record_measurement(channel, p_occ, occ_event):
            nonlocal measurement_log_weight
            logp = _event_log(p_occ, bool(occ_event))
            measurement_log_weight += logp
            with self._update_timer("branch_record_bookkeeping", detailed=True):
                branch_events.append(
                    {
                        "channel": channel,
                        "kind": "measurement",
                        "probability": float(p_occ),
                        "outcome_occupied": bool(occ_event),
                        "log_weight": logp,
                    }
                )

        def _record_correction(channel, expected_occupied, target):
            nonlocal correction_log_weight
            if perfect_correction:
                logp = 0.0
                prob = 1.0
            elif bool(expected_occupied):
                prob = p_gain_eff
                logp = _event_log(prob, bool(target))
            else:
                prob = p_loss_eff
                logp = _event_log(prob, not bool(target))
            correction_log_weight += logp
            with self._update_timer("branch_record_bookkeeping", detailed=True):
                branch_events.append(
                    {
                        "channel": channel,
                        "kind": "correction",
                        "expected_occupied": bool(expected_occupied),
                        "target_occupied": bool(target),
                        "probability": float(prob),
                        "perfect_correction": bool(perfect_correction),
                        "log_weight": float(logp),
                    }
                )

        def _apply_choi(chi, s_in, s_out, channel, support_idx=None):
            if choi_state is None:
                return
            self._choi_apply_rank_one(
                choi_state,
                sample_offsets,
                chi,
                eta1=-float(s_in),
                eta2=float(s_out),
                support_idx=support_idx,
                context=_channel_context(channel),
            )

        def _sample_feedback_target(expected_occupied, channel):
            with self._update_timer("branch_sample_or_validate", detailed=True):
                replay_event = _consume_replay_event("correction", channel)
                if replay_event is not None:
                    if bool(replay_event.get("expected_occupied")) != bool(expected_occupied):
                        raise ValueError(
                            f"Trajectory replay correction target mismatch for channel {channel!r}."
                        )
                    target = bool(replay_event["target_occupied"])
                    if perfect_correction:
                        probability = 1.0 if target == bool(expected_occupied) else 0.0
                        occurred = True
                    elif bool(expected_occupied):
                        probability = p_gain_eff
                        occurred = target
                    else:
                        probability = p_loss_eff
                        occurred = not target
                    _validate_replayed_probability(
                        probability, occurred, kind="correction", channel=channel
                    )
                    return target
                if perfect_correction:
                    return bool(expected_occupied)
                if bool(expected_occupied):
                    return bool(self._rng_random(rng) < p_gain_eff)
                return not bool(self._rng_random(rng) < p_loss_eff)

        def _finish(Gout):
            if replay_events is not None and replay_position != len(replay_events):
                raise ValueError(
                    "Trajectory replay site contained unused branch events: "
                    f"consumed {replay_position} of {len(replay_events)}."
                )
            if not return_weight_summary:
                return Gout
            summary = {
                "branch_log_weight": float(measurement_log_weight + correction_log_weight),
                "measurement_log_weight": float(measurement_log_weight),
                "correction_log_weight": float(correction_log_weight),
                "forced_postselect": False,
                "branch_events": tuple(branch_events),
            }
            return Gout, summary

        def _measure(G, P, Q, chi, particle):
            sign = 1.0 if bool(particle) else -1.0
            mode = self._normalize_physical_covariance_update(
                getattr(self, "_physical_covariance_update_mode", "rank1")
            )
            if mode == "dense":
                solve_mat = Il + sign * (G @ P)
                eps_scale = np.linalg.norm(solve_mat, ord=np.inf)
                if not np.isfinite(eps_scale) or eps_scale < 1.0:
                    eps_scale = 1.0
                Z = self._solve_regularized(solve_mat, G @ Q, eps=1e-9 * eps_scale)
            else:
                Z = self._physical_rank1_resolvent_action(G, chi, G @ Q, sign)
            G_upd = sign * P + Q @ Z
            return 0.5 * (G_upd + G_upd.conj().T)

        def _ancilla_swap_top(Gtop, P, n_a, target_occupied=None, chi=None):
            if target_occupied is None:
                if self._rng_random(rng) < n_a:
                    ancilla_cov = 1
                else:
                    ancilla_cov = -1
            else:
                ancilla_cov = 1 if bool(target_occupied) else -1

            if chi is not None:
                v = np.asarray(chi, dtype=np.complex128).reshape(-1)
                vn = np.linalg.norm(v)
                if np.isfinite(vn) and vn > 0:
                    v = v / vn
                gv = Gtop @ v
                vG = v.conj().T @ Gtop
                scalar = v.conj().T @ gv
                P_loc = np.outer(v, v.conj())
                Gnew = Gtop - np.outer(v, vG) - np.outer(gv, v.conj()) + (scalar + ancilla_cov) * P_loc
            else:
                Q = Il - P
                Gnew = Q @ Gtop @ Q + ancilla_cov * P

            Gnew = 0.5 * (Gnew + Gnew.conj().T)
            if not np.all(np.isfinite(Gnew)):
                raise FloatingPointError(f"Non-finite ancilla swap result (n_a={n_a}, occ={ancilla_cov})")
            return Gnew, float(ancilla_cov)

        # Optional local mode: measure the canonical μ=1 (unoccupied) and μ=2 (occupied) orbitals only. 
        if local_mode:
            e_mu1, e_mu2, P_mu1, P_mu2, Q_mu1, Q_mu2 = self._get_local_mode_ops(Rx, Ry)

            p_occ = self._occ_prob_from_vector(G, e_mu1)
            occ_event = _sample_measurement("A", p_occ)  # expect unoccupied
            _record_measurement("A", p_occ, occ_event)
            s_in = 1.0 if occ_event else -1.0
            s_out = s_in
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_mu1, particle=occ_event)
            if occ_event:
                G = _measure(G, P_mu1, Q_mu1, e_mu1, particle=occ_event)
                target = _sample_feedback_target(False, "A")
                _record_correction("A", False, target)
                self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_mu1)
                G, s_out = _ancilla_swap_top(G, P_mu1, n_a, target_occupied=target, chi=e_mu1)
            else:
                G = _measure(G, P_mu1, Q_mu1, e_mu1, particle=occ_event)
            _apply_choi(e_mu1, s_in, s_out, "A")

            p_occ = self._occ_prob_from_vector(G, e_mu2)
            occ_event = _sample_measurement("B", p_occ)  # expect occupied
            _record_measurement("B", p_occ, occ_event)
            s_in = 1.0 if occ_event else -1.0
            s_out = s_in
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_mu2, particle=occ_event)
            if occ_event:
                G = _measure(G, P_mu2, Q_mu2, e_mu2, particle=occ_event)
            else:
                G = _measure(G, P_mu2, Q_mu2, e_mu2, particle=occ_event)
                target = _sample_feedback_target(True, "B")
                _record_correction("B", True, target)
                self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_mu2)
                G, s_out = _ancilla_swap_top(G, P_mu2, n_a, target_occupied=target, chi=e_mu2)
            _apply_choi(e_mu2, s_in, s_out, "B")
            return _finish(G)

        if use_local_ow:
            with self._update_timer("orbital_fetch", detailed=True):
                payload = self._get_ow_local_support_data(Rx, Ry)

            def _apply_local_channel(Gloc, channel_key, expected_occupied):
                collector = getattr(self, "_update_timing_collector", None)
                channel_started = (
                    time.perf_counter_ns()
                    if collector is not None and collector.enabled(detailed=True)
                    else None
                )
                chi_local = payload[channel_key]
                p_occ = self._occ_prob_from_vector(
                    self._local_support_block(Gloc, payload["idx"], np.array([], dtype=np.int64))[0],
                    chi_local,
                )
                occ_event = _sample_measurement(channel_key, p_occ)
                _record_measurement(channel_key, p_occ, occ_event)
                s_in = 1.0 if occ_event else -1.0
                s_out = s_in

                if lyapunov_state is not None:
                    self._lyapunov_apply_local_channel(
                        lyapunov_state,
                        Gloc,
                        sample_offsets,
                        payload["idx"],
                        payload["comp"],
                        chi_local,
                        particle=bool(occ_event),
                    )
                with self._update_timer("measurement_total", detailed=True):
                    Gloc = self._measure_only_top_layer_local(
                        Gloc,
                        payload["idx"],
                        payload["comp"],
                        chi_local,
                        particle=bool(occ_event),
                    )
                mismatch = bool(occ_event) if not expected_occupied else (not bool(occ_event))
                if mismatch:
                    target = _sample_feedback_target(expected_occupied, channel_key)
                    s_out = 1.0 if bool(target) else -1.0
                    _record_correction(channel_key, expected_occupied, target)
                    self._lyapunov_apply_local_reset(
                        lyapunov_state, sample_offsets, payload["idx"], payload["comp"], chi_local
                    )
                    with self._update_timer("feedback_total", detailed=True):
                        Gloc = self._ancilla_swap_top_local(
                            Gloc,
                            payload["idx"],
                            payload["comp"],
                            chi_local,
                            n_a=n_a,
                            target_occupied=target,
                        )
                _apply_choi(chi_local, s_in, s_out, channel_key, support_idx=payload["idx"])
                self._update_count(
                    f"channel.{channel_key}.expected_{int(expected_occupied)}."
                    f"outcome_{int(occ_event)}.corrected_{int(mismatch)}"
                )
                if state_event_observer is not None:
                    observer_payload = {} if event_context is None else dict(event_context)
                    state_event_observer(
                        **observer_payload,
                        channel=str(channel_key),
                        expected_occupied=bool(expected_occupied),
                        outcome_occupied=bool(occ_event),
                        correction_applied=bool(mismatch),
                        target_occupied=bool(target) if mismatch else bool(occ_event),
                        state=Gloc,
                    )
                if channel_started is not None:
                    collector.add_time(
                        "channel_total",
                        time.perf_counter_ns() - channel_started,
                        detailed=True,
                    )
                return Gloc

            G = _apply_local_channel(G, "Ap", expected_occupied=False)
            G = _apply_local_channel(G, "Am", expected_occupied=True)
            G = _apply_local_channel(G, "Bp", expected_occupied=False)
            G = _apply_local_channel(G, "Bm", expected_occupied=True)
            return _finish(G)

        site_data = self._get_ow_local_data(Rx, Ry)
        chi_Ap = site_data["chi_Ap"]
        chi_Bp = site_data["chi_Bp"]
        chi_Am = site_data["chi_Am"]
        chi_Bm = site_data["chi_Bm"]
        P_Ap = np.outer(chi_Ap, chi_Ap.conj())
        P_Bp = np.outer(chi_Bp, chi_Bp.conj())
        P_Am = np.outer(chi_Am, chi_Am.conj())
        P_Bm = np.outer(chi_Bm, chi_Bm.conj())
        Q_Ap = Il - P_Ap
        Q_Bp = Il - P_Bp
        Q_Am = Il - P_Am
        Q_Bm = Il - P_Bm

        # Upper/lower A channels
        p_occ = self._occ_prob_from_vector(G, chi_Ap)
        occ_event = _sample_measurement("Ap", p_occ)
        _record_measurement("Ap", p_occ, occ_event)
        s_in = 1.0 if occ_event else -1.0
        s_out = s_in
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Ap, particle=occ_event)
        G = _measure(G, P_Ap, Q_Ap, chi_Ap, particle=occ_event)
        if occ_event:
            target = _sample_feedback_target(False, "Ap")
            _record_correction("Ap", False, target)
            self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_Ap)
            G, s_out = _ancilla_swap_top(G, P_Ap, n_a, target_occupied=target, chi=chi_Ap)  # pump out
        _apply_choi(chi_Ap, s_in, s_out, "Ap")

        p_occ = self._occ_prob_from_vector(G, chi_Am)
        occ_event = _sample_measurement("Am", p_occ)
        _record_measurement("Am", p_occ, occ_event)
        s_in = 1.0 if occ_event else -1.0
        s_out = s_in
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Am, particle=occ_event)
        G = _measure(G, P_Am, Q_Am, chi_Am, particle=occ_event)
        if not occ_event:
            target = _sample_feedback_target(True, "Am")
            _record_correction("Am", True, target)
            self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_Am)
            G, s_out = _ancilla_swap_top(G, P_Am, n_a, target_occupied=target, chi=chi_Am)  # pump in
        _apply_choi(chi_Am, s_in, s_out, "Am")

        # Upper/lower B channels
        p_occ = self._occ_prob_from_vector(G, chi_Bp)
        occ_event = _sample_measurement("Bp", p_occ)
        _record_measurement("Bp", p_occ, occ_event)
        s_in = 1.0 if occ_event else -1.0
        s_out = s_in
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Bp, particle=occ_event)
        G = _measure(G, P_Bp, Q_Bp, chi_Bp, particle=occ_event)
        if occ_event:
            target = _sample_feedback_target(False, "Bp")
            _record_correction("Bp", False, target)
            self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_Bp)
            G, s_out = _ancilla_swap_top(G, P_Bp, n_a, target_occupied=target, chi=chi_Bp)  # pump out
        _apply_choi(chi_Bp, s_in, s_out, "Bp")

        p_occ = self._occ_prob_from_vector(G, chi_Bm)
        occ_event = _sample_measurement("Bm", p_occ)
        _record_measurement("Bm", p_occ, occ_event)
        s_in = 1.0 if occ_event else -1.0
        s_out = s_in
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Bm, particle=occ_event)
        G = _measure(G, P_Bm, Q_Bm, chi_Bm, particle=occ_event)
        if not occ_event:
            target = _sample_feedback_target(True, "Bm")
            _record_correction("Bm", True, target)
            self._lyapunov_apply_dense_reset(lyapunov_state, sample_offsets, P_Bm)
            G, s_out = _ancilla_swap_top(G, P_Bm, n_a, target_occupied=target, chi=chi_Bm)  # pump in
        _apply_choi(chi_Bm, s_in, s_out, "Bm")

        return _finish(G)

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
        G = self.measure_top_layer(G, P_Am, particle=True)   # Am occ
        G = self.measure_top_layer(G, P_Bp, particle=False)  # Bp unocc
        G = self.measure_top_layer(G, P_Bm, particle=True)   # Bm occ
        return G

    def post_selection_markov_top_layer(
        self,
        G,
        Rx,
        Ry,
        local_mode=False,
        sample_offsets=None,
        choi_state=None,
        choi_context=None,
        lyapunov_state=None,
    ):
        """
        Post-selection update for top-layer-only Markov covariance (Nlayer x Nlayer).
        Uses measure_only_top_layer, so it is safe for run_markov_circuit states.
        """
        if isinstance(G, OccupiedFrameState):
            if sample_offsets is None:
                sample_offsets = np.asarray([0], dtype=np.int64)
            if local_mode:
                e_mu1, e_mu2, _, _, _, _ = self._get_local_mode_ops(Rx, Ry)
                channels = (
                    ("A", False, np.arange(e_mu1.size, dtype=np.int64), e_mu1),
                    ("B", True, np.arange(e_mu2.size, dtype=np.int64), e_mu2),
                )
            elif self.nshell is not None:
                payload = self._get_ow_local_support_data(Rx, Ry)
                channels = tuple(
                    (label, target, payload["idx"], payload[label])
                    for label, target in (
                        ("Ap", False),
                        ("Am", True),
                        ("Bp", False),
                        ("Bm", True),
                    )
                )
            else:
                payload = self._get_ow_local_data(Rx, Ry)
                all_idx = np.arange(G.physical_dimension, dtype=np.int64)
                channels = (
                    ("Ap", False, all_idx, payload["chi_Ap"]),
                    ("Am", True, all_idx, payload["chi_Am"]),
                    ("Bp", False, all_idx, payload["chi_Bp"]),
                    ("Bm", True, all_idx, payload["chi_Bm"]),
                )
            for label, target_occupied, support_idx, orbital_local in channels:
                self._lyapunov_apply_occupied_frame_channel(
                    lyapunov_state,
                    G,
                    sample_offsets,
                    support_idx,
                    orbital_local,
                    particle=target_occupied,
                )
                if target_occupied:
                    G.project_occupied_local(support_idx, orbital_local)
                else:
                    G.project_empty_local(support_idx, orbital_local)
                if choi_state is not None:
                    sign = 1.0 if target_occupied else -1.0
                    context = {} if choi_context is None else dict(choi_context)
                    context["channel"] = label
                    self._choi_apply_rank_one(
                        choi_state,
                        sample_offsets,
                        orbital_local,
                        eta1=-sign,
                        eta2=sign,
                        support_idx=(
                            support_idx
                            if np.asarray(support_idx).size != G.physical_dimension
                            else None
                        ),
                        context=context,
                    )
            return G
        G = np.asarray(G, dtype=np.complex128)
        Nlayer = self.Ntot // 2
        if G.shape != (Nlayer, Nlayer):
            raise ValueError(
                f"post_selection_markov_top_layer expects shape ({Nlayer},{Nlayer}); got {G.shape}"
            )
        if (choi_state is not None or lyapunov_state is not None) and sample_offsets is None:
            sample_offsets = np.asarray([0], dtype=np.int64)
        if sample_offsets is None:
            sample_offsets = np.asarray([0], dtype=np.int64)

        def _channel_context(channel):
            if choi_state is None:
                return None
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = channel
            return payload

        def _apply_choi(chi, particle, channel, support_idx=None):
            if choi_state is None:
                return
            sign = 1.0 if bool(particle) else -1.0
            self._choi_apply_rank_one(
                choi_state,
                sample_offsets,
                chi,
                eta1=-sign,
                eta2=sign,
                support_idx=support_idx,
                context=_channel_context(channel),
            )

        if local_mode:
            e_mu1, e_mu2, P_mu1, P_mu2, _, _ = self._get_local_mode_ops(Rx, Ry)

            _apply_choi(e_mu1, False, "A")
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_mu1, particle=False)
            G = self.measure_only_top_layer(G, P_mu1, particle=False, chi=e_mu1)
            _apply_choi(e_mu2, True, "B")
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_mu2, particle=True)
            G = self.measure_only_top_layer(G, P_mu2, particle=True, chi=e_mu2)
            return G

        if self.nshell is not None:
            payload = self._get_ow_local_support_data(Rx, Ry)
            _apply_choi(payload["Ap"], False, "Ap", support_idx=payload["idx"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Ap"], particle=False
                )
            G = self._measure_only_top_layer_local(G, payload["idx"], payload["comp"], payload["Ap"], particle=False)
            _apply_choi(payload["Am"], True, "Am", support_idx=payload["idx"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Am"], particle=True
                )
            G = self._measure_only_top_layer_local(G, payload["idx"], payload["comp"], payload["Am"], particle=True)
            _apply_choi(payload["Bp"], False, "Bp", support_idx=payload["idx"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Bp"], particle=False
                )
            G = self._measure_only_top_layer_local(G, payload["idx"], payload["comp"], payload["Bp"], particle=False)
            _apply_choi(payload["Bm"], True, "Bm", support_idx=payload["idx"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Bm"], particle=True
                )
            G = self._measure_only_top_layer_local(G, payload["idx"], payload["comp"], payload["Bm"], particle=True)
            return G

        site_data = self._get_ow_local_data(Rx, Ry)
        chi_Ap = site_data["chi_Ap"]
        chi_Bp = site_data["chi_Bp"]
        chi_Am = site_data["chi_Am"]
        chi_Bm = site_data["chi_Bm"]
        P_Ap = np.outer(chi_Ap, chi_Ap.conj())
        P_Bp = np.outer(chi_Bp, chi_Bp.conj())
        P_Am = np.outer(chi_Am, chi_Am.conj())
        P_Bm = np.outer(chi_Bm, chi_Bm.conj())

        _apply_choi(chi_Ap, False, "Ap")
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Ap, particle=False)
        G = self.measure_only_top_layer(G, P_Ap, particle=False, chi=chi_Ap)  # Ap unocc
        _apply_choi(chi_Am, True, "Am")
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Am, particle=True)
        G = self.measure_only_top_layer(G, P_Am, particle=True, chi=chi_Am)   # Am occ
        _apply_choi(chi_Bp, False, "Bp")
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Bp, particle=False)
        G = self.measure_only_top_layer(G, P_Bp, particle=False, chi=chi_Bp)  # Bp unocc
        _apply_choi(chi_Bm, True, "Bm")
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P_Bm, particle=True)
        G = self.measure_only_top_layer(G, P_Bm, particle=True, chi=chi_Bm)   # Bm occ
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

    def _sequence_helper(self, sequence, Nx=None, Ny=None, rng=None, dw_exclude=None, top_region_last=False, skip_trivial=False):
        """
        Build iterators for site-ordering across measurement-feedback sweeps.

        This helper generates the coordinate list for a single circuit cycle, handling 
        geometric patterns, exclusion zones, and region prioritization.

        Parameters
        ----------
        sequence : str
            The fundamental spatial scanning pattern.
            
            Supported scan patterns:
            - "raster_y": Scans columns (x) left-to-right, within each column scans y bottom-to-top.
            - "raster_x": Scans rows (y) bottom-to-top, within each row scans x left-to-right.
            - "reverse_raster_y": Exact reverse of the filtered ``raster_y`` word.
            - "random": Randomized permutation of all sites, reshuffled every cycle.
            - "dw_symmetric_random": For each sweep, scan x in DW-boundary-first
              order (right DW boundary downwards in x with periodic wrap), and for
              each Rx choose a fresh random permutation of y.

            Legacy aliases "snake_y" and "snake_x" are accepted and mapped to
            "raster_y" and "raster_x", respectively.

        dw_exclude : int or None
            If an integer s >= 0 is provided (and self.DW is active), a symmetric buffer zone 
            [DW_loc - s, DW_loc + s] is strictly excluded from the coordinate list. 
            This protects the gapless edge modes from direct projection noise.

        top_region_last : bool
            If True (and self.DW is active), the coordinate list is reordered into two phases:
            1. Trivial Regions (Outside the Domain Walls): Measured FIRST.
            2. Bulk Region (Between the Domain Walls): Measured LAST.
            
            The relative order within these phases follows `sequence`. This ordering "pins" 
            the trivial vacuum boundaries before perturbing the topological bulk.

        skip_trivial : bool
            If True (and self.DW is active), strictly restricts the measurement scan to the 
            region between (and including) the domain walls. The trivial vacuum regions 
            (x < DW_1 and x > DW_2) are skipped entirely.

        Returns
        -------
        dict
            - "iter_fn": Callable returning the full coordinate list for one cycle.
            - "iter_bulk_fn": Callable returning ONLY the bulk coordinates (used for extra bulk_cycles).
            - "bulk_len": Integer count of bulk sites.
        """
        
        Nx = self.Nx if Nx is None else int(Nx)
        Ny = self.Ny if Ny is None else int(Ny)
        if sequence is None:
            sequence = "raster_y"
        if not isinstance(sequence, str):
            raise ValueError("sequence must be a string label.")
        mode = sequence.strip().lower()
        aliases = {
            "snake_y": "raster_y",
            "snake_x": "raster_x",
            "reverse_raster": "reverse_raster_y",
        }
        mode = aliases.get(mode, mode)

        allowed = (
            "raster_y",
            "reverse_raster_y",
            "raster_x",
            "random",
            "dw_symmetric_random",
        )
        if mode not in allowed:
            raise ValueError(f"sequence must be one of: {', '.join(allowed)}.")

        rng = np.random.default_rng() if rng is None else rng
        # Normalize dw_exclude: treat False as None, keep all other values
        if isinstance(dw_exclude, bool):
            dw_exclude = None if dw_exclude is False else int(dw_exclude)
        
        # 1. Generate Base Coordinates (The Pattern)
        coords = []
        
        if mode == "raster_y":
            coords = [(Rx, Ry) for Rx in range(Nx) for Ry in range(Ny)]
        elif mode == "reverse_raster_y":
            coords = [(Rx, Ry) for Rx in range(Nx) for Ry in range(Ny)]
        elif mode == "raster_x":
            coords = [(Rx, Ry) for Ry in range(Ny) for Rx in range(Nx)]
        elif mode == "random":
            coords = [(Rx, Ry) for Rx in range(Nx) for Ry in range(Ny)]
        elif mode == "dw_symmetric_random":
            if not (hasattr(self, "DW_loc") and len(self.DW_loc) >= 2):
                raise ValueError("dw_symmetric_random requires DW_loc to define the slab.")
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_start = int(dw_sorted[-1])
            x_order = list(range(x_start, -1, -1)) + list(range(Nx - 1, x_start, -1))
            for Rx in x_order:
                for Ry in range(Ny):
                    coords.append((Rx, Ry))

        # 2. Apply "Skip Trivial" Logic (Restrict to Bulk+Walls)
        if skip_trivial and getattr(self, "DW", False) and hasattr(self, "DW_loc") and len(self.DW_loc) >= 2:
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_min, x_max = dw_sorted[0], dw_sorted[-1]
            # Keep only sites inside or on the boundary
            coords = [c for c in coords if x_min <= c[0] <= x_max]

        # 3. Apply Exclusion Logic (Buffer Zone)
        if dw_exclude is not None and getattr(self, "DW", False) and hasattr(self, "DW_loc"):
            s = int(dw_exclude)
            if s < 0: raise ValueError("dw_exclude must be non-negative.")
            
            excluded_x = set()
            cutoff_occured = False

            dw_sorted = sorted(list(set(self.DW_loc)))
            if len(dw_sorted) >= 2:
                x0, x1 = dw_sorted[0], dw_sorted[-1]  # assume x1 > x0
                intervals = [
                    (x0 - s, x0),     # buffer on the left DW (inclusive)
                    (x1, x1 + s)      # buffer on the right DW (inclusive)
                ]
            else:
                # Fallback: symmetric buffer around the single DW position
                x0 = dw_sorted[0]
                intervals = [(x0 - s, x0 + s)]

            for start, end in intervals:
                if start < 0:
                    start = 0; cutoff_occured = True
                if end >= Nx:
                    end = Nx - 1; cutoff_occured = True
                for x_ex in range(start, end + 1):
                    excluded_x.add(x_ex)

            if cutoff_occured:
                print(f"[Warning] dw_exclude={s} clipped by lattice boundaries.")

            if excluded_x:
                excluded_list = sorted(excluded_x)
                print(f"[info] dw_exclude={s} excludes x-columns {excluded_list}")

            coords = [c for c in coords if c[0] not in excluded_x]

        # 4. Apply Region Reordering (Trivial First, Bulk Last)
        bulk_only_coords = [] 
        
        # Note: If skip_trivial=True, 'trivial_coords' below will naturally be empty or very sparse
        # because step 2 already filtered them out.
        if top_region_last and getattr(self, "DW", False) and hasattr(self, "DW_loc") and len(self.DW_loc) >= 2:
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_min, x_max = dw_sorted[0], dw_sorted[-1]

            trivial_coords = []
            bulk_coords = []

            for c in coords:
                Rx = c[0]
                # If strictly inside walls -> Bulk
                if x_min < Rx < x_max:
                    bulk_coords.append(c)
                else:
                    trivial_coords.append(c)
            
            coords = trivial_coords + bulk_coords
            bulk_only_coords = bulk_coords

        # 5. Define Iterators
        def _shuffle_copy(seq):
            seq_copy = list(seq)
            rng.shuffle(seq_copy)
            return seq_copy

        coords_for_len = coords
        
        if mode == "random":
            if top_region_last and getattr(self, "DW", False) and len(self.DW_loc) >= 2:
                # Robust split for random shuffle
                dw_sorted_iter = sorted(list(set(self.DW_loc)))
                x_min_i, x_max_i = dw_sorted_iter[0], dw_sorted_iter[-1]
                
                # Re-derive split from the final filtered coords
                static_trivial = [c for c in coords if not (x_min_i < c[0] < x_max_i)]
                static_bulk = [c for c in coords if (x_min_i < c[0] < x_max_i)]
                
                def _split_shuffle():
                    t = list(static_trivial); rng.shuffle(t)
                    b = list(static_bulk);    rng.shuffle(b)
                    return t + b
                
                iter_fn = _split_shuffle
                iter_bulk_fn = lambda: _shuffle_copy(static_bulk)
            else:
                iter_fn = lambda: _shuffle_copy(coords)
                iter_bulk_fn = lambda: []
        elif mode == "dw_symmetric_random":
            if not (hasattr(self, "DW_loc") and len(self.DW_loc) >= 2):
                raise ValueError("dw_symmetric_random requires DW_loc to define the slab.")
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_start = int(dw_sorted[-1])
            x_order = list(range(x_start, -1, -1)) + list(range(Nx - 1, x_start, -1))
            allowed_ry_by_x = {}
            for Rx, Ry in coords_for_len:
                allowed_ry_by_x.setdefault(Rx, []).append(Ry)
            x_order_filtered = [Rx for Rx in x_order if Rx in allowed_ry_by_x]

            def _dw_sym_rand_iter():
                coords_out = []
                for Rx in x_order_filtered:
                    y_order = rng.permutation(allowed_ry_by_x[Rx])
                    for Ry in y_order:
                        coords_out.append((Rx, Ry))
                return coords_out

            iter_fn = _dw_sym_rand_iter
            iter_bulk_fn = lambda: bulk_only_coords
        else:
            static_coords = (
                list(reversed(coords)) if mode == "reverse_raster_y" else coords
            )
            iter_fn = lambda: static_coords
            iter_bulk_fn = lambda: bulk_only_coords

        return {
            "mode": mode, 
            "coords_for_len": coords_for_len, 
            "iter_fn": iter_fn,
            "iter_bulk_fn": iter_bulk_fn,
            "bulk_len": len(bulk_only_coords)
        }

    # ==================== run_adaptive_circuit ====================

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
        store="none",
        init_mode="default",
        G_init=None,
        remember_init=True,
        save=True,
        save_suffix=None,
        sequence="raster_y",
        dw_exclude=None,
        top_region_last=False,
        bulk_cycles=0,
        skip_trivial=False,      # <--- New Parameter
    ):
        """
        Execute the adaptive circuit with optional history collection.

        skip_trivial : bool
            If True, measurements are restricted to the region between and including 
            the domain walls (x_min <= x <= x_max).

        """
        have_ow = all(hasattr(self, a) for a in ("WF_Ap","WF_Bp","WF_Am","WF_Bm"))
        if not have_ow:
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
            )

        # Validate cycles
        base_cycles = 5 if cycles is None else int(cycles)
        base_bulk_cycles = int(bulk_cycles)
        cycles = base_cycles
        bulk_cycles = base_bulk_cycles

        is_dw_active = getattr(self, "DW", False)
        use_bulk_extra = (top_region_last and is_dw_active and bulk_cycles > 0)
        dw_exclude_norm = self._normalize_dw_exclude(dw_exclude)
        exclude_arg = dw_exclude_norm if (is_dw_active and dw_exclude_norm is not None) else None

        seq_info = self._sequence_helper(
            sequence, 
            Nx=self.Nx, 
            Ny=self.Ny, 
            rng=np.random.default_rng(),
            dw_exclude=exclude_arg,
            top_region_last=(top_region_last and is_dw_active),
            skip_trivial=(skip_trivial and is_dw_active)
        )
        sequence_mode = seq_info["mode"]

        def _cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_mode, store_mode, alpha_1, alpha_2, seq, exclude, bulk_last, b_cycles, st):
            nsh = "None" if nshell is None else str(nshell)
            ex_str = str(exclude) if exclude is not None else "None"
            return (f"N{int(Nx)}x{int(Ny)}"
                    f"_C{int(cycles)}"
                    f"_S{int(samples)}"
                    f"_nsh{nsh}"
                    f"_DW{int(bool(DW))}"
                    f"_a1{alpha_1}"
                    f"_a2{alpha_2}"
                    f"_init-{init_mode}"
                    f"_store-{store_mode}"
                    f"_seq-{seq}"
                    f"_excl{ex_str}"
                    f"_bl{int(bulk_last)}"
                    f"_bc{int(b_cycles)}"
                    f"_st{int(st)}")

        def _save_histories(array, samples_count):
            if not (save and store != "none" and array is not None):
                return None
            outdir = self._g_history_outdir()
            key = _cache_key(Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                             nshell=self.nshell, DW=self.DW, init_mode=init_mode, store_mode=store,
                             alpha_1=self.alpha_1, alpha_2=self.alpha_2, seq=sequence_mode, 
                             exclude=exclude_arg, 
                             bulk_last=(top_region_last and is_dw_active),
                             b_cycles=bulk_cycles, st=(skip_trivial and is_dw_active))
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
            outdir = self._g_history_outdir_rel()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                nshell=self.nshell, DW=self.DW, init_mode=init_mode,
                store_mode=store, alpha_1=self.alpha_1, alpha_2=self.alpha_2,
                seq=sequence_mode, exclude=exclude_arg,
                bulk_last=(top_region_last and is_dw_active),
                b_cycles=bulk_cycles, st=(skip_trivial and is_dw_active),
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

        if not parallelize_samples or (samples is None or int(samples) <= 1):
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
            
            coords_for_len = seq_info["coords_for_len"]
            iter_fn = seq_info["iter_fn"]
            iter_bulk_fn = seq_info["iter_bulk_fn"]

            total_sites = (cycles * len(coords_for_len))
            if use_bulk_extra:
                total_sites += (bulk_cycles * seq_info["bulk_len"])

            pbar = tqdm(total=total_sites, desc="RAC (sites)", unit="site", leave=True) if progress else None

            _emit_save_notice(1)

            for _c in range(cycles):
                iter_coords = iter_fn()
                for Rx, Ry in iter_coords:
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

            if use_bulk_extra:
                for _c in range(bulk_cycles):
                    bulk_coords = iter_bulk_fn()
                    for Rx, Ry in bulk_coords:
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
            G_hist = np.expand_dims(np.stack(hist, axis=0), axis=0)
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

        samples = 1 if samples is None else int(samples)
        if samples <= 0: raise ValueError("samples must be a positive integer")
        S = samples
        n_jobs_eff = n_jobs
        if backend not in ("loky", "threading"): raise ValueError("backend error")

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
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                child.G0 = _make_G0()
                child.run_adaptive_circuit(
                    G_history=True, tol=tol, progress=False, 
                    cycles=cycles, 
                    postselect=postselect,
                    parallelize_samples=False, store="none", init_mode="default",
                    remember_init=remember_init, sequence=sequence_mode,
                    dw_exclude=exclude_arg,
                    top_region_last=(top_region_last and is_dw_active),
                    bulk_cycles=bulk_cycles,
                    skip_trivial=(skip_trivial and is_dw_active),
                )
                full_hist = [np.asarray(Gk) for Gk in child.G_list]
                if store == "full":
                    return np.stack(full_hist, axis=0)
                elif store == "top":
                    top_hist = [Gk[:Nlayer, :Nlayer] for Gk in full_hist]
                    return np.stack(top_hist, axis=0)
                else:
                    raise ValueError("When parallelizing samples, set store='top' or 'full'.")

        with self._joblib_tqdm_ctx(S, "samples", show_datetime=True):
            if backend == "loky":
                with parallel_backend("loky", n_jobs=n_jobs_eff, inner_max_num_threads=1):
                    with threadpool_limits(limits=1):
                        G_hist_list = Parallel(n_jobs=n_jobs_eff)(
                            delayed(_worker)(seeds[i]) for i in range(S)
                        )
            else:
                os.environ.setdefault("OMP_NUM_THREADS", "1")
                with threadpool_limits(limits=1):
                    G_hist_list = Parallel(n_jobs=n_jobs_eff, backend="threading")(
                        delayed(_worker)(seeds[i]) for i in range(S)
                    )

        G_hist = np.stack(G_hist_list, axis=0)
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

    @staticmethod
    def _checkpoint_array_signature(value):
        """Return a stable content signature for checkpoint compatibility checks."""
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.complexfloating):
            array = np.round(array.real, decimals=14) + 1j * np.round(
                array.imag, decimals=14
            )
        elif np.issubdtype(array.dtype, np.floating):
            array = np.round(array, decimals=14)
        array = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.view(np.uint8))
        return digest.hexdigest()

    def _normalize_measurement_site_ids(self, measurement_site_ids):
        """Validate an optional subset of unit-cell measurement centers."""
        if measurement_site_ids is None:
            return None
        array = np.asarray(measurement_site_ids)
        if array.ndim != 1:
            raise ValueError("measurement_site_ids must be a one-dimensional array.")
        if array.size == 0:
            raise ValueError("measurement_site_ids must not be empty.")
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
            array.dtype, np.bool_
        ):
            raise ValueError("measurement_site_ids must contain integer site IDs.")
        normalized = np.ascontiguousarray(array, dtype=np.int64)
        if np.any(normalized < 0) or np.any(normalized >= self.Nx * self.Ny):
            raise ValueError(
                "measurement_site_ids must lie in "
                f"0..{self.Nx * self.Ny - 1}."
            )
        if np.unique(normalized).size != normalized.size:
            raise ValueError("measurement_site_ids must contain no duplicates.")
        return normalized

    @staticmethod
    def _filter_sequence_info_to_sites(sequence_info, *, Nx, site_ids):
        """Filter a canonical sequence while preserving its ordering semantics."""
        if site_ids is None:
            return sequence_info
        selected = frozenset(int(value) for value in np.asarray(site_ids).tolist())
        canonical_ids = {
            int(x) + int(Nx) * int(y)
            for x, y in sequence_info["coords_for_len"]
        }
        missing = sorted(selected - canonical_ids)
        if missing:
            raise ValueError(
                "measurement_site_ids contains sites excluded by the canonical "
                f"geometry/schedule: {missing}."
            )

        def _keep(coords):
            return [
                (int(x), int(y))
                for x, y in coords
                if int(x) + int(Nx) * int(y) in selected
            ]

        iter_fn_base = sequence_info["iter_fn"]
        iter_bulk_fn_base = sequence_info["iter_bulk_fn"]
        filtered = dict(sequence_info)
        filtered["coords_for_len"] = _keep(sequence_info["coords_for_len"])
        filtered["iter_fn"] = lambda: _keep(iter_fn_base())
        filtered["iter_bulk_fn"] = lambda: _keep(iter_bulk_fn_base())
        filtered["bulk_len"] = len(filtered["iter_bulk_fn"]())
        return filtered

    def _markov_checkpoint_signature(
        self,
        *,
        sequence,
        active_top_layer_indices,
        meas_slab_only,
        dw_exclude,
        perfect_correction,
        postselect_probability,
        n_a,
        p_gain,
        p_loss,
        physical_covariance_update,
        state_representation,
        measurement_site_ids=None,
        controller_twist_schedule_sha256=None,
        controller_twist_gauge=None,
        frame_init_prepared=False,
    ):
        projector_names = ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm")
        projector_hashes = {
            name: self._checkpoint_array_signature(getattr(self, name))
            for name in projector_names
        }
        dw_loc = getattr(self, "DW_loc", None)
        signature = {
            "version": 2,
            "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_circuit",
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "nshell": self.nshell,
            "DW": bool(self.DW),
            "DW_loc": [] if dw_loc is None else [int(value) for value in dw_loc],
            "alpha_top": float(np.real(self.alpha_top)),
            "alpha_triv": float(np.real(self.alpha_triv)),
            "trial_orbitals": str(self.trial_orbitals),
            "dw_truncation": bool(self.dw_truncation),
            "twist_x": float(getattr(self, "twist_x", 0.0)),
            "twist_y": float(getattr(self, "twist_y", 0.0)),
            "sequence": str(sequence),
            "active_top_layer_indices": np.asarray(
                active_top_layer_indices, dtype=np.int64
            ).tolist(),
            "meas_slab_only": bool(meas_slab_only),
            "dw_exclude": dw_exclude,
            "channel_order": ["Ap", "Am", "Bp", "Bm"],
            "perfect_correction": bool(perfect_correction),
            "postselect_probability": float(postselect_probability),
            "n_a": float(n_a),
            "p_gain": float(p_gain),
            "p_loss": float(p_loss),
            "physical_covariance_update": str(physical_covariance_update),
            "state_representation": str(state_representation),
            "controller_twist_schedule_sha256": controller_twist_schedule_sha256,
            "controller_twist_gauge": controller_twist_gauge,
            "frame_init_prepared": bool(frame_init_prepared),
            "frame_algorithm_version": (
                FRAME_ALGORITHM_VERSION
                if state_representation in FRAME_REPRESENTATIONS
                else None
            ),
            "projector_hashes": projector_hashes,
        }
        # Preserve byte-for-byte compatibility with pre-selection checkpoints
        # whenever the new interface is not used.
        if measurement_site_ids is not None:
            signature["measurement_site_ids"] = np.asarray(
                measurement_site_ids, dtype=np.int64
            ).tolist()
        return signature

    def run_markov_circuit(
        self,
        G_history=True,
        progress=True,
        cycles=None,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=False,
        samples=None,
        n_jobs=None,
        backend="loky",
        parallelize_samples=False,
        init_mode="default",
        G_init=None,
        frame_init=None,
        initial_purity_tolerance=1e-9,
        save_final_G_only=False,
        save=True,
        save_init=True,
        save_history_stride=None,
        save_suffix=None,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        sequence="raster_y",
        meas_slab_only=True,
        dw_exclude=None,
        random_seed=None,
        throttle=None,
        max_in_flight=None,
        throttle_mem_frac=0.75,
        track_choi=False,
        choi_observer=None,
        choi_observer_cycles=None,
        choi_singular_tol=1e-10,
        choi_failure_mode="raise",
        physical_covariance_update="rank1",
        state_representation="auto",
        cycle_observer=None,
        native_cycle_observer=None,
        native_event_observer=None,
        return_native_state=False,
        require_no_covariance_materialization=False,
        frame_reorthonormalize_interval=1,
        timing_level="off",
        timing_observer=None,
        trajectory_weight_observer=None,
        trajectory_replay=None,
        site_schedule_replay=None,
        measurement_site_ids=None,
        controller_twist_schedule=None,
        controller_twist_gauge="uniform",
        frame_init_prepared=False,
        trajectory_replay_probability_tol=1e-14,
        checkpoint_state=None,
        checkpoint_observer=None,
        lyapunov_observer=None,
        lyapunov_nvec=None,
        lyapunov_frame_observer=None,
        lyapunov_initial_frame=None,
        lyapunov_basis_mode="canonical",
        lyapunov_start_cycle=1,
        lyapunov_full_space=False,
        lyapunov_track_restricted_core=False,
        lyapunov_track_record_fisher=False,
        lyapunov_singular_tol=1e-12,
        lyapunov_failure_mode="raise",
    ):
        """
        Execute the Markovian adaptive circuit.
        postselect: If True, apply deterministic target-outcome projections on the top layer
        at each visited site (no ancilla-swap feedback). Equivalent to
        postselect_probability=1.0.
        postselect_probability: Probability of applying the full post-selected
        site map at each visited site; otherwise the usual Born-rule Markov
        measurement/feedback map is applied.
        perfect_correction: If True, corrective ancilla swaps are deterministic
        (pump-out -> unoccupied, pump-in -> occupied), independent of n_a.
        p_gain/p_loss: Optional independent stochastic correction probabilities.
        If omitted, legacy n_a sets p_gain=n_a and p_loss=1-n_a.
        meas_slab_only: If True with DW=True and dw_truncation=True, project the
        exterior sequentially in the canonical unit-cell/orbital basis before
        cycle 0, then apply OW dynamics only to slab cells. Ordinary and partial
        postselection runs sample the exterior by the Born rule; full forced
        postselection selects occupied exterior orbitals.
        save_init: If True, include the initial state when saving history.
        save_history_stride: Optional integer stride for saving history cycles.
        throttle: If None, defaults to parallelize_samples; if True, throttle in-flight workers per batch.
        max_in_flight: Optional int cap on concurrent workers; if None, use auto (memory + CPU load).
        throttle_mem_frac: Fraction of available memory used to estimate auto cap.
        random_seed: Optional nonnegative integer. When provided, initialization,
        exterior preparation, schedules, and Born-rule draws use reproducible,
        independent per-sample random streams.
        cycle_observer: Optional callback invoked after initialization (cycle=0)
        and after every completed cycle with ``cycle``, ``G``, ``batch_index``,
        ``batch_start``, and ``batch_count`` keyword arguments.
        native_cycle_observer: Optional callback receiving the live native state at
        cycle 0 and after every completed cycle. Frame observers receive an
        ``OccupiedFrameState`` and must copy anything retained after the callback.
        native_event_observer: Optional serial diagnostic callback invoked after each
        realized OW channel with the live native state and branch labels.
        return_native_state: For a serial frame run with ``G_history=False``, return
        the final frame snapshot rather than lazily materializing a covariance.
        timing_level: ``off``, ``coarse``, or ``detailed`` update instrumentation.
        timing_observer: Optional callback invoked after every cycle with the raw
        timing accumulator and per-cycle counts.
        trajectory_weight_observer: Optional serial callback invoked once per
        visited site with the site branch log weight and cumulative log weight.
        trajectory_replay: Optional saved serial site record. When supplied,
        its site ordering and branch outcomes are replayed deterministically;
        probabilities are recomputed from the current covariance and projectors.
        site_schedule_replay: Optional integer array with shape
        ``(cycles, active_sites)``. Each row supplies the ordered site IDs for
        one cycle while Born-rule outcomes continue to be sampled normally.
        measurement_site_ids: Optional one-dimensional integer subset of the
        canonical unit-cell measurement centers. The ordinary schedule is
        filtered to this set without changing its ordering or update rules.
        controller_twist_schedule: Optional finite one-dimensional array of
        absolute y-boundary twists with length ``cycles + 1``. Entry zero is
        the cycle-zero basis; entry ``c`` is installed before the first update
        of physical cycle ``c`` by rebuilding the OW projectors.
        controller_twist_gauge: Gauge convention for a controller schedule.
        The canonical CPU implementation currently supports ``"uniform"``.
        frame_init_prepared: If True, ``frame_init`` is already the endpoint of
        hard-wall exterior preparation, so that preparation is not repeated.
        checkpoint_state: Optional state emitted by ``checkpoint_observer`` from
        an earlier serial one-sample run. ``cycles`` remains the target total
        cycle count, and initialization/exterior preparation are not repeated.
        checkpoint_observer: Optional callback invoked after every completed
        cycle with ``cycle`` and a deep-copied, restartable ``state`` mapping.
        lyapunov_observer: Optional serial callback invoked after each cycle
        with QR-stabilized tangent-cocycle spectra.
        lyapunov_frame_observer: Optional serial callback invoked after every
        propagated tangent cycle. In addition to the legacy spectra it receives
        phase-consistent QR factors and the requested restricted diagnostics.
        A callback that sets ``requires_physical_covariance = False`` receives
        ``G=None`` and the live occupied-frame object as ``native_state``; this
        avoids constructing a dense covariance solely for tangent bookkeeping.
        lyapunov_initial_frame: Optional orthonormal full-space (or active-space)
        tangent frame. A batched leading sample axis is also accepted.
        lyapunov_basis_mode: ``canonical`` for the ordinary tangent frame or
        ``pure_occupied_empty`` for exact occupied/empty one-leg blocks of a
        pure physical-frame trajectory.
        lyapunov_start_cycle: First physical cycle included in the tangent
        cocycle, enabling burn-in without restarting the physical trajectory.
        lyapunov_full_space: Track the full top-layer tangent space even when
        measurements are restricted to the domain-wall slab.
        """
        have_ow = all(hasattr(self, attr) for attr in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm"))
        if not have_ow and controller_twist_schedule is None:
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
            )

        state_representation_requested = self._normalize_state_representation(
            state_representation
        )
        initial_purity_tolerance = float(initial_purity_tolerance)
        if (
            not np.isfinite(initial_purity_tolerance)
            or initial_purity_tolerance <= 0.0
        ):
            raise ValueError("initial_purity_tolerance must be positive and finite.")
        if G_init is not None and frame_init is not None:
            raise ValueError("G_init and frame_init are mutually exclusive.")
        init_mode_norm = str(init_mode).strip().lower()
        if init_mode_norm not in ("default", "maxmix"):
            raise ValueError("init_mode must be 'default' or 'maxmix'.")
        init_mode = init_mode_norm
        lyapunov_basis_mode = str(lyapunov_basis_mode).strip().lower()
        if lyapunov_basis_mode not in ("canonical", "pure_occupied_empty"):
            raise ValueError(
                "lyapunov_basis_mode must be 'canonical' or "
                "'pure_occupied_empty'."
            )
        initial_purity_defects = None
        detected_initial_pure = None
        if frame_init is not None:
            detected_initial_pure = True
            resolution_reason = "explicit_frame_init"
        elif G_init is not None:
            initial_array = np.asarray(G_init, dtype=np.complex128)
            if initial_array.shape[-2:] == (self.Ntot, self.Ntot):
                initial_array = initial_array[..., : self.Ntot // 2, : self.Ntot // 2]
            elif initial_array.shape[-2:] != (self.Ntot // 2, self.Ntot // 2):
                raise ValueError(f"G_init shape error: {initial_array.shape}")
            initial_purity_defects = self._centered_purity_defect(initial_array)
            pure_mask = initial_purity_defects <= initial_purity_tolerance
            if np.any(pure_mask) and not np.all(pure_mask):
                raise ValueError(
                    "G_init contains a heterogeneous pure/mixed batch; split it into "
                    "separate run_markov_circuit calls."
                )
            detected_initial_pure = bool(np.all(pure_mask))
            resolution_reason = (
                "G_init_pure_within_tolerance"
                if detected_initial_pure
                else "G_init_mixed"
            )
        elif init_mode == "default":
            detected_initial_pure = True
            resolution_reason = "default_random_pure_initialization"
        else:
            detected_initial_pure = False
            resolution_reason = "maximally_mixed_initialization"

        if state_representation_requested == "auto":
            state_representation_norm = (
                "physical_frame" if detected_initial_pure else "covariance"
            )
        else:
            state_representation_norm = state_representation_requested
        if state_representation_norm == "physical_frame" and not detected_initial_pure:
            defect_text = (
                "unknown"
                if initial_purity_defects is None
                else f"{float(np.max(initial_purity_defects)):.3e}"
            )
            raise ValueError(
                "state_representation='physical_frame' requires a pure initial "
                f"state; maximum occupation defect is {defect_text}."
            )
        explicit_covariance_override = bool(
            state_representation_requested == "covariance" and detected_initial_pure
        )
        frame_native = state_representation_norm in FRAME_REPRESENTATIONS
        if (
            lyapunov_basis_mode == "pure_occupied_empty"
            and state_representation_norm != "physical_frame"
        ):
            raise ValueError(
                "lyapunov_basis_mode='pure_occupied_empty' requires a pure "
                "physical-frame trajectory."
            )
        require_no_covariance_materialization = bool(
            require_no_covariance_materialization
        )
        frame_reorthonormalize_interval = int(frame_reorthonormalize_interval)
        if frame_reorthonormalize_interval <= 0:
            raise ValueError("frame_reorthonormalize_interval must be positive.")
        covariance_materializations = []
        timing_level_norm = self._normalize_timing_level(timing_level)
        return_native_state = bool(return_native_state)
        if return_native_state and not frame_native:
            raise ValueError(
                "return_native_state=True is meaningful only for a frame representation."
            )
        if return_native_state and G_history:
            raise ValueError(
                "return_native_state=True requires G_history=False; use native_cycle_observer "
                "for frame histories."
            )
        if return_native_state and save:
            raise ValueError(
                "return_native_state=True requires save=False because the legacy saver stores covariances."
            )

        track_choi = bool(track_choi)
        if choi_observer is not None and not track_choi:
            raise ValueError("choi_observer requires track_choi=True.")
        if choi_observer_cycles is not None and choi_observer is None:
            raise ValueError("choi_observer_cycles requires choi_observer.")
        choi_singular_tol = float(choi_singular_tol)
        if not np.isfinite(choi_singular_tol) or choi_singular_tol <= 0.0:
            raise ValueError("choi_singular_tol must be a positive finite scalar.")
        choi_failure_mode = str(choi_failure_mode).strip().lower()
        if choi_failure_mode not in ("raise", "censor"):
            raise ValueError("choi_failure_mode must be either 'raise' or 'censor'.")
        physical_covariance_update_norm = self._normalize_physical_covariance_update(
            physical_covariance_update
        )
        self._physical_covariance_update_mode = physical_covariance_update_norm
        p_gain_requested = p_gain
        p_loss_requested = p_loss
        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        if random_seed is not None:
            original_seed = random_seed
            if isinstance(original_seed, (bool, np.bool_)):
                raise ValueError("random_seed must be a nonnegative integer or None.")
            try:
                random_seed = int(original_seed)
            except Exception as exc:
                raise ValueError("random_seed must be a nonnegative integer or None.") from exc
            if isinstance(original_seed, (float, np.floating)) and not float(original_seed).is_integer():
                raise ValueError("random_seed must be a nonnegative integer or None.")
            if random_seed < 0:
                raise ValueError("random_seed must be a nonnegative integer or None.")
        feedback_key_suffix = (
            ""
            if p_gain_requested is None and p_loss_requested is None
            else f"_pg{p_gain_eff:g}_pl{p_loss_eff:g}"
        )

        cycles = 5 if cycles is None else int(cycles)
        if cycles < 0:
            raise ValueError("cycles must be a nonnegative integer.")
        frame_init_prepared = bool(frame_init_prepared)
        if frame_init_prepared and frame_init is None:
            raise ValueError("frame_init_prepared=True requires frame_init.")
        controller_twist_gauge_norm = str(controller_twist_gauge).strip().lower()
        controller_twist_schedule_array = None
        controller_twist_schedule_hash = None
        if controller_twist_schedule is not None:
            if controller_twist_gauge_norm != "uniform":
                raise ValueError(
                    "controller_twist_gauge must be 'uniform' when a twist schedule is supplied."
                )
            controller_twist_schedule_array = np.asarray(
                controller_twist_schedule, dtype=np.float64
            )
            if controller_twist_schedule_array.ndim != 1:
                raise ValueError("controller_twist_schedule must be one-dimensional.")
            if controller_twist_schedule_array.shape != (cycles + 1,):
                raise ValueError(
                    "controller_twist_schedule must contain exactly cycles + 1 "
                    f"absolute twists; got {controller_twist_schedule_array.size} "
                    f"values for cycles={cycles}."
                )
            if not np.all(np.isfinite(controller_twist_schedule_array)):
                raise ValueError("controller_twist_schedule must contain only finite values.")
            controller_twist_schedule_array = np.ascontiguousarray(
                controller_twist_schedule_array, dtype=np.float64
            )
            controller_twist_schedule_hash = self._checkpoint_array_signature(
                controller_twist_schedule_array
            )
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
                twist_y=float(controller_twist_schedule_array[0]),
            )
        lyapunov_enabled = (
            lyapunov_observer is not None or lyapunov_frame_observer is not None
        )
        lyapunov_start_cycle = int(lyapunov_start_cycle)
        if lyapunov_enabled and not (1 <= lyapunov_start_cycle <= cycles):
            raise ValueError(
                "lyapunov_start_cycle is the first propagated cycle and must lie "
                f"in 1..{cycles}; got {lyapunov_start_cycle}."
            )
        lyapunov_singular_tol = float(lyapunov_singular_tol)
        if not np.isfinite(lyapunov_singular_tol) or lyapunov_singular_tol <= 0.0:
            raise ValueError("lyapunov_singular_tol must be a positive finite scalar.")
        lyapunov_failure_mode = str(lyapunov_failure_mode).strip().lower()
        if lyapunov_failure_mode not in ("raise", "censor"):
            raise ValueError(
                "lyapunov_failure_mode must be either 'raise' or 'censor'."
            )
        lyapunov_track_restricted_core = bool(
            lyapunov_track_restricted_core or lyapunov_track_record_fisher
        )
        lyapunov_track_record_fisher = bool(lyapunov_track_record_fisher)
        if lyapunov_basis_mode == "pure_occupied_empty":
            if lyapunov_initial_frame is not None or lyapunov_nvec is not None:
                raise ValueError(
                    "pure_occupied_empty constructs its complete cycle-zero basis; "
                    "do not pass lyapunov_initial_frame or lyapunov_nvec."
                )
            if lyapunov_track_record_fisher:
                raise ValueError(
                    "pure_occupied_empty is incompatible with record-Fisher tracking."
                )
            lyapunov_track_restricted_core = True
        if (
            lyapunov_initial_frame is not None
            or lyapunov_basis_mode != "canonical"
            or lyapunov_track_restricted_core
            or lyapunov_track_record_fisher
        ) and not lyapunov_enabled:
            raise ValueError(
                "Custom tangent frames and restricted diagnostics require a "
                "lyapunov_observer or lyapunov_frame_observer."
            )
        choi_observer_cycles_norm = self._normalize_choi_observer_cycles(choi_observer_cycles, cycles)
        if choi_observer is not None and choi_observer_cycles_norm is None:
            choi_observer_cycles_norm = [cycles]
        postselect_probability = float(postselect_probability)
        if not np.isfinite(postselect_probability):
            raise ValueError("postselect_probability must be finite.")
        if postselect_probability < 0.0 or postselect_probability > 1.0:
            raise ValueError("postselect_probability must satisfy 0 <= p <= 1.")
        if bool(postselect):
            if postselect_probability not in (0.0, 1.0):
                raise ValueError(
                    "postselect=True is equivalent to postselect_probability=1.0; "
                    "do not pass an intermediate postselect_probability with postselect=True."
                )
            postselect_probability = 1.0
        effective_postselect = postselect_probability == 1.0
        trajectory_replay_probability_tol = float(trajectory_replay_probability_tol)
        if (
            not np.isfinite(trajectory_replay_probability_tol)
            or trajectory_replay_probability_tol < 0.0
        ):
            raise ValueError(
                "trajectory_replay_probability_tol must be a finite nonnegative scalar."
            )
        trajectory_replay_entries = None
        replay_by_cycle = None
        site_schedule_replay_array = None
        site_schedule_replay_hash = None
        measurement_site_ids_array = self._normalize_measurement_site_ids(
            measurement_site_ids
        )
        measurement_site_ids_hash = (
            None
            if measurement_site_ids_array is None
            else self._checkpoint_array_signature(measurement_site_ids_array)
        )
        if site_schedule_replay is not None:
            schedule_array = np.asarray(site_schedule_replay)
            if schedule_array.ndim != 2:
                raise ValueError(
                    "site_schedule_replay must be a two-dimensional integer array."
                )
            if not np.issubdtype(schedule_array.dtype, np.integer) or np.issubdtype(
                schedule_array.dtype, np.bool_
            ):
                raise ValueError("site_schedule_replay must contain integer site IDs.")
            if schedule_array.shape[0] != cycles:
                raise ValueError(
                    "site_schedule_replay must contain exactly one row per cycle; "
                    f"got {schedule_array.shape[0]} rows for cycles={cycles}."
                )
            site_schedule_replay_array = np.ascontiguousarray(
                schedule_array, dtype=np.int64
            )
            site_schedule_replay_hash = self._checkpoint_array_signature(
                site_schedule_replay_array
            )
        if trajectory_replay is not None:
            if site_schedule_replay_array is not None:
                raise ValueError(
                    "site_schedule_replay cannot be combined with trajectory_replay."
                )
            if effective_postselect or postselect_probability != 0.0:
                raise ValueError(
                    "trajectory_replay currently supports Born measurement/feedback runs "
                    "with postselection disabled."
                )
            trajectory_replay_entries = tuple(dict(entry) for entry in trajectory_replay)
            replay_by_cycle = {cycle: [] for cycle in range(1, cycles + 1)}
            for position, entry in enumerate(trajectory_replay_entries):
                cycle = int(entry.get("cycle", -1))
                if cycle not in replay_by_cycle:
                    raise ValueError(
                        f"Trajectory replay entry {position} has cycle={cycle}, outside 1..{cycles}."
                    )
                if "site_id" not in entry or "branch_events" not in entry:
                    raise ValueError(
                        "Each trajectory replay entry requires cycle, site_id, and branch_events."
                    )
                replay_by_cycle[cycle].append(entry)
            if any(not entries for entries in replay_by_cycle.values()):
                missing = [cycle for cycle, entries in replay_by_cycle.items() if not entries]
                raise ValueError(f"Trajectory replay is missing complete cycles: {missing}.")
        
        is_dw_active = getattr(self, "DW", False)
        meas_slab_only_requested = bool(meas_slab_only)
        meas_slab_only_effective = self._meas_slab_only_effective(meas_slab_only_requested)
        active_top_layer_indices = self.active_top_layer_indices(meas_slab_only=meas_slab_only_requested)
        lyapunov_full_space = bool(lyapunov_full_space)
        cycles_eff = cycles
        dw_exclude_norm = self._normalize_dw_exclude(dw_exclude)
        exclude_arg = dw_exclude_norm if (is_dw_active and dw_exclude_norm is not None) else None
        
        seq_info = self._sequence_helper(
            sequence, 
            Nx=self.Nx, 
            Ny=self.Ny, 
            rng=np.random.default_rng(),
            dw_exclude=exclude_arg,
            skip_trivial=meas_slab_only_effective,
        )
        seq_info = self._filter_sequence_info_to_sites(
            seq_info,
            Nx=self.Nx,
            site_ids=measurement_site_ids_array,
        )
        sequence_mode = seq_info["mode"]
        site_schedule_by_cycle = None
        if site_schedule_replay_array is not None:
            expected_site_ids = np.asarray(
                sorted(
                    int(x) + int(self.Nx) * int(y)
                    for x, y in seq_info["coords_for_len"]
                ),
                dtype=np.int64,
            )
            if site_schedule_replay_array.shape[1] != expected_site_ids.size:
                raise ValueError(
                    "site_schedule_replay has the wrong active-site count: "
                    f"expected {expected_site_ids.size}, got "
                    f"{site_schedule_replay_array.shape[1]}."
                )
            for row_index, row in enumerate(site_schedule_replay_array):
                if not np.array_equal(np.sort(row), expected_site_ids):
                    raise ValueError(
                        "Each site_schedule_replay row must visit every active site "
                        f"exactly once; cycle {row_index + 1} is invalid."
                    )
            site_schedule_by_cycle = {
                cycle: site_schedule_replay_array[cycle - 1]
                for cycle in range(1, cycles + 1)
            }
        sample_seed_values = None
        checkpoint_states = []
        checkpoint_signature = self._markov_checkpoint_signature(
            sequence=sequence_mode,
            active_top_layer_indices=active_top_layer_indices,
            meas_slab_only=meas_slab_only_effective,
            dw_exclude=exclude_arg,
            perfect_correction=perfect_correction,
            postselect_probability=postselect_probability,
            n_a=n_a,
            p_gain=p_gain_eff,
            p_loss=p_loss_eff,
            physical_covariance_update=self._physical_covariance_update_label(
                physical_covariance_update_norm
            ),
            state_representation=state_representation_norm,
            measurement_site_ids=measurement_site_ids_array,
            controller_twist_schedule_sha256=controller_twist_schedule_hash,
            controller_twist_gauge=(
                controller_twist_gauge_norm
                if controller_twist_schedule_array is not None
                else None
            ),
            frame_init_prepared=frame_init_prepared,
        )
        resume_checkpoint = None
        if checkpoint_state is not None:
            if trajectory_replay_entries is not None:
                raise ValueError("checkpoint_state cannot be combined with trajectory_replay.")
            if site_schedule_replay_array is not None:
                raise ValueError(
                    "checkpoint_state cannot be combined with site_schedule_replay."
                )
            if track_choi or lyapunov_enabled:
                raise ValueError(
                    "checkpoint resume currently supports the physical trajectory only; "
                    "Choi and Lyapunov observer states are not restartable."
                )
            resume_checkpoint = copy.deepcopy(dict(checkpoint_state))
            expected_checkpoint_version = 2 if frame_native else 1
            if int(resume_checkpoint.get("version", -1)) != expected_checkpoint_version:
                raise ValueError("Unsupported or missing Markov checkpoint version.")
            saved_signature = copy.deepcopy(resume_checkpoint.get("signature"))
            if isinstance(saved_signature, dict) and "twist_x" not in saved_signature:
                # Checkpoints written before two-dimensional twist support are
                # unambiguously zero-flux in x.
                saved_signature["twist_x"] = 0.0
            if saved_signature != checkpoint_signature:
                raise ValueError(
                    "Checkpoint signature does not match the current geometry, projectors, "
                    "schedule, channel order, or protocol."
                )
            completed_cycles = int(resume_checkpoint.get("completed_cycles", -1))
            if not (0 <= completed_cycles < cycles_eff):
                raise ValueError(
                    "Checkpoint completed_cycles must lie in 0..cycles-1 when cycles is "
                    f"the target total; got completed_cycles={completed_cycles}, cycles={cycles_eff}."
                )
            if frame_native:
                native_state = resume_checkpoint.get("native_state")
                if not isinstance(native_state, dict):
                    raise ValueError("Frame checkpoint is missing native_state.")
                checkpoint_frame = np.asarray(
                    native_state.get("frame"), dtype=np.complex128
                )
                if (
                    checkpoint_frame.ndim != 2
                    or checkpoint_frame.shape[0] != self.Ntot // 2
                ):
                    raise ValueError("Frame checkpoint has an invalid frame shape.")
            else:
                checkpoint_G = np.asarray(
                    resume_checkpoint.get("G"), dtype=np.complex128
                )
                expected_shape = (self.Ntot // 2, self.Ntot // 2)
                if checkpoint_G.shape != expected_shape:
                    raise ValueError(
                        f"Checkpoint covariance must have shape {expected_shape}; got {checkpoint_G.shape}."
                    )
            required_rngs = {"initialization", "exterior", "schedule", "dynamics"}
            rng_states = resume_checkpoint.get("rng_states")
            if not isinstance(rng_states, dict) or set(rng_states) != required_rngs:
                raise ValueError(
                    "Checkpoint rng_states must contain initialization, exterior, schedule, "
                    "and dynamics states."
                )
            saved_root_seed = resume_checkpoint.get("random_seed")
            if saved_root_seed is None:
                raise ValueError("Checkpoint is missing its explicit random_seed.")
            if random_seed is None:
                random_seed = int(saved_root_seed)
            elif int(random_seed) != int(saved_root_seed):
                raise ValueError(
                    "Checkpoint random_seed does not match the requested random_seed."
                )
        if checkpoint_observer is not None and random_seed is None:
            raise ValueError(
                "checkpoint_observer requires an explicit nonnegative random_seed."
            )

        def _cache_key(*, Nx, Ny, cycles, samples, nshell, DW, init_mode, n_a, seq, store_mode, exclude, ps, psp, pc, mslab, feedback_suffix):
            nsh = "None" if nshell is None else str(nshell)
            ex_str = str(exclude) if exclude is not None else "None"
            key = (
                f"N{int(Nx)}x{int(Ny)}"
                f"_C{int(cycles)}"
                f"_S{int(samples)}"
                f"_nsh{nsh}"
                f"_DW{int(bool(DW))}"
                f"_alpha_top{self.alpha_top}"
                f"_alpha_triv{self.alpha_triv}"
                f"_trial-{self.trial_orbitals}"
                f"_dwtrunc{int(bool(self.dw_truncation))}"
                f"_init-{init_mode}"
                f"_n_a{n_a}"
                f"{feedback_suffix}"
                f"_seq-{seq}"
                f"_excl{ex_str}"
                f"_ps{int(ps)}"
                f"_psp{float(psp):g}"
                f"_pc{int(pc)}"
                f"_mslab{int(bool(mslab))}"
                f"_repr-{state_representation_norm}"
                "_engine-frame-v2"
                "_markov_circuit"
            )
            if store_mode == "final":
                key += "_final"
            if site_schedule_replay_hash is not None:
                key += f"_schedule-{site_schedule_replay_hash[:12]}"
            if measurement_site_ids_hash is not None:
                key += f"_sites-{measurement_site_ids_hash[:12]}"
            return key

        def _run_config(samples_count, store_mode):
            return {
                "Nx": int(self.Nx),
                "Ny": int(self.Ny),
                "cycles": int(cycles),
                "samples": int(samples_count),
                "nshell": self.nshell,
                "DW": bool(self.DW),
                "alpha_top": float(np.real(self.alpha_top)),
                "alpha_triv": float(np.real(self.alpha_triv)),
                "trial_orbitals": self.trial_orbitals,
                "dw_truncation": bool(self.dw_truncation),
                "store_mode": store_mode,
                "init_mode": init_mode,
                "n_a": float(n_a),
                "p_gain": None if p_gain_requested is None else float(p_gain_requested),
                "p_loss": None if p_loss_requested is None else float(p_loss_requested),
                "p_gain_effective": float(p_gain_eff),
                "p_loss_effective": float(p_loss_eff),
                "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_circuit",
                "sequence": sequence_mode,
                "dw_exclude": exclude_arg,
                "postselect": bool(effective_postselect),
                "postselect_probability": float(postselect_probability),
                "perfect_correction": bool(perfect_correction),
                "meas_slab_only_requested": meas_slab_only_requested,
                "meas_slab_only_effective": meas_slab_only_effective,
                "physical_covariance_update": self._physical_covariance_update_label(
                    physical_covariance_update_norm
                ),
                "state_representation_requested": state_representation_requested,
                "state_representation": state_representation_norm,
                "state_representation_resolved": state_representation_norm,
                "state_representation_resolution_reason": resolution_reason,
                "initial_purity_tolerance": float(initial_purity_tolerance),
                "initial_purity_defects": (
                    None
                    if initial_purity_defects is None
                    else initial_purity_defects.tolist()
                ),
                "explicit_covariance_override": explicit_covariance_override,
                "frame_algorithm_version": (
                    FRAME_ALGORITHM_VERSION if frame_native else None
                ),
                "require_no_covariance_materialization": bool(
                    require_no_covariance_materialization
                ),
                "native_cycle_observer": native_cycle_observer is not None,
                "native_event_observer": native_event_observer is not None,
                "return_native_state": bool(return_native_state),
                "timing_level": timing_level_norm,
                "timing_observer": timing_observer is not None,
                "frame_covariance_materialized_for_cycle_observer": bool(
                    frame_native and cycle_observer is not None
                ),
                "active_top_layer_indices": active_top_layer_indices.tolist(),
                "track_choi": bool(track_choi),
                "choi_observer_cycles": choi_observer_cycles_norm,
                "choi_singular_tol": float(choi_singular_tol),
                "choi_failure_mode": choi_failure_mode,
                "choi_initialization": "Sigma_LL=0,Sigma_LR=I,Sigma_RR=0",
                "choi_formula": "regularized_resolvent_rank_one_v2",
                "choi_basis": "reduced_topological_slab" if meas_slab_only_effective else "full_top_layer",
                "trajectory_weight_observer": trajectory_weight_observer is not None,
                "trajectory_replay": trajectory_replay_entries is not None,
                "trajectory_replay_sites": (
                    0 if trajectory_replay_entries is None else len(trajectory_replay_entries)
                ),
                "trajectory_replay_probability_tol": float(
                    trajectory_replay_probability_tol
                ),
                "site_schedule_replay": site_schedule_replay_array is not None,
                "site_schedule_replay_shape": (
                    None
                    if site_schedule_replay_array is None
                    else list(site_schedule_replay_array.shape)
                ),
                "site_schedule_replay_sha256": site_schedule_replay_hash,
                "measurement_site_ids": (
                    None
                    if measurement_site_ids_array is None
                    else measurement_site_ids_array.tolist()
                ),
                "measurement_site_ids_sha256": measurement_site_ids_hash,
                "controller_twist_schedule": (
                    None
                    if controller_twist_schedule_array is None
                    else controller_twist_schedule_array.tolist()
                ),
                "controller_twist_schedule_sha256": controller_twist_schedule_hash,
                "controller_twist_gauge": (
                    controller_twist_gauge_norm
                    if controller_twist_schedule_array is not None
                    else None
                ),
                "frame_init_prepared": bool(frame_init_prepared),
                "checkpoint_resume": resume_checkpoint is not None,
                "checkpoint_completed_cycles": (
                    None
                    if resume_checkpoint is None
                    else int(resume_checkpoint["completed_cycles"])
                ),
                "checkpoint_signature": checkpoint_signature,
                "lyapunov_observer": lyapunov_observer is not None,
                "lyapunov_frame_observer": lyapunov_frame_observer is not None,
                "lyapunov_nvec": None if lyapunov_nvec is None else int(lyapunov_nvec),
                "lyapunov_initial_frame_shape": (
                    None
                    if lyapunov_initial_frame is None
                    else list(np.asarray(lyapunov_initial_frame).shape)
                ),
                "lyapunov_start_cycle": int(lyapunov_start_cycle),
                "lyapunov_full_space": bool(lyapunov_full_space),
                "lyapunov_observation_cycles": int(
                    cycles_eff - lyapunov_start_cycle + 1
                ),
                "lyapunov_track_restricted_core": bool(
                    lyapunov_track_restricted_core
                ),
                "lyapunov_track_record_fisher": bool(
                    lyapunov_track_record_fisher
                ),
                "lyapunov_singular_tol": float(lyapunov_singular_tol),
                "lyapunov_failure_mode": lyapunov_failure_mode,
                "lyapunov_formula": (
                    "fixed_record_rank_one_measurement_plus_explicit_reset_v1"
                ),
                "random_seed": random_seed,
                "twist_x": float(getattr(self, "twist_x", 0.0)),
                "twist_y": float(getattr(self, "twist_y", 0.0)),
                "sample_seeds": sample_seed_values,
                "rng_streams": ["initialization", "exterior", "schedule", "dynamics"],
                "exterior_preparation": (
                    "skipped_prepared_frame"
                    if meas_slab_only_effective and frame_init_prepared
                    else
                    "forced_occupied_onsite_before_cycle_0"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned_onsite_before_cycle_0"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_mode": (
                    "already_prepared"
                    if meas_slab_only_effective and frame_init_prepared
                    else
                    "forced_occupied"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_basis": (
                    "canonical_unit_cell_orbital"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_orbital_count": (
                    2 * len(self._exterior_site_coordinates())
                    if meas_slab_only_effective
                    else 0
                ),
            }

        def _save_histories(array, samples_count):
            if not (save and G_history and array is not None):
                return None
            outdir = self._g_history_outdir()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                nshell=self.nshell, DW=self.DW, init_mode=init_mode,
                n_a=n_a, seq=sequence_mode, store_mode="history", 
                exclude=exclude_arg,
                ps=effective_postselect,
                psp=postselect_probability,
                pc=perfect_correction,
                mslab=meas_slab_only_effective,
                feedback_suffix=feedback_key_suffix,
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            np.savez_compressed(
                path,
                G_hist=np.asarray(array, dtype=np.complex128),
                run_config=np.asarray(json.dumps(_run_config(samples_count, "history"), sort_keys=True)),
            )
            return path

        def _save_finals(array, samples_count):
            if not (save and (not G_history) and array is not None):
                return None
            outdir = self._g_history_outdir()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                nshell=self.nshell, DW=self.DW, init_mode=init_mode,
                n_a=n_a, seq=sequence_mode, store_mode="final", 
                exclude=exclude_arg,
                ps=effective_postselect,
                psp=postselect_probability,
                pc=perfect_correction,
                mslab=meas_slab_only_effective,
                feedback_suffix=feedback_key_suffix,
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            np.savez_compressed(
                path,
                G_final=np.asarray(array, dtype=np.complex128),
                run_config=np.asarray(json.dumps(_run_config(samples_count, "final"), sort_keys=True)),
            )
            return path

        def _expected_save_path(samples_count):
            if not save:
                return None
            outdir = self._g_history_outdir_rel()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, cycles=cycles, samples=samples_count,
                nshell=self.nshell, DW=self.DW, init_mode=init_mode,
                n_a=n_a, seq=sequence_mode,
                store_mode="history" if G_history else "final",
                exclude=exclude_arg,
                ps=effective_postselect,
                psp=postselect_probability,
                pc=perfect_correction,
                mslab=meas_slab_only_effective,
                feedback_suffix=feedback_key_suffix,
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
            label = "history" if G_history else "final state(s)"
            print(f"[info] Markov circuit will save {label} to {path}")
            time.sleep(2)

        Nlayer = self.Ntot // 2

        def _materialize_frame(state, *, reason, cycle, sample_index):
            if not isinstance(state, OccupiedFrameState):
                return state
            if require_no_covariance_materialization:
                raise RuntimeError(
                    "require_no_covariance_materialization=True forbids the requested "
                    f"covariance reconstruction ({reason}, cycle={cycle}, "
                    f"sample={sample_index})."
                )
            covariance_materializations.append(
                {
                    "reason": str(reason),
                    "cycle": int(cycle),
                    "sample_index": int(sample_index),
                }
            )
            return state.centered_covariance(reason=str(reason))

        def _emit_native_cycle_observer(
            *, cycle, state, batch_index, batch_start, batch_count
        ):
            if native_cycle_observer is None:
                return
            before = int(getattr(state, "materialization_count", 0))
            native_cycle_observer(
                cycle=int(cycle),
                state=state,
                batch_index=int(batch_index),
                batch_start=int(batch_start),
                batch_count=int(batch_count),
            )
            after = int(getattr(state, "materialization_count", 0))
            if after > before:
                reasons = list(getattr(state, "materialization_reasons", []))[before:after]
                for reason in reasons:
                    covariance_materializations.append(
                        {
                            "reason": str(reason),
                            "cycle": int(cycle),
                            "sample_index": int(batch_start),
                            "requested_by": "native_cycle_observer",
                        }
                    )
                if require_no_covariance_materialization:
                    raise RuntimeError(
                        "require_no_covariance_materialization=True was violated by "
                        f"native_cycle_observer at cycle={cycle}, sample={batch_start}."
                    )

        def _select_sample_initial(value, sample_index):
            if isinstance(value, dict) and "frame" in value:
                return value
            array = np.asarray(value)
            if array.ndim == 3:
                if sample_index >= array.shape[0]:
                    raise ValueError(
                        "Batched initial state does not contain enough samples."
                    )
                return array[sample_index]
            return value

        def _prepare_initial_top(instance, rng=None, sample_index=0):
            if frame_init is not None:
                selected = _select_sample_initial(frame_init, sample_index)
                if isinstance(selected, dict):
                    frame_array = np.asarray(selected["frame"], dtype=np.complex128)
                else:
                    frame_array = np.asarray(selected, dtype=np.complex128)
                if frame_array.ndim != 2 or frame_array.shape[0] != Nlayer:
                    raise ValueError(
                        f"frame_init must have trailing shape ({Nlayer}, rank); "
                        f"got {frame_array.shape}."
                    )
                return OccupiedFrameState(
                    frame_array,
                    representation="physical_frame",
                    physical_dimension=Nlayer,
                    timing=instance._update_timing_collector,
                    zero_tolerance=trajectory_replay_probability_tol,
                )
            if G_init is not None:
                arr = np.asarray(
                    _select_sample_initial(G_init, sample_index),
                    dtype=np.complex128,
                )
                if arr.shape == (instance.Ntot, instance.Ntot):
                    arr = arr[:Nlayer, :Nlayer]
                elif arr.shape != (Nlayer, Nlayer):
                    raise ValueError(f"G_init shape error: {arr.shape}")
                centered = 0.5 * (arr + arr.conj().T)
                if state_representation_norm == "physical_frame":
                    return OccupiedFrameState.from_centered_covariance(
                        centered,
                        representation="physical_frame",
                        timing=instance._update_timing_collector,
                        purity_tolerance=initial_purity_tolerance,
                        zero_tolerance=trajectory_replay_probability_tol,
                    )
                return centered
            if init_mode == "default":
                if state_representation_norm == "physical_frame":
                    rank = int(round(instance.filling_frac * Nlayer))
                    return OccupiedFrameState.random_pure(
                        Nlayer,
                        rank,
                        rng=rng,
                        timing=instance._update_timing_collector,
                        zero_tolerance=trajectory_replay_probability_tol,
                    )
                return instance.random_complex_fermion_covariance(N=Nlayer, rng=rng)
            if init_mode == "maxmix":
                return np.zeros((Nlayer, Nlayer), dtype=np.complex128)
            raise ValueError("init_mode must be 'default' or 'maxmix'.")

        store_full_history = G_history
        if save_final_G_only and G_history:
            print("[warn] save_final_G_only is ignored; full history is stored because G_history=True.")

        history_stride_outer = save_history_stride

        choi_diagnostics = [] if track_choi else None
        lyapunov_diagnostics = [] if lyapunov_enabled else None

        def _run_single(
            instance,
            enable_progress,
            return_eta=False,
            sample_idx=None,
            total_samples=None,
            sample_seed=None,
        ):
            update_timing = UpdateTimingCollector(timing_level_norm)
            instance._update_timing_collector = update_timing
            local_resume = resume_checkpoint
            if local_resume is not None:
                initial_rng = np.random.default_rng()
                exterior_rng = np.random.default_rng()
                schedule_rng = np.random.default_rng()
                dynamics_rng = np.random.default_rng()
                rng_objects = {
                    "initialization": initial_rng,
                    "exterior": exterior_rng,
                    "schedule": schedule_rng,
                    "dynamics": dynamics_rng,
                }
                for name, generator in rng_objects.items():
                    generator.bit_generator.state = copy.deepcopy(
                        local_resume["rng_states"][name]
                    )
                with update_timing.measure("state_copy_for_replay"):
                    if frame_native:
                        native_state = local_resume["native_state"]
                        G_top = OccupiedFrameState(
                            np.asarray(native_state["frame"], dtype=np.complex128),
                            representation=str(native_state["representation"]),
                            physical_dimension=int(native_state["physical_dimension"]),
                            timing=update_timing,
                            zero_tolerance=trajectory_replay_probability_tol,
                        )
                        G_top.log_weight = float(native_state.get("log_weight", 0.0))
                        G_top.min_rank = int(native_state.get("min_rank", G_top.rank))
                        G_top.max_rank = int(native_state.get("max_rank", G_top.rank))
                    else:
                        G_top = np.array(
                            local_resume["G"], dtype=np.complex128, copy=True
                        )
                start_cycle = int(local_resume["completed_cycles"]) + 1
                cumulative_log_weight = float(
                    local_resume.get("cumulative_log_weight", 0.0)
                )
                saved_sample_seed = local_resume.get("sample_seed")
                if sample_seed is not None and saved_sample_seed is not None:
                    if int(sample_seed) != int(saved_sample_seed):
                        raise ValueError(
                            "Checkpoint sample_seed does not match the seed derived from random_seed."
                        )
                sample_seed = saved_sample_seed
            elif sample_seed is None:
                initial_rng = None
                exterior_rng = None
                schedule_rng = np.random.default_rng()
                dynamics_rng = None
                with update_timing.measure("initial_state_generation"):
                    initial_top = _prepare_initial_top(
                        instance,
                        rng=initial_rng,
                        sample_index=max(0, int(sample_idx or 1) - 1),
                    )
                    G_top = initial_top.copy() if isinstance(initial_top, OccupiedFrameState) else np.array(initial_top, copy=True)
                if meas_slab_only_effective and not frame_init_prepared:
                    G_top = instance._prepare_exterior_product_state(
                        G_top,
                        mode=(
                            "forced_occupied"
                            if effective_postselect
                            else "born_conditioned"
                        ),
                        rng=exterior_rng,
                    )
                start_cycle = 1
                cumulative_log_weight = 0.0
            else:
                stream_sequences = np.random.SeedSequence(int(sample_seed)).spawn(4)
                initial_rng, exterior_rng, schedule_rng, dynamics_rng = (
                    np.random.default_rng(stream) for stream in stream_sequences
                )
                with update_timing.measure("initial_state_generation"):
                    initial_top = _prepare_initial_top(
                        instance,
                        rng=initial_rng,
                        sample_index=max(0, int(sample_idx or 1) - 1),
                    )
                    G_top = initial_top.copy() if isinstance(initial_top, OccupiedFrameState) else np.array(initial_top, copy=True)
                if meas_slab_only_effective and not frame_init_prepared:
                    G_top = instance._prepare_exterior_product_state(
                        G_top,
                        mode=(
                            "forced_occupied"
                            if effective_postselect
                            else "born_conditioned"
                        ),
                        rng=exterior_rng,
                    )
                start_cycle = 1
                cumulative_log_weight = 0.0
            if frame_native and not isinstance(G_top, OccupiedFrameState):
                if state_representation_norm == "physical_frame":
                    with update_timing.measure("pure_frame_factorization"):
                        G_top = OccupiedFrameState.from_centered_covariance(
                            G_top,
                            representation="physical_frame",
                            timing=update_timing,
                            zero_tolerance=trajectory_replay_probability_tol,
                        )
                elif init_mode == "maxmix" and np.count_nonzero(G_top) == 0:
                    with update_timing.measure("maxmix_frame_allocation"):
                        G_top = OccupiedFrameState.maximally_mixed(
                            Nlayer,
                            timing=update_timing,
                            zero_tolerance=trajectory_replay_probability_tol,
                        )
                else:
                    with update_timing.measure("maxmix_frame_allocation"):
                        G_top = OccupiedFrameState.from_centered_covariance(
                            G_top,
                            representation="purification_frame",
                            timing=update_timing,
                            zero_tolerance=trajectory_replay_probability_tol,
                        )
            batch_start = int(sample_idx) - 1 if sample_idx is not None else 0
            batch_index = batch_start
            sample_offsets = np.asarray([0], dtype=np.int64)
            choi_state = (
                instance._init_choi_state(
                    batch_count=1,
                    singular_tol=choi_singular_tol,
                    basis_idx=active_top_layer_indices if meas_slab_only_effective else None,
                    failure_mode=choi_failure_mode,
                )
                if track_choi
                else None
            )
            # Initialize the tangent frame natively after burn-in. Splitting the
            # physical trajectory into two runs would re-prepare the exterior
            # when meas_slab_only is active.
            lyapunov_state = None
            history = [] if store_full_history else None
            last_saved_cycle = start_cycle - 1
            history_stride = history_stride_outer
            if store_full_history and save_init:
                history.append(
                    _materialize_frame(
                        G_top,
                        reason="history",
                        cycle=0,
                        sample_index=batch_start,
                    )
                    if frame_native
                    else G_top.copy()
                )
            if cycle_observer is not None and local_resume is None:
                cycle_observer(
                    cycle=0,
                    G=(
                        _materialize_frame(
                            G_top,
                            reason="cycle_observer",
                            cycle=0,
                            sample_index=batch_start,
                        )
                        if frame_native
                        else G_top
                    ),
                    batch_index=int(batch_index),
                    batch_start=int(batch_start),
                    batch_count=1,
                )
            if native_cycle_observer is not None and local_resume is None:
                _emit_native_cycle_observer(
                    cycle=0,
                    state=G_top,
                    batch_index=int(batch_index),
                    batch_start=int(batch_start),
                    batch_count=1,
                )
            if timing_observer is not None and local_resume is None:
                timing_observer(
                    cycle=0,
                    timing=update_timing.snapshot(cycle=0),
                    batch_index=int(batch_index),
                    batch_start=int(batch_start),
                    batch_count=1,
                )

            exclude_arg_inner = dw_exclude_norm if (getattr(instance, "DW", False) and dw_exclude_norm is not None) else None
            
            local_seq = instance._sequence_helper(
                sequence_mode, 
                Nx=instance.Nx, 
                Ny=instance.Ny, 
                rng=schedule_rng,
                dw_exclude=exclude_arg_inner,
                skip_trivial=meas_slab_only_effective,
            )
            local_seq = instance._filter_sequence_info_to_sites(
                local_seq,
                Nx=instance.Nx,
                site_ids=measurement_site_ids_array,
            )
            coords_for_len = local_seq["coords_for_len"]
            iter_fn = local_seq["iter_fn"]
            coords_len = len(coords_for_len)
            total_sites = ((cycles_eff - start_cycle + 1) * coords_len)
            desc = "Markov RAC (sites)"
            if sequence_mode == "dw_symmetric_random":
                desc = "Markov RAC (DW symmetric random)"

            if sample_idx is not None:
                suffix = f"{sample_idx}/{total_samples}" if total_samples and total_samples > 1 else f"{sample_idx}"
                desc = f"Sample {suffix} | {desc}"
            pbar = tqdm(total=total_sites, desc=desc, unit="site", leave=True) if (enable_progress and progress) else None
            last_eta = None

            def _capture_eta():
                nonlocal last_eta
                if pbar is None: return
                rem = pbar.format_dict.get("remaining", None)
                if rem is not None:
                    try:
                        rem = float(rem)
                        if rem > 0 and np.isfinite(rem): last_eta = rem
                    except: pass

            final_checkpoint = None

            def _checkpoint_payload(completed_cycle, ordered_site_ids):
                if initial_rng is None or exterior_rng is None or dynamics_rng is None:
                    raise ValueError(
                        "checkpoint_observer requires a nonnegative random_seed so all RNG "
                        "streams are explicit and restartable."
                    )
                payload = {
                    "version": 2 if frame_native else 1,
                    "signature": copy.deepcopy(checkpoint_signature),
                    "completed_cycles": int(completed_cycle),
                    "cumulative_log_weight": float(cumulative_log_weight),
                    "random_seed": random_seed,
                    "sample_seed": None if sample_seed is None else int(sample_seed),
                    "last_ordered_site_ids": np.asarray(
                        ordered_site_ids, dtype=np.int64
                    ).copy(),
                    "rng_states": {
                        "initialization": copy.deepcopy(initial_rng.bit_generator.state),
                        "exterior": copy.deepcopy(exterior_rng.bit_generator.state),
                        "schedule": copy.deepcopy(schedule_rng.bit_generator.state),
                        "dynamics": copy.deepcopy(dynamics_rng.bit_generator.state),
                    },
                }
                if frame_native:
                    payload["native_state"] = G_top.snapshot(copy=True)
                else:
                    payload["G"] = np.array(
                        G_top, dtype=np.complex128, copy=True
                    )
                return payload

            trajectory_update_total_ns = 0
            for cyc in range(start_cycle, cycles_eff + 1):
                if controller_twist_schedule_array is not None:
                    instance.construct_OW_projectors(
                        nshell=instance.nshell,
                        DW=instance.DW,
                        trial_orbitals=instance.trial_orbitals,
                        dw_truncation=instance.dw_truncation,
                        twist_y=float(controller_twist_schedule_array[cyc]),
                    )
                update_timing.set_cycle(cyc)
                cycle_started_ns = (
                    time.perf_counter_ns() if update_timing.enabled() else None
                )
                if lyapunov_enabled and cyc == lyapunov_start_cycle:
                    tangent_basis_idx = (
                        active_top_layer_indices
                        if meas_slab_only_effective and not lyapunov_full_space
                        else None
                    )
                    if lyapunov_basis_mode == "pure_occupied_empty":
                        lyapunov_state = (
                            instance._init_pure_occupied_empty_lyapunov_state(
                                G_top,
                                basis_idx=tangent_basis_idx,
                                singular_tol=lyapunov_singular_tol,
                                failure_mode=lyapunov_failure_mode,
                                purity_tolerance=initial_purity_tolerance,
                            )
                        )
                    else:
                        lyapunov_state = instance._init_lyapunov_state(
                            batch_count=1,
                            n_vec=lyapunov_nvec,
                            basis_idx=tangent_basis_idx,
                            initial_frame=lyapunov_initial_frame,
                            sample_start=batch_start,
                            track_restricted_core=lyapunov_track_restricted_core,
                            track_record_fisher=lyapunov_track_record_fisher,
                            singular_tol=lyapunov_singular_tol,
                            failure_mode=lyapunov_failure_mode,
                        )
                schedule_started_ns = (
                    time.perf_counter_ns() if update_timing.enabled(detailed=True) else None
                )
                if replay_by_cycle is None and site_schedule_by_cycle is None:
                    cycle_replay_entries = None
                    iter_coords = iter_fn()
                elif replay_by_cycle is None:
                    cycle_replay_entries = None
                    iter_coords = [
                        (
                            int(site_id) % int(instance.Nx),
                            int(site_id) // int(instance.Nx),
                        )
                        for site_id in site_schedule_by_cycle[cyc]
                    ]
                else:
                    cycle_replay_entries = replay_by_cycle[cyc]
                    iter_coords = [
                        (
                            int(entry["site_id"]) % int(instance.Nx),
                            int(entry["site_id"]) // int(instance.Nx),
                        )
                        for entry in cycle_replay_entries
                    ]
                    expected_sites = {
                        int(x) + int(instance.Nx) * int(y) for x, y in coords_for_len
                    }
                    replay_sites = [
                        int(entry["site_id"]) for entry in cycle_replay_entries
                    ]
                    if (
                        len(replay_sites) != len(expected_sites)
                        or len(set(replay_sites)) != len(replay_sites)
                        or set(replay_sites) != expected_sites
                    ):
                        raise ValueError(
                            f"Trajectory replay cycle {cyc} does not visit the expected "
                            "active site set exactly once."
                        )
                ordered_site_ids = [
                    int(x) + int(instance.Nx) * int(y) for x, y in iter_coords
                ]
                if schedule_started_ns is not None:
                    update_timing.add_time(
                        "schedule_or_replay_decode",
                        time.perf_counter_ns() - schedule_started_ns,
                        detailed=True,
                    )
                for site_position, (Rx, Ry) in enumerate(iter_coords):
                    site_started_ns = (
                        time.perf_counter_ns()
                        if update_timing.enabled(detailed=True)
                        else None
                    )
                    site_id = int(Rx) + int(instance.Nx) * int(Ry)
                    replay_entry = (
                        None
                        if cycle_replay_entries is None
                        else cycle_replay_entries[site_position]
                    )
                    if replay_entry is not None and int(replay_entry["site_id"]) != site_id:
                        raise ValueError(
                            f"Trajectory replay site mismatch in cycle {cyc}: "
                            f"expected {int(replay_entry['site_id'])}, got {site_id}."
                        )
                    choi_context = {
                        "cycle": int(cyc),
                        "site_id": site_id,
                        "batch_index": int(batch_index),
                        "batch_start": int(batch_start),
                    }
                    if effective_postselect:
                        G_top = instance.post_selection_markov_top_layer(
                            G_top,
                            Rx,
                            Ry,
                            sample_offsets=sample_offsets,
                            choi_state=choi_state,
                            choi_context=choi_context,
                            lyapunov_state=lyapunov_state,
                        )
                        branch_summary = {
                            "branch_log_weight": 0.0,
                            "measurement_log_weight": 0.0,
                            "correction_log_weight": 0.0,
                            "forced_postselect": True,
                            "branch_events": (),
                        }
                    elif (
                        postselect_probability > 0.0
                        and instance._rng_random(dynamics_rng) < postselect_probability
                    ):
                        G_top = instance.post_selection_markov_top_layer(
                            G_top,
                            Rx,
                            Ry,
                            sample_offsets=sample_offsets,
                            choi_state=choi_state,
                            choi_context=choi_context,
                            lyapunov_state=lyapunov_state,
                        )
                        branch_summary = {
                            "branch_log_weight": 0.0,
                            "measurement_log_weight": 0.0,
                            "correction_log_weight": 0.0,
                            "forced_postselect": True,
                            "branch_events": (),
                        }
                    else:
                        markov_result = instance.markov_meas_feedback(
                            G_top, Rx, Ry, n_a=n_a,
                            p_gain=p_gain_eff,
                            p_loss=p_loss_eff,
                            perfect_correction=perfect_correction,
                            sample_offsets=sample_offsets,
                            choi_state=choi_state,
                            choi_context=choi_context,
                            lyapunov_state=lyapunov_state,
                            return_weight_summary=(
                                trajectory_weight_observer is not None
                                or replay_entry is not None
                            ),
                            rng=dynamics_rng,
                            branch_replay_events=(
                                None
                                if replay_entry is None
                                else replay_entry["branch_events"]
                            ),
                            replay_probability_tol=trajectory_replay_probability_tol,
                            state_event_observer=native_event_observer,
                            event_context=choi_context,
                        )
                        if trajectory_weight_observer is not None or replay_entry is not None:
                            G_top, branch_summary = markov_result
                        else:
                            G_top = markov_result
                            branch_summary = None

                    if site_started_ns is not None:
                        update_timing.add_time(
                            "site_total",
                            time.perf_counter_ns() - site_started_ns,
                            detailed=True,
                        )

                    if trajectory_weight_observer is not None:
                        cumulative_log_weight += float(branch_summary["branch_log_weight"])
                        trajectory_weight_observer(
                            cycle=int(cyc),
                            site_id=site_id,
                            sample_index=int(batch_start),
                            batch_index=int(batch_index),
                            batch_start=int(batch_start),
                            batch_count=1,
                            branch_log_weight=float(branch_summary["branch_log_weight"]),
                            measurement_log_weight=float(branch_summary["measurement_log_weight"]),
                            correction_log_weight=float(branch_summary["correction_log_weight"]),
                            cumulative_log_weight=float(cumulative_log_weight),
                            forced_postselect=bool(branch_summary["forced_postselect"]),
                            branch_events=branch_summary["branch_events"],
                        )

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
                        _capture_eta()
                if cycle_started_ns is not None:
                    cycle_elapsed_ns = time.perf_counter_ns() - cycle_started_ns
                    update_timing.add_time(
                        "cycle_total", cycle_elapsed_ns
                    )
                    trajectory_update_total_ns += int(cycle_elapsed_ns)
                if store_full_history:
                    if history_stride is None or (cyc % int(max(1, history_stride))) == 0:
                        history.append(
                            _materialize_frame(
                                G_top,
                                reason="history",
                                cycle=cyc,
                                sample_index=batch_start,
                            )
                            if frame_native
                            else G_top.copy()
                        )
                        last_saved_cycle = cyc
                if cycle_observer is not None:
                    cycle_observer(
                        cycle=int(cyc),
                        G=(
                            _materialize_frame(
                                G_top,
                                reason="cycle_observer",
                                cycle=cyc,
                                sample_index=batch_start,
                            )
                            if frame_native
                            else G_top
                        ),
                        batch_index=int(batch_index),
                        batch_start=int(batch_start),
                        batch_count=1,
                    )
                if frame_native:
                    if cyc % frame_reorthonormalize_interval == 0:
                        G_top.reorthonormalize()
                    with update_timing.measure("gram_residual_check", detailed=True):
                        gram_residual = G_top.gram_residual()
                    if not np.isfinite(gram_residual) or gram_residual > 1e-8:
                        raise FloatingPointError(
                            f"Frame Gram residual exceeded the hard runtime ceiling at "
                            f"cycle {cyc}: {gram_residual:.3e}."
                        )
                if native_cycle_observer is not None:
                    _emit_native_cycle_observer(
                        cycle=int(cyc),
                        state=G_top,
                        batch_index=int(batch_index),
                        batch_start=int(batch_start),
                        batch_count=1,
                    )
                if timing_observer is not None:
                    timing_observer(
                        cycle=int(cyc),
                        timing=update_timing.snapshot(cycle=cyc),
                        batch_index=int(batch_index),
                        batch_start=int(batch_start),
                        batch_count=1,
                    )
                if checkpoint_observer is not None or local_resume is not None:
                    final_checkpoint = _checkpoint_payload(cyc, ordered_site_ids)
                    if checkpoint_observer is not None:
                        checkpoint_observer(
                            cycle=int(cyc), state=copy.deepcopy(final_checkpoint)
                        )
                if lyapunov_state is not None:
                    elapsed_lyapunov_cycle = int(cyc - lyapunov_start_cycle + 1)
                    spectra = instance._lyapunov_end_cycle(
                        lyapunov_state, elapsed_lyapunov_cycle
                    )
                    lyapunov_extra = (
                        instance._lyapunov_min_abs_vector_payload(
                            lyapunov_state, elapsed_lyapunov_cycle
                        )
                        if cyc == cycles_eff
                        else {}
                    )
                    if lyapunov_observer is not None:
                        lyapunov_G = (
                            _materialize_frame(
                                G_top,
                                reason="lyapunov_observer",
                                cycle=cyc,
                                sample_index=batch_start,
                            )
                            if frame_native
                            else G_top
                        )
                        lyapunov_observer(
                            cycle=elapsed_lyapunov_cycle,
                            spectra=spectra,
                            G=lyapunov_G,
                            batch_index=int(batch_index),
                            batch_start=int(batch_start),
                            batch_count=1,
                            **lyapunov_extra,
                        )
                    if lyapunov_frame_observer is not None:
                        requires_covariance = bool(
                            getattr(
                                lyapunov_frame_observer,
                                "requires_physical_covariance",
                                True,
                            )
                        )
                        lyapunov_frame_G = (
                            _materialize_frame(
                                G_top,
                                reason="lyapunov_frame_observer",
                                cycle=cyc,
                                sample_index=batch_start,
                            )
                            if frame_native and requires_covariance
                            else G_top
                            if not frame_native
                            else None
                        )
                        lyapunov_frame_observer(
                            cycle=int(cyc),
                            lyapunov_cycle=elapsed_lyapunov_cycle,
                            spectra=spectra,
                            G=lyapunov_frame_G,
                            native_state=(G_top if frame_native else None),
                            batch_index=int(batch_index),
                            batch_start=int(batch_start),
                            batch_count=1,
                            **instance._lyapunov_frame_payload(lyapunov_state),
                        )
                if choi_observer is not None and cyc in set(choi_observer_cycles_norm or ()):
                    choi_observer_payload = {
                        "cycle": int(cyc),
                        "sigma_ll": choi_state["LL"],
                        "sigma_lr": choi_state["LR"],
                        "sigma_rr": choi_state["RR"],
                        "batch_index": int(batch_index),
                        "batch_start": int(batch_start),
                        "batch_count": 1,
                        "min_abs_d": choi_state["min_abs_d"],
                        "min_abs_d_context": choi_state["min_abs_d_context"],
                        "choi_active_mask": choi_state["active"],
                        "choi_failure_records": tuple(choi_state["failure_records"]),
                    }
                    if meas_slab_only_effective:
                        choi_observer_payload.update(
                            {
                                "active_top_layer_indices": choi_state["basis_idx"],
                                "full_nlayer": Nlayer,
                            }
                        )
                    choi_observer_result = choi_observer(**choi_observer_payload)
                    if choi_observer_result is not None:
                        if choi_failure_mode != "censor":
                            raise RuntimeError(
                                "choi_observer requested trajectory censoring while choi_failure_mode is not 'censor'."
                            )
                        deactivate = np.asarray(
                            choi_observer_result.get("deactivate_sample_offsets", ()),
                            dtype=np.int64,
                        ).reshape(-1)
                        if deactivate.size:
                            choi_state["active"][deactivate] = False
                        reasons = choi_observer_result.get("failure_records", ())
                        choi_state["failure_records"].extend(dict(record) for record in reasons)

            if update_timing.enabled():
                update_timing.add_time(
                    "trajectory_total",
                    trajectory_update_total_ns,
                )
            update_timing.set_cycle(None)
            instance._last_update_timing = update_timing.snapshot()
            if store_full_history and history_stride is not None and last_saved_cycle != cycles_eff:
                history.append(
                    _materialize_frame(
                        G_top,
                        reason="history_final_stride",
                        cycle=cycles_eff,
                        sample_index=batch_start,
                    )
                    if frame_native
                    else G_top.copy()
                )
            if final_checkpoint is not None:
                checkpoint_states.append(final_checkpoint)
            if track_choi:
                choi_diagnostics.append(
                    {
                        "batch_index": int(batch_index),
                        "batch_start": int(batch_start),
                        "batch_count": 1,
                        "min_abs_d": float(choi_state["min_abs_d"]),
                        "min_abs_d_context": choi_state["min_abs_d_context"],
                        "choi_active_final": choi_state["active"].tolist(),
                        "choi_failure_records": list(choi_state["failure_records"]),
                    }
                )
            if lyapunov_state is not None:
                lyapunov_diagnostics.append(
                    {
                        "batch_index": int(batch_index),
                        "batch_start": int(batch_start),
                        "batch_count": 1,
                        "lyapunov_cycles": int(
                            cycles_eff - lyapunov_start_cycle + 1
                        ),
                        "null_counts": lyapunov_state["null_counts"].tolist(),
                        "active_final": lyapunov_state["active"].tolist(),
                        "min_branch_probability": lyapunov_state[
                            "min_branch_probability"
                        ].tolist(),
                        "min_abs_born_denominator": lyapunov_state[
                            "min_abs_born_denominator"
                        ].tolist(),
                        "invalid_branch_count": lyapunov_state[
                            "invalid_branch_count"
                        ].tolist(),
                        "failure_records": list(
                            lyapunov_state["failure_records"]
                        ),
                    }
                )

            if pbar is not None:
                _capture_eta()
                pbar.close()

            if G_history:
                result = np.stack(history, axis=0) if history else np.empty((0, Nlayer, Nlayer), dtype=np.complex128)
            elif frame_native and return_native_state:
                result = G_top.snapshot(copy=True)
            else:
                result = (
                    _materialize_frame(
                        G_top,
                        reason="legacy_final_return",
                        cycle=cycles_eff,
                        sample_index=batch_start,
                    )
                    if frame_native
                    else G_top.copy()
                )

            if return_eta:
                return result, (last_eta if last_eta is not None else None)
            return result

        if effective_postselect:
            if samples is not None and int(samples) != 1:
                print("[info] postselect_probability=1 overrides samples; using samples=1.")
            if parallelize_samples:
                print("[info] postselect_probability=1 overrides parallelize_samples; using serial execution.")
            samples = 1
            parallelize_samples = False
            n_jobs = 1
        else:
            samples = 1 if samples is None else int(samples)
        if samples <= 0: raise ValueError("samples must be a positive integer")
        if (resume_checkpoint is not None or checkpoint_observer is not None) and (
            parallelize_samples or samples != 1
        ):
            raise ValueError(
                "CPU checkpoint export/resume requires exactly one serial sample."
            )
        if trajectory_replay_entries is not None and (
            parallelize_samples or samples != 1
        ):
            raise ValueError(
                "CPU trajectory_replay requires exactly one serial sample."
            )
        if site_schedule_replay_array is not None and (
            parallelize_samples or samples != 1
        ):
            raise ValueError(
                "CPU site_schedule_replay requires exactly one serial sample."
            )
        if track_choi and parallelize_samples:
            raise ValueError("CPU Choi tracking currently requires serial sample execution; set parallelize_samples=False.")
        if cycle_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU cycle_observer requires serial sample execution; "
                "set parallelize_samples=False and parallelize trajectories externally."
            )
        if native_cycle_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU native_cycle_observer requires serial sample execution; "
                "parallelize trajectories externally."
            )
        if native_event_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU native_event_observer requires serial sample execution; "
                "parallelize trajectories externally."
            )
        if timing_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU timing_observer requires serial sample execution; "
                "parallelize trajectories externally."
            )
        if checkpoint_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU checkpoint_observer requires serial sample execution; "
                "parallelize trajectories externally."
            )
        if trajectory_weight_observer is not None and parallelize_samples:
            raise ValueError(
                "CPU trajectory_weight_observer requires serial sample execution; "
                "set parallelize_samples=False and parallelize trajectories externally."
            )
        if lyapunov_enabled and parallelize_samples:
            raise ValueError(
                "CPU Lyapunov observers require serial sample execution; "
                "set parallelize_samples=False and parallelize trajectories externally."
            )
        if random_seed is None:
            if parallelize_samples:
                entropy_state = np.random.SeedSequence().generate_state(samples, dtype=np.uint64)
                sample_seed_values = [int(value) for value in entropy_state]
            else:
                sample_seed_values = [None] * samples
        else:
            sample_sequences = np.random.SeedSequence(random_seed).spawn(samples)
            sample_seed_values = [
                int(sequence_state.generate_state(1, dtype=np.uint64)[0])
                for sequence_state in sample_sequences
            ]

        if throttle is None:
            throttle = bool(parallelize_samples)
        else:
            throttle = bool(throttle)

        def _choi_result_metadata():
            return {
                "choi_tracked": bool(track_choi),
                "choi_observer_cycles": choi_observer_cycles_norm,
                "choi_diagnostics": choi_diagnostics,
                "choi_failure_mode": choi_failure_mode,
                "lyapunov_tracked": bool(lyapunov_enabled),
                "lyapunov_basis_mode": lyapunov_basis_mode,
                "lyapunov_start_cycle": int(lyapunov_start_cycle),
                "lyapunov_full_space": bool(lyapunov_full_space),
                "lyapunov_observation_cycles": int(
                    cycles_eff - lyapunov_start_cycle + 1
                ),
                "lyapunov_track_restricted_core": bool(
                    lyapunov_track_restricted_core
                ),
                "lyapunov_track_record_fisher": bool(
                    lyapunov_track_record_fisher
                ),
                "lyapunov_singular_tol": float(lyapunov_singular_tol),
                "lyapunov_failure_mode": lyapunov_failure_mode,
                "lyapunov_diagnostics": lyapunov_diagnostics,
                "physical_covariance_update": self._physical_covariance_update_label(
                    physical_covariance_update_norm
                ),
                "state_representation": state_representation_norm,
                "state_representation_requested": state_representation_requested,
                "state_representation_resolved": state_representation_norm,
                "state_representation_resolution_reason": resolution_reason,
                "initial_purity_tolerance": float(initial_purity_tolerance),
                "initial_purity_defects": (
                    None
                    if initial_purity_defects is None
                    else initial_purity_defects.tolist()
                ),
                "explicit_covariance_override": explicit_covariance_override,
                "frame_algorithm_version": (
                    FRAME_ALGORITHM_VERSION if frame_native else None
                ),
                "covariance_materialization_count": len(
                    covariance_materializations
                ),
                "covariance_materializations": copy.deepcopy(
                    covariance_materializations
                ),
                "require_no_covariance_materialization": bool(
                    require_no_covariance_materialization
                ),
                "native_cycle_observer": native_cycle_observer is not None,
                "native_event_observer": native_event_observer is not None,
                "return_native_state": bool(return_native_state),
                "timing_level": timing_level_norm,
                "timing_observer": timing_observer is not None,
                "frame_covariance_materialized_for_cycle_observer": bool(
                    frame_native and cycle_observer is not None
                ),
                "timing": copy.deepcopy(
                    getattr(self, "_last_update_timing", None)
                ),
                "trajectory_replay": trajectory_replay_entries is not None,
                "trajectory_replay_sites": (
                    0 if trajectory_replay_entries is None else len(trajectory_replay_entries)
                ),
                "trajectory_replay_probability_tol": float(
                    trajectory_replay_probability_tol
                ),
                "site_schedule_replay": site_schedule_replay_array is not None,
                "site_schedule_replay_shape": (
                    None
                    if site_schedule_replay_array is None
                    else list(site_schedule_replay_array.shape)
                ),
                "site_schedule_replay_sha256": site_schedule_replay_hash,
                "measurement_site_ids": (
                    None
                    if measurement_site_ids_array is None
                    else measurement_site_ids_array.tolist()
                ),
                "measurement_site_ids_sha256": measurement_site_ids_hash,
                "controller_twist_schedule": (
                    None
                    if controller_twist_schedule_array is None
                    else controller_twist_schedule_array.tolist()
                ),
                "controller_twist_schedule_sha256": controller_twist_schedule_hash,
                "controller_twist_gauge": (
                    controller_twist_gauge_norm
                    if controller_twist_schedule_array is not None
                    else None
                ),
                "frame_init_prepared": bool(frame_init_prepared),
                "checkpoint_resume": resume_checkpoint is not None,
                "checkpoint_state": (
                    copy.deepcopy(checkpoint_states[0])
                    if len(checkpoint_states) == 1
                    else None
                ),
                "twist_x": float(getattr(self, "twist_x", 0.0)),
                "twist_y": float(getattr(self, "twist_y", 0.0)),
                "random_seed": random_seed,
                "sample_seeds": sample_seed_values,
                "rng_streams": ["initialization", "exterior", "schedule", "dynamics"],
                "seed_derivation": "SeedSequence(root).spawn(samples); SeedSequence(sample_seed).spawn(4)",
                "exterior_preparation": (
                    "skipped_prepared_frame"
                    if meas_slab_only_effective and frame_init_prepared
                    else
                    "forced_occupied_onsite_before_cycle_0"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned_onsite_before_cycle_0"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_mode": (
                    "already_prepared"
                    if meas_slab_only_effective and frame_init_prepared
                    else
                    "forced_occupied"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_basis": (
                    "canonical_unit_cell_orbital"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_orbital_count": (
                    2 * len(self._exterior_site_coordinates())
                    if meas_slab_only_effective
                    else 0
                ),
            }

        if not parallelize_samples:
            histories = [] if G_history else None
            native_finals = [] if (frame_native and return_native_state) else None
            finals = (
                None
                if G_history or native_finals is not None
                else np.empty((samples, Nlayer, Nlayer), dtype=np.complex128)
            )
            total_samples = samples
            _emit_save_notice(total_samples)
            for idx in range(total_samples):
                if progress:
                    single_result, eta = _run_single(
                        self,
                        enable_progress=True,
                        return_eta=True,
                        sample_idx=idx + 1,
                        total_samples=total_samples,
                        sample_seed=sample_seed_values[idx],
                    )
                else:
                    single_result = _run_single(
                        self,
                        enable_progress=False,
                        return_eta=False,
                        sample_idx=idx + 1,
                        total_samples=total_samples,
                        sample_seed=sample_seed_values[idx],
                    )
                    eta = None
                if histories is not None:
                    histories.append(single_result)
                if native_finals is not None:
                    native_finals.append(single_result)
                if finals is not None:
                    finals[idx] = single_result
                if progress and total_samples > 1:
                    completed = idx + 1
                    rem = total_samples - completed
                    if eta is not None and np.isfinite(eta) and rem > 0:
                        print(f"Samples {completed}/{total_samples} completed. Est remaining time: {self.format_interval(eta * rem)}", flush=True)
                    else:
                        print(f"Samples {completed}/{total_samples} completed.", flush=True)

            if G_history:
                G_hist = np.stack(histories, axis=0)
                saved_path = _save_histories(G_hist, samples_count=G_hist.shape[0])
                result = {
                    "G_hist": G_hist,
                    "G_hist_avg": np.mean(G_hist, axis=0),
                    "samples": total_samples,
                    "T": G_hist.shape[1],
                    "save_path": saved_path,
                    "postselect_probability": float(postselect_probability),
                    "meas_slab_only_requested": meas_slab_only_requested,
                    "meas_slab_only_effective": meas_slab_only_effective,
                    "active_top_layer_indices": active_top_layer_indices,
                    **_choi_result_metadata(),
                }
                return result

            if native_finals is not None:
                result = {
                    "native_final": (
                        native_finals[0]
                        if total_samples == 1
                        else native_finals
                    ),
                    "samples": total_samples,
                    "T": cycles_eff + 1,
                    "save_path": None,
                    "postselect_probability": float(postselect_probability),
                    "meas_slab_only_requested": meas_slab_only_requested,
                    "meas_slab_only_effective": meas_slab_only_effective,
                    "active_top_layer_indices": active_top_layer_indices,
                    **_choi_result_metadata(),
                }
                return result

            G_final = finals if finals is not None else np.empty((0, Nlayer, Nlayer), dtype=np.complex128)
            saved_path = _save_finals(G_final, samples_count=G_final.shape[0])
            result = {
                "G_final": G_final,
                "G_final_avg": np.mean(G_final, axis=0),
                "samples": total_samples,
                "T": cycles_eff + 1,
                "save_path": saved_path,
                "postselect_probability": float(postselect_probability),
                "meas_slab_only_requested": meas_slab_only_requested,
                "meas_slab_only_effective": meas_slab_only_effective,
                "active_top_layer_indices": active_top_layer_indices,
                **_choi_result_metadata(),
            }
            return result

        S = samples
        n_jobs_eff = n_jobs
        _emit_save_notice(S)
        if backend not in ("loky", "threading"): raise ValueError("backend error")
        seeds = sample_seed_values

        def _worker(seed_u32, sample_zero_index):
            with threadpool_limits(limits=1):
                np.random.seed(int(seed_u32) & 0xFFFFFFFF)
                child = self._spawn_for_parallel()
                # Fast path: execute a single trajectory directly, avoiding recursive
                # run_markov_circuit(...) setup inside each worker.
                G_out = _run_single(
                    child,
                    enable_progress=False,
                    return_eta=False,
                    sample_idx=int(sample_zero_index) + 1,
                    total_samples=S,
                    sample_seed=seed_u32,
                )
                return G_out

        def _available_mem_bytes():
            try:
                import psutil  # optional; used only if available
                return psutil.virtual_memory().available
            except Exception:
                return None

        def _estimate_frames():
            if not store_full_history:
                return 1
            init_frames = 1 if save_init else 0
            if history_stride_outer is None:
                return init_frames + cycles_eff
            stride = int(max(1, history_stride_outer))
            frames = init_frames + (cycles_eff // stride)
            if cycles_eff % stride != 0:
                frames += 1
            return max(1, frames)

        def _estimate_bytes_per_worker():
            frames = _estimate_frames()
            bytes_per_state = Nlayer * Nlayer * np.dtype(np.complex128).itemsize
            return int(bytes_per_state * frames * 1.2)

        def _load_based_cap(base):
            try:
                load1 = os.getloadavg()[0]
                if not math.isfinite(load1) or load1 <= 0:
                    return base
                if load1 <= base:
                    return base
                return max(1, int(base * (base / load1)))
            except Exception:
                return base

        if max_in_flight is not None:
            try:
                max_in_flight = int(max_in_flight)
            except Exception as exc:
                raise ValueError("max_in_flight must be an integer > 0.") from exc
            if max_in_flight <= 0:
                raise ValueError("max_in_flight must be an integer > 0.")
        try:
            throttle_mem_frac = float(throttle_mem_frac)
        except Exception as exc:
            raise ValueError("throttle_mem_frac must be a float in (0, 1].") from exc
        if not (0.0 < throttle_mem_frac <= 1.0):
            raise ValueError("throttle_mem_frac must be in (0, 1].")

        def _base_workers():
            try:
                eff = joblib.effective_n_jobs(n_jobs_eff)
                return max(1, int(eff))
            except Exception:
                if n_jobs_eff is None:
                    return max(1, int(joblib.cpu_count()))
                return max(1, int(n_jobs_eff))

        def _calc_batch_workers():
            base = _base_workers()
            if not throttle:
                return base
            if max_in_flight is not None:
                return max(1, min(base, max_in_flight))
            bytes_per_worker = _estimate_bytes_per_worker()
            avail = _available_mem_bytes()
            mem_cap = base if (avail is None or bytes_per_worker <= 0) else int((avail * throttle_mem_frac) // bytes_per_worker)
            load_cap = _load_based_cap(base)
            return max(1, min(base, mem_cap, load_cap))

        def _run_parallel_batch(start_idx, count, workers):
            if backend == "loky":
                with parallel_backend("loky", n_jobs=workers, inner_max_num_threads=1):
                    with threadpool_limits(limits=1):
                        return Parallel(n_jobs=workers)(
                            delayed(_worker)(seeds[i], i) for i in range(start_idx, start_idx + count)
                        )
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            with threadpool_limits(limits=1):
                return Parallel(n_jobs=workers, backend="threading")(
                    delayed(_worker)(seeds[i], i) for i in range(start_idx, start_idx + count)
                )

        if not throttle:
            with self._joblib_tqdm_ctx(S, "samples", show_datetime=True):
                if backend == "loky":
                    with parallel_backend("loky", n_jobs=n_jobs_eff, inner_max_num_threads=1):
                        with threadpool_limits(limits=1):
                            G_hist_list = Parallel(n_jobs=n_jobs_eff)(
                                delayed(_worker)(seeds[i], i) for i in range(S)
                            )
                else:
                    os.environ.setdefault("OMP_NUM_THREADS", "1")
                    with threadpool_limits(limits=1):
                        G_hist_list = Parallel(n_jobs=n_jobs_eff, backend="threading")(
                            delayed(_worker)(seeds[i], i) for i in range(S)
                        )
        else:
            G_hist_list = []
            with self._joblib_tqdm_ctx(S, "samples", show_datetime=True):
                idx = 0
                while idx < S:
                    workers = _calc_batch_workers()
                    remaining = S - idx
                    batch_size = min(remaining, workers)
                    G_hist_list.extend(_run_parallel_batch(idx, batch_size, workers))
                    idx += batch_size

        if G_history:
            G_hist = np.stack(G_hist_list, axis=0)
            G_hist_avg = np.mean(G_hist, axis=0)
            saved_path = _save_histories(G_hist, samples_count=G_hist.shape[0])
            result = {
                "G_hist": G_hist,
                "G_hist_avg": G_hist_avg,
                "samples": S,
                "T": G_hist.shape[1],
                "save_path": saved_path,
                "postselect_probability": float(postselect_probability),
                "meas_slab_only_requested": meas_slab_only_requested,
                "meas_slab_only_effective": meas_slab_only_effective,
                "active_top_layer_indices": active_top_layer_indices,
                **_choi_result_metadata(),
            }
            return result

        G_final = np.stack(G_hist_list, axis=0)
        G_final_avg = np.mean(G_final, axis=0)
        saved_path = _save_finals(G_final, samples_count=G_final.shape[0])
        result = {
            "G_final": G_final,
            "G_final_avg": G_final_avg,
            "samples": S,
            "T": cycles_eff + 1,
            "save_path": saved_path,
            "postselect_probability": float(postselect_probability),
            "meas_slab_only_requested": meas_slab_only_requested,
            "meas_slab_only_effective": meas_slab_only_effective,
            "active_top_layer_indices": active_top_layer_indices,
            **_choi_result_metadata(),
        }
        return result

    def run_markov_channel(
        self,
        G_history=True,
        progress=True,
        cycles=20,
        init_mode="default",
        remember_init=True,
        save=True,
        save_suffix=None,
        n_a=0.5,
        sequence="raster_y",
        decoh=True,
        perfect_correction=False,
        schedule_seed=None,
        cycle_observer=None,
        cycle_observer_cycles=None,
    ):
        """
        Deterministic trajectory-averaged Markov channel.

        ``schedule_seed`` makes randomized site schedules reproducible without
        changing the historical unseeded default. ``cycle_observer`` is invoked
        on the live state at cycle zero and after selected completed cycles. The
        callback must treat ``G`` as read-only and may stream compact observables
        without retaining the dense covariance history. If
        ``cycle_observer_cycles`` is omitted, every cycle is observed.
        """
        have_ow = all(hasattr(self, attr) for attr in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm"))
        if not have_ow:
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
            )

        if cycles < 0: raise ValueError("cycles must be non-negative.")
        n_a = float(n_a)
        if not np.isfinite(n_a):
            raise ValueError("n_a must be finite.")
        if n_a < 0.0 or n_a > 1.0:
            raise ValueError("n_a must lie in [0, 1].")
        
        sweeps = int(cycles)
        if schedule_seed is not None:
            original_schedule_seed = schedule_seed
            if isinstance(original_schedule_seed, (bool, np.bool_)):
                raise ValueError("schedule_seed must be a nonnegative integer or None.")
            try:
                schedule_seed = int(original_schedule_seed)
            except Exception as exc:
                raise ValueError(
                    "schedule_seed must be a nonnegative integer or None."
                ) from exc
            if (
                isinstance(original_schedule_seed, (float, np.floating))
                and not float(original_schedule_seed).is_integer()
            ) or schedule_seed < 0:
                raise ValueError("schedule_seed must be a nonnegative integer or None.")

        if cycle_observer_cycles is not None and cycle_observer is None:
            raise ValueError("cycle_observer_cycles requires cycle_observer.")
        observer_cycles = None
        if cycle_observer_cycles is not None:
            try:
                raw_observer_cycles = tuple(cycle_observer_cycles)
            except TypeError as exc:
                raise ValueError(
                    "cycle_observer_cycles must be an iterable of cycle indices."
                ) from exc
            normalized_observer_cycles = []
            for value in raw_observer_cycles:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("cycle_observer_cycles must contain integers.")
                try:
                    cycle_value = int(value)
                except Exception as exc:
                    raise ValueError(
                        "cycle_observer_cycles must contain integers."
                    ) from exc
                if (
                    isinstance(value, (float, np.floating))
                    and not float(value).is_integer()
                ):
                    raise ValueError("cycle_observer_cycles must contain integers.")
                if not 0 <= cycle_value <= sweeps:
                    raise ValueError(
                        "cycle_observer_cycles entries must lie in "
                        f"0..{sweeps}; got {cycle_value}."
                    )
                normalized_observer_cycles.append(cycle_value)
            observer_cycles = tuple(sorted(set(normalized_observer_cycles)))

        channel_order = ("Ap", "Am", "Bp", "Bm")
        # Deprecated modes removed from the public API; keep disabled internally.
        bulk_sweeps = 0
        top_region_last = False
        skip_trivial = False
        trivial_product_state_DW = False
        top_last = False

        Nlayer = self.Ntot // 2
        is_dw_active = getattr(self, "DW", False)
        use_bulk_extra = (top_region_last and is_dw_active and bulk_sweeps > 0)
        use_bulk_extra_effective = use_bulk_extra and (not trivial_product_state_DW) and (not top_last)
        
        seq_info = self._sequence_helper(
            sequence, 
            Nx=self.Nx, 
            Ny=self.Ny, 
            rng=np.random.default_rng(schedule_seed),
            top_region_last=(top_region_last and is_dw_active),
            skip_trivial=(skip_trivial and is_dw_active)
        )
        sequence_mode = seq_info["mode"]

        def _cache_key(
            *, Nx, Ny, sweeps, nshell, DW, init_mode, n_a, decoh,
            perfect_correction, alpha_top, alpha_triv, dw_truncation,
            trial_orbitals, sequence_mode, schedule_seed, channel_order, dw_loc,
        ):
            nsh = "None" if nshell is None else str(nshell)
            seed_token = "None" if schedule_seed is None else str(int(schedule_seed))
            order_token = "-".join(channel_order)
            dw_loc_token = "None" if not dw_loc else "-".join(str(int(x)) for x in dw_loc)
            return (
                f"N{int(Nx)}x{int(Ny)}"
                f"_S{int(sweeps)}"
                f"_nsh{nsh}"
                f"_DW{int(bool(DW))}"
                f"_dwloc{dw_loc_token}"
                f"_dwtrunc{int(bool(dw_truncation))}"
                f"_a1{float(np.real(alpha_top)):g}"
                f"_a2{float(np.real(alpha_triv)):g}"
                f"_trial{str(trial_orbitals)}"
                f"_init-{init_mode}"
                f"_n_a{n_a}"
                f"_decoh{int(bool(decoh))}"
                f"_pc{int(bool(perfect_correction))}"
                f"_seq-{sequence_mode}"
                f"_seed{seed_token}"
                f"_order-{order_token}"
                "_markov_channel"
            )

        def _run_config(store_history):
            n_a_fill = 1.0 if perfect_correction else n_a
            n_a_deplete = 0.0 if perfect_correction else n_a
            return {
                "Nx": int(self.Nx),
                "Ny": int(self.Ny),
                "sweeps": int(sweeps),
                "nshell": self.nshell,
                "DW": bool(self.DW),
                "DW_loc": [int(value) for value in getattr(self, "DW_loc", [])],
                "dw_interval": (
                    None
                    if getattr(self, "dw_interval", None) is None
                    else [int(value) for value in self.dw_interval]
                ),
                "alpha_top": float(np.real(self.alpha_top)),
                "alpha_triv": float(np.real(self.alpha_triv)),
                "trial_orbitals": self.trial_orbitals,
                "dw_truncation": bool(self.dw_truncation),
                "init_mode": init_mode,
                "n_a": float(n_a),
                "n_a_fill": float(n_a_fill),
                "n_a_deplete": float(n_a_deplete),
                "sequence": sequence_mode,
                "schedule_seed": schedule_seed,
                "channel_order": list(channel_order),
                "decoh": bool(decoh),
                "perfect_correction": bool(perfect_correction),
                "cycle_observer_cycles": (
                    None if observer_cycles is None else list(observer_cycles)
                ),
                "store_mode": "history" if store_history else "final",
                "canonical_dynamics_entry_point": "classA_U1FGTN.run_markov_channel",
            }

        def _save_results(history_array, final_G, store_history):
            if not save: return None
            outdir = self._g_history_outdir()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, sweeps=sweeps, nshell=self.nshell,
                DW=self.DW, init_mode=init_mode, n_a=n_a,
                decoh=decoh, perfect_correction=perfect_correction,
                alpha_top=self.alpha_top, alpha_triv=self.alpha_triv,
                dw_truncation=self.dw_truncation,
                trial_orbitals=self.trial_orbitals,
                sequence_mode=sequence_mode, schedule_seed=schedule_seed,
                channel_order=channel_order,
                dw_loc=getattr(self, "DW_loc", None),
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            path = os.path.join(outdir, filename)
            payload = {}
            if store_history and history_array is not None:
                payload["G_hist"] = np.asarray(history_array, dtype=np.complex128)
            if final_G is not None:
                payload["G_final"] = np.asarray(final_G, dtype=np.complex128)
            payload["run_config"] = np.asarray(json.dumps(_run_config(store_history), sort_keys=True))
            np.savez_compressed(path, **payload)
            return path

        def _expected_save_path():
            if not save: return None
            outdir = self._g_history_outdir_rel()
            key = _cache_key(
                Nx=self.Nx, Ny=self.Ny, sweeps=sweeps, nshell=self.nshell,
                DW=self.DW, init_mode=init_mode, n_a=n_a,
                decoh=decoh, perfect_correction=perfect_correction,
                alpha_top=self.alpha_top, alpha_triv=self.alpha_triv,
                dw_truncation=self.dw_truncation,
                trial_orbitals=self.trial_orbitals,
                sequence_mode=sequence_mode, schedule_seed=schedule_seed,
                channel_order=channel_order,
                dw_loc=getattr(self, "DW_loc", None),
            )
            filename = f"{key}.npz"
            if save_suffix:
                root, ext = os.path.splitext(filename)
                filename = f"{root}{save_suffix}{ext}"
            return os.path.join(outdir, filename)

        def _emit_save_notice():
            path = _expected_save_path()
            if path is None: return
            label = "history" if G_history else "steady-state"
            print(f"[info] Markov channel will save {label} to {path}")
            time.sleep(2)

        G_top = np.array(self.random_complex_fermion_covariance(N=Nlayer) if init_mode=="default" else np.zeros((Nlayer, Nlayer), dtype=np.complex128) if init_mode=="maxmix" else self.G0, copy=True)

        def _observe_cycle(cycle, ordered_coords=()):
            if cycle_observer is None:
                return
            if observer_cycles is not None and int(cycle) not in observer_cycles:
                return
            ordered_site_ids = np.asarray(
                [int(x) + int(self.Nx) * int(y) for x, y in ordered_coords],
                dtype=np.int64,
            )
            cycle_observer(
                cycle=int(cycle),
                G=G_top,
                ordered_site_ids=ordered_site_ids,
                sequence=sequence_mode,
                schedule_seed=schedule_seed,
            )

        history = [] if G_history else None
        if G_history and remember_init:
            history.append(G_top.copy())
        _observe_cycle(0)

        Nx, Ny = int(self.Nx), int(self.Ny)
        coords_for_len = seq_info["coords_for_len"]
        iter_fn = seq_info["iter_fn"]
        iter_bulk_fn = seq_info["iter_bulk_fn"]

        extra_trivial_sweep = 0
        top_last_extra = 0
        total_sites = None
        if trivial_product_state_DW:
            Nx = self.Nx
            half = Nx // 2
            w = max(1, Nx // 4)
            x0 = max(0, half - w) - 1
            x1 = min(Nx, half + w + 1)
            post_cycles = sweeps // 2
            outside_len = len([c for c in coords_for_len if not (x0 <= c[0] < x1)])
            mid_xs = {half, half - 1}
            mid_xs = {x for x in mid_xs if 0 <= x < Nx}
            mid_len = len([c for c in coords_for_len if c[0] in mid_xs])
            extra_trivial_sweep = post_cycles * (outside_len + mid_len)
        elif top_last:
            Nx = self.Nx
            half = Nx // 2
            w = max(1, Nx // 4)
            x0 = max(0, half - w) - 1
            x1 = min(Nx, half + w + 1)
            self.DW_loc = [int(x0), int(x1 - 1)]
            phase_cycles = sweeps // 2
            coords_len = len(coords_for_len)
            outside_len = len([c for c in coords_for_len if not (x0 <= c[0] < x1)])
            inner_len = len([c for c in coords_for_len if (x0 <= c[0] <= x1)])
            top_last_extra = (phase_cycles * (3 * outside_len + 2 * inner_len))
        if total_sites is None:
            total_sites = (sweeps * len(coords_for_len)) + extra_trivial_sweep + top_last_extra
        if use_bulk_extra_effective:
            total_sites += (bulk_sweeps * seq_info["bulk_len"])

        pbar = tqdm(total=total_sites, desc="Markov channel (sites)", unit="site", leave=True) if (progress and total_sites > 0) else None

        _emit_save_notice()

        n_a_fill = 1.0 if perfect_correction else n_a
        n_a_deplete = 0.0 if perfect_correction else n_a
        use_finite_support_pc = bool(
            self.nshell is not None and perfect_correction and decoh
        )

        # Dense work buffers are unnecessary for the exact finite-support path.
        Il_top = np.eye(Nlayer, dtype=np.complex128)
        Il_half = None
        G2pt = None
        tmp_pg = None
        tmp_gp = None
        tmp_P = None
        chi_Ap_all = np.asarray(self.WF_Ap, dtype=np.complex128)
        chi_Bp_all = np.asarray(self.WF_Bp, dtype=np.complex128)
        chi_Am_all = np.asarray(self.WF_Am, dtype=np.complex128)
        chi_Bm_all = np.asarray(self.WF_Bm, dtype=np.complex128)

        def _ensure_dense_buffers():
            nonlocal Il_half, G2pt, tmp_pg, tmp_gp, tmp_P
            if G2pt is not None:
                return
            Il_half = 0.5 * Il_top
            G2pt = np.empty_like(G_top)
            tmp_pg = np.empty_like(G_top)
            tmp_gp = np.empty_like(G_top)
            tmp_P = np.empty_like(G_top)

        def _apply_deplete(chi):
            nonlocal G2pt, tmp_pg, tmp_gp, tmp_P
            chi_conj = chi.conj()
            chiHG = chi_conj @ G2pt
            Gchi = G2pt @ chi
            np.multiply.outer(chi, chiHG, out=tmp_pg)
            np.multiply.outer(Gchi, chi_conj, out=tmp_gp)
            tmp_pg += tmp_gp
            np.multiply.outer(chi, chi_conj, out=tmp_P)
            if decoh:
                scalar = chiHG @ chi
                np.multiply(tmp_P, (1.0 + n_a_deplete) * scalar, out=tmp_gp)
                G2pt -= tmp_pg
                G2pt += tmp_gp
            else:
                G2pt -= 0.5 * (1.0 - n_a_deplete) * tmp_pg

        def _apply_fill(chi):
            nonlocal G2pt, tmp_pg, tmp_gp, tmp_P
            chi_conj = chi.conj()
            chiHG = chi_conj @ G2pt
            Gchi = G2pt @ chi
            np.multiply.outer(chi, chiHG, out=tmp_pg)
            np.multiply.outer(Gchi, chi_conj, out=tmp_gp)
            tmp_pg += tmp_gp
            np.multiply.outer(chi, chi_conj, out=tmp_P)
            G2pt += n_a_fill * tmp_P
            if decoh:
                scalar = chiHG @ chi
                np.multiply(tmp_P, (2.0 - n_a_fill) * scalar, out=tmp_gp)
                G2pt -= tmp_pg
                G2pt += tmp_gp
            else:
                G2pt -= 0.5 * n_a_fill * tmp_pg

        def _apply_finite_support_pc(payload, channel_key, target_occupied):
            """Apply ``G -> Q G Q + eta P`` on a compact OW support in place."""
            support_idx = payload["idx"]
            comp_idx = payload["comp"]
            chi_local = payload[channel_key]
            G_ss, G_sr = self._local_support_block(G_top, support_idx, comp_idx)
            projector_cache_key = f"_{channel_key}_projector"
            complement_cache_key = f"_{channel_key}_complement"
            projector = payload.get(projector_cache_key)
            complement = payload.get(complement_cache_key)
            if projector is None or complement is None:
                projector = np.outer(chi_local, chi_local.conj())
                complement = np.eye(
                    support_idx.size, dtype=np.complex128
                ) - projector
                payload[projector_cache_key] = projector
                payload[complement_cache_key] = complement
            target_covariance = 1.0 if target_occupied else -1.0
            G_ss_new = (
                complement @ G_ss @ complement
                + target_covariance * projector
            )
            G_top[np.ix_(support_idx, support_idx)] = 0.5 * (
                G_ss_new + G_ss_new.conj().T
            )
            if comp_idx.size > 0:
                G_sr_new = complement @ G_sr
                G_top[np.ix_(support_idx, comp_idx)] = G_sr_new
                G_top[np.ix_(comp_idx, support_idx)] = G_sr_new.conj().T

        def _execute_sweep_with_projectors(coords_list, chi_Ap, chi_Bp, chi_Am, chi_Bm):
            nonlocal G_top, G2pt
            _ensure_dense_buffers()
            for Rx, Ry in coords_list:
                np.multiply(G_top, 0.5, out=G2pt); G2pt += Il_half
                chi_Ap_loc = chi_Ap[:, Rx, Ry]; chi_Bp_loc = chi_Bp[:, Rx, Ry]
                chi_Am_loc = chi_Am[:, Rx, Ry]; chi_Bm_loc = chi_Bm[:, Rx, Ry]
                _apply_deplete(chi_Ap_loc); _apply_fill(chi_Am_loc)
                _apply_deplete(chi_Bp_loc); _apply_fill(chi_Bm_loc)
                np.multiply(G2pt, 2.0, out=G_top); G_top -= Il_top
                if pbar is not None: pbar.update(1)

        def _execute_sweep(coords_list):
            coords_list = list(coords_list)
            if use_finite_support_pc:
                for Rx, Ry in coords_list:
                    payload = self._get_ow_local_support_data(Rx, Ry)
                    _apply_finite_support_pc(payload, "Ap", False)
                    _apply_finite_support_pc(payload, "Am", True)
                    _apply_finite_support_pc(payload, "Bp", False)
                    _apply_finite_support_pc(payload, "Bm", True)
                    if pbar is not None:
                        pbar.update(1)
            else:
                _execute_sweep_with_projectors(
                    coords_list,
                    chi_Ap_all,
                    chi_Bp_all,
                    chi_Am_all,
                    chi_Bm_all,
                )
            return coords_list

        def _execute_local_mode_sweep(coords_list):
            nonlocal G_top, G2pt
            _ensure_dense_buffers()
            for Rx, Ry in coords_list:
                np.multiply(G_top, 0.5, out=G2pt); G2pt += Il_half
                idx_mu1 = 0 + 2 * Rx + 2 * self.Nx * Ry
                idx_mu2 = 1 + 2 * Rx + 2 * self.Nx * Ry
                chi_mu1 = Il_top[idx_mu1]
                chi_mu2 = Il_top[idx_mu2]
                _apply_fill(chi_mu1)
                _apply_deplete(chi_mu2)
                np.multiply(G2pt, 2.0, out=G_top); G_top -= Il_top
                if pbar is not None: pbar.update(1)

        if trivial_product_state_DW:
            Nx = self.Nx
            half = Nx // 2
            w = max(1, Nx // 4)
            x0 = max(0, half - w) - 1
            x1 = min(Nx, half + w + 1)
            self.DW_loc = [int(x0), int(x1 - 1)]
            post_cycles = sweeps // 2
            mid_xs = {half, half - 1}
            mid_xs = {x for x in mid_xs if 0 <= x < Nx}
            for _ in range(sweeps):
                _execute_sweep(iter_fn())
                if G_history: history.append(G_top.copy())
            if post_cycles > 0:
                print(f"[info] Switching to local_mode=True outside slab x in [{x0}, {x1}) after {sweeps} sweep(s).")
            for _ in range(post_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if not (x0 <= Rx < x1)]
                _execute_local_mode_sweep(iter_coords)
                if G_history: history.append(G_top.copy())
            if mid_xs:
                for _ in range(post_cycles):
                    iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if Rx in mid_xs]
                    _execute_sweep(iter_coords)
                    if G_history: history.append(G_top.copy())
        elif top_last:
            Nx = self.Nx
            half = Nx // 2
            w = max(1, Nx // 4)
            x0 = max(0, half - w) - 1
            x1 = min(Nx, half + w + 1)
            phase_cycles = sweeps // 2
            for _ in range(sweeps):
                _execute_sweep(iter_fn())
                if G_history: history.append(G_top.copy())

            chi_Ap_top, chi_Bp_top, chi_Am_top, chi_Bm_top = self._compute_ow_projectors_for_alpha(self.alpha_1, self.nshell)
            chi_Ap_triv, chi_Bp_triv, chi_Am_triv, chi_Bm_triv = self._compute_ow_projectors_for_alpha(self.alpha_2, self.nshell)

            for _ in range(phase_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if not (x0 <= Rx < x1)]
                _execute_sweep_with_projectors(iter_coords, chi_Ap_triv, chi_Bp_triv, chi_Am_triv, chi_Bm_triv)
                if G_history: history.append(G_top.copy())

            for _ in range(phase_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if (x0 <= Rx <= x1)]
                _execute_sweep_with_projectors(iter_coords, chi_Ap_top, chi_Bp_top, chi_Am_top, chi_Bm_top)
                if G_history: history.append(G_top.copy())

            for _ in range(phase_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if not (x0 <= Rx < x1)]
                _execute_sweep_with_projectors(iter_coords, chi_Ap_triv, chi_Bp_triv, chi_Am_triv, chi_Bm_triv)
                if G_history: history.append(G_top.copy())

            for _ in range(phase_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if (x0 <= Rx <= x1)]
                _execute_sweep_with_projectors(iter_coords, chi_Ap_top, chi_Bp_top, chi_Am_top, chi_Bm_top)
                if G_history: history.append(G_top.copy())

            for _ in range(phase_cycles):
                iter_coords = [(Rx, Ry) for Rx, Ry in iter_fn() if not (x0 <= Rx < x1)]
                _execute_sweep_with_projectors(iter_coords, chi_Ap_triv, chi_Bp_triv, chi_Am_triv, chi_Bm_triv)
                if G_history: history.append(G_top.copy())
        else:
            for cycle in range(1, sweeps + 1):
                ordered_coords = _execute_sweep(iter_fn())
                if G_history: history.append(G_top.copy())
                _observe_cycle(cycle, ordered_coords)

        if use_bulk_extra_effective:
            for _ in range(bulk_sweeps):
                _execute_sweep(iter_bulk_fn())
                if G_history: history.append(G_top.copy())

        if pbar is not None: pbar.close()

        base_sweeps = sweeps
        final_G = G_top.copy()
        if not G_history:
            saved_path = _save_results(history_array=None, final_G=final_G, store_history=False)
            extra_cycles = 0
            if trivial_product_state_DW:
                extra_cycles = (2 * (sweeps // 2))
            elif top_last:
                extra_cycles = sweeps + (5 * (sweeps // 2))
            return {
                "G_final": final_G,
                "save_path": saved_path,
                "cycles": base_sweeps + extra_cycles + (bulk_sweeps if use_bulk_extra_effective else 0),
                "T": 1,
                "decoh": bool(decoh),
                "perfect_correction": bool(perfect_correction),
                "run_config": _run_config(False),
            }

        if not history: history.append(final_G.copy())
        G_hist = np.stack(history, axis=0)
        G_hist_avg = np.mean(G_hist, axis=0) if G_hist.shape[0] > 0 else None
        saved_path = _save_results(history_array=G_hist, final_G=final_G, store_history=True)
        return {
            "G_hist": G_hist,
            "G_hist_avg": G_hist_avg,
            "G_final": final_G,
            "cycles": base_sweeps
            + ((2 * (sweeps // 2)) if trivial_product_state_DW else 0)
            + ((sweeps + (5 * (sweeps // 2))) if top_last else 0)
            + (bulk_sweeps if use_bulk_extra_effective else 0),
            "T": G_hist.shape[0],
            "save_path": saved_path,
            "decoh": bool(decoh),
            "perfect_correction": bool(perfect_correction),
            "run_config": _run_config(True),
        }

    def run_lindblad_evolution(
        self,
        *,
        cycles=20,
        dt=0.05,
        init_mode="maxmix",
        G_init=None,
        initial_convention="correlation",
        include_number_dephasing=True,
        observation_times=None,
        representation="q0",
        progress=False,
    ):
        """Evolve the exact closed perfect-correction two-point Lindbladian.

        The production equation uses independent unit-rate lower-band gain and
        upper-band loss.  ``include_number_dephasing`` controls only the
        additional Hermitian number-jump dissipators; unlike the historical
        ``run_markov_channel(..., decoh=False)`` branch, both choices here are
        genuine continuous-time Lindblad evolutions.

        Parameters
        ----------
        cycles : float
            Final physical time in circuit-cycle units.
        dt : float
            Fixed RK4 time step.
        init_mode : {"maxmix", "empty", "filled"}
            Initial physical correlation matrix when ``G_init`` is omitted.
        G_init : array-like or None
            Optional dense initial matrix in the convention selected by
            ``initial_convention``.
        initial_convention : {"correlation", "class"}
            ``"correlation"`` means the PRR convention
            ``G_ij=Tr(rho c_i^dagger c_j)``. ``"class"`` means the legacy
            top-layer convention ``Q=2G-1`` used by ``run_markov_channel``.
        include_number_dephasing : bool
            Include the exact two-point action of all OW number jumps.
        observation_times : iterable or None
            Grid-aligned times to retain. Defaults to integer cycles from zero
            through the requested final time, including a noninteger endpoint.
        representation : {"q0", "dense"}
            Use the translation-invariant momentum-diagonal sector or the full
            dense real-space correlation matrix.
        progress : bool
            Display the RK4 integration-step progress bar.

        Returns
        -------
        dict
            Selected physical-correlation snapshots and complete run metadata.
        """
        try:
            from .diagnostics.mean_lindblad import PerfectCorrectionLindblad
        except ImportError:  # Support legacy direct imports from src/fgtn.
            from diagnostics.mean_lindblad import PerfectCorrectionLindblad

        final_time = float(cycles)
        if not np.isfinite(final_time) or final_time < 0.0:
            raise ValueError("cycles must be a finite nonnegative physical time.")
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite scalar.")

        representation = str(representation).strip().lower()
        if representation not in {"q0", "dense"}:
            raise ValueError("representation must be 'q0' or 'dense'.")
        initial_convention = str(initial_convention).strip().lower()
        if initial_convention not in {"correlation", "class"}:
            raise ValueError(
                "initial_convention must be 'correlation' or 'class'."
            )

        if observation_times is None:
            integer_times = np.arange(int(np.floor(final_time)) + 1, dtype=float)
            if integer_times.size == 0 or not np.isclose(integer_times[-1], final_time):
                observation_times = np.concatenate(
                    [integer_times, np.asarray([final_time], dtype=float)]
                )
            else:
                observation_times = integer_times
        else:
            observation_times = np.asarray(observation_times, dtype=float).reshape(-1)
            if observation_times.size == 0:
                raise ValueError("observation_times must not be empty.")
            if np.max(observation_times) > final_time + 1e-12:
                raise ValueError("observation_times cannot exceed cycles.")

        have_ow = all(
            hasattr(self, attr)
            for attr in ("WF_Ap", "WF_Am", "WF_Bp", "WF_Bm")
        )
        if not have_ow:
            self.construct_OW_projectors(
                nshell=self.nshell,
                DW=self.DW,
                trial_orbitals=self.trial_orbitals,
                dw_truncation=self.dw_truncation,
            )
        engine = PerfectCorrectionLindblad.from_canonical_model(
            self, construct_if_missing=False
        )

        dimension = int(self.Ntot // 2)
        block_dimension = int(2 * self.Nx)
        if G_init is None:
            mode = str(init_mode).strip().lower()
            if mode == "maxmix":
                occupation = 0.5
            elif mode == "empty":
                occupation = 0.0
            elif mode == "filled":
                occupation = 1.0
            else:
                raise ValueError("init_mode must be 'maxmix', 'empty', or 'filled'.")
            if representation == "q0":
                initial = np.broadcast_to(
                    occupation * np.eye(block_dimension, dtype=np.complex128),
                    (self.Ny, block_dimension, block_dimension),
                ).copy()
            else:
                initial = occupation * np.eye(dimension, dtype=np.complex128)
        else:
            initial_dense = np.asarray(G_init, dtype=np.complex128)
            if initial_dense.shape != (dimension, dimension):
                raise ValueError(
                    f"G_init must have shape ({dimension}, {dimension}); "
                    f"got {initial_dense.shape}."
                )
            if initial_convention == "class":
                initial_dense = 0.5 * (
                    initial_dense + np.eye(dimension, dtype=np.complex128)
                )
            if representation == "q0":
                momentum = engine.dense_q_sector(initial_dense, q_index=0)
                discarded = initial_dense - engine.q_sector_to_dense(
                    momentum, q_index=0
                )
                relative = float(
                    np.linalg.norm(discarded)
                    / max(np.linalg.norm(initial_dense), np.finfo(float).tiny)
                )
                if relative > 1e-10:
                    raise ValueError(
                        "representation='q0' requires a y-translation-invariant "
                        f"G_init; discarded relative norm={relative:.3e}."
                    )
                initial = momentum
            else:
                initial = initial_dense.copy()

        if representation == "q0":
            evolution = engine.integrate_q_sector(
                initial,
                q_index=0,
                dt=dt,
                observation_times=observation_times,
                include_number_dephasing=bool(include_number_dephasing),
                progress=bool(progress),
            )
        else:
            evolution = engine.integrate_dense(
                initial,
                dt=dt,
                observation_times=observation_times,
                include_number_dephasing=bool(include_number_dephasing),
                progress=bool(progress),
            )

        run_config = {
            "Nx": int(self.Nx),
            "Ny": int(self.Ny),
            "physical_time": final_time,
            "dt": dt,
            "integrator": "fixed_step_rk4",
            "perfect_correction": True,
            "gain_rate": 1.0,
            "loss_rate": 1.0,
            "number_dephasing_rate": (
                1.0 if bool(include_number_dephasing) else 0.0
            ),
            "include_number_dephasing": bool(include_number_dephasing),
            "representation": representation,
            "correlation_convention": "G_ij=Tr(rho c_i^dagger c_j)",
            "nshell": self.nshell,
            "DW": bool(self.DW),
            "DW_loc": [int(value) for value in getattr(self, "DW_loc", [])],
            "dw_interval": (
                None
                if getattr(self, "dw_interval", None) is None
                else [int(value) for value in self.dw_interval]
            ),
            "dw_truncation": bool(self.dw_truncation),
            "alpha_top": float(np.real(self.alpha_top)),
            "alpha_triv": float(np.real(self.alpha_triv)),
            "trial_orbitals": str(self.trial_orbitals),
            "canonical_dynamics_entry_point": (
                "classA_U1FGTN.run_lindblad_evolution"
            ),
        }
        return {
            "times": evolution.times,
            "steps": evolution.steps,
            "correlation_history": evolution.states,
            "correlation_final": evolution.states[-1].copy(),
            "dt": evolution.dt,
            "include_number_dephasing": bool(include_number_dephasing),
            "representation": representation,
            "run_config": run_config,
        }

    # ---------------------------- Chern observables ----------------------------

    def real_space_chern_number(self, G, *, xref=None, yref=None, radius=None):
        '''Compute the disk-partition real-space Chern number from a covariance.'''
        Nx, Ny = self.Nx, self.Ny
        Nlayer = 2 * Nx * Ny

        # Tri-partition masks inside a disk
        R = 0.4 * min(Nx, Ny) if radius is None else float(radius)
        xref = Nx // 2 if xref is None else int(xref)
        yref = Ny // 2 if yref is None else int(yref)
        if R <= 0:
            raise ValueError(f"radius must be positive; got {R}")
        if not (0 <= xref < Nx):
            raise ValueError(f"xref must satisfy 0 <= xref < {Nx}; got {xref}")
        if not (0 <= yref < Ny):
            raise ValueError(f"yref must satisfy 0 <= yref < {Ny}; got {yref}")
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
        if G.shape == (self.Ntot, self.Ntot):
            Gtt = G[:Nlayer, :Nlayer]
        elif G.shape == (Nlayer, Nlayer):
            Gtt = G
        else:
            raise ValueError(
                f"Expected top-layer shape ({Nlayer},{Nlayer}) or full "
                f"({self.Ntot},{self.Ntot}); got {G.shape}"
            )
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

    def local_chern_marker_flat(self, G, mask_outside=False, inside_mask=None, apply_tanh=True):
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
        if apply_tanh:
            C = np.tanh(np.real(C)).astype(np.float64, copy=False)
        else:
            C = np.real(C).astype(np.float64, copy=False)
        if mask_outside and inside_mask is not None:
            C = np.where(inside_mask, C, 0.0)
        return C

    def sum_chern_marker_dw_region(self, G, buffer=0, whole_region=False, whole_region_except_bndry=False):
        """
        Sum the local Chern marker (tanh off) in the DW-defined slab region.
        """
        Nx, Ny = self.Nx, self.Ny
        if whole_region and whole_region_except_bndry:
            raise ValueError("whole_region and whole_region_except_bndry are mutually exclusive.")
        if whole_region:
            inside_mask = np.ones((Nx, Ny), dtype=bool)
        elif whole_region_except_bndry:
            inside_mask = np.ones((Nx, Ny), dtype=bool)
            inside_mask[0, :] = False
            inside_mask[-1, :] = False
            inside_mask[:, 0] = False
            inside_mask[:, -1] = False
        else:
            if not (getattr(self, "DW", False) and hasattr(self, "DW_loc") and len(self.DW_loc) >= 2):
                raise ValueError("DW_loc must be set to define the DW region.")
            xL, xR = int(self.DW_loc[0]) % Nx, int(self.DW_loc[1]) % Nx
            if buffer < 0:
                raise ValueError("buffer must be >= 0.")
            if buffer > 0:
                xL = (xL + buffer) % Nx
                xR = (xR - buffer) % Nx

            inside_mask = np.zeros((Nx, Ny), dtype=bool)
            if xL <= xR:
                inside_mask[xL:xR + 1, :] = True
            else:
                inside_mask[:xR + 1, :] = True
                inside_mask[xL:, :] = True

        if not np.any(inside_mask):
            return 0.0

        C = self.local_chern_marker_flat(G, mask_outside=True, inside_mask=inside_mask, apply_tanh=False)
        return float(np.sum(C))

    # ------------------ Real-space Wilson loop / spectral flow ------------------

    def compute_spectral_flow(self, G, axis='y'):
        """
        Computes the Real-Space Wilson Loop (Spectral Flow) for a given covariance matrix G.

        Parameters
        ----------
        G : ndarray
            Covariance matrix (can be top layer or full).
        axis : str
            'y' computes the flow of Y-centers vs X-position (standard for X-domain walls).
            'x' computes the flow of X-centers vs Y-position.

        Returns
        -------
        centers : array
            The spatial center of mass (X if axis='y') for each mode.
        phases : array
            The Wilson loop phase (Y-position if axis='y') for each mode [-pi, pi].
        """
        Nx, Ny = self.Nx, self.Ny
        Nlayer = 2 * Nx * Ny

        axis = str(axis).lower()
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")

        # 1. Handle Input Shape
        G = np.asarray(G, dtype=np.complex128)
        if G.shape == (self.Ntot, self.Ntot):
            G = G[:Nlayer, :Nlayer]  # Extract top layer if full given
        elif G.shape != (Nlayer, Nlayer):
            raise ValueError(f"Expected top-layer ({Nlayer},{Nlayer}) or full ({self.Ntot},{self.Ntot}); got {G.shape}")

        # 2. Extract Occupied Subspace (Purification)
        # G has eigenvalues approx +1 (occupied) and -1 (empty).
        # We want the eigenvectors corresponding to +1.
        evals, evecs = np.linalg.eigh(G)

        # Select occupied states (evals > 0)
        # For a half-filled system, this should be exactly Nlayer // 2 states
        occ_mask = evals > 0.0
        V = evecs[:, occ_mask]  # Shape (N_sites, N_occupied)

        if V.shape[1] == 0:
            print("[Warn] No occupied states found in G.")
            return np.array([]), np.array([])

        # 3. Construct Operators
        # Coordinate grids (x fastest, y next: i = x + Nx*y)
        x_grid = np.tile(np.arange(Nx), Ny)
        y_grid = np.repeat(np.arange(Ny), Nx)

        # Expand for spinors (repeat twice)
        x_vec = np.repeat(x_grid, 2)
        y_vec = np.repeat(y_grid, 2)

        if axis == 'y':
            # We want to measure Position X, and Winding Phase Y
            # Flux Operator U_y = exp(i * 2pi * y / Ny)
            U_flux = np.exp(1j * 2 * np.pi * y_vec / Ny)
            Pos_vec = x_vec
        else:
            # We want to measure Position Y, and Winding Phase X
            # Flux Operator U_x = exp(i * 2pi * x / Nx)
            U_flux = np.exp(1j * 2 * np.pi * x_vec / Nx)
            Pos_vec = y_vec

        # 4. Projected Flux Operator (The "Coupling Matrix")
        # W = V^dag @ U_flux @ V
        # U_flux is diagonal, so we broadcast multiply
        V_twisted = V * U_flux[:, np.newaxis]
        W_small = V.conj().T @ V_twisted  # Shape (N_occ, N_occ)

        # 5. Diagonalize W_small
        # The eigenvalues contain the "Wannier Centers" in the flux direction
        w_evals, w_evecs = np.linalg.eig(W_small)

        # Phase is the position in the periodic direction (-pi to pi)
        phases = np.angle(w_evals)

        # 6. Compute Center of Mass in the other direction
        # We need expectation value <psi | Position | psi>
        # The eigenstates of W_small are 'v'. The full state is psi = V @ v.
        # <psi|X|psi> = v^dag @ (V^dag @ X @ V) @ v

        # Project Position operator
        V_weighted = V * Pos_vec[:, np.newaxis]
        X_small = V.conj().T @ V_weighted

        # Compute diag(w_evecs.H @ X_small @ w_evecs)
        X_centers = []
        for i in range(w_evecs.shape[1]):
            vec = w_evecs[:, i]
            # Expectation value (real part)
            val = np.vdot(vec, X_small @ vec)
            X_centers.append(np.real(val))

        return np.array(X_centers), phases

    def plot_spectral_flow(self, G, filename=None, title=None):
        """
        Visualizes the output of compute_spectral_flow.
        """
        centers, phases = self.compute_spectral_flow(G, axis='y')

        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        ax.scatter(centers, phases, s=10, alpha=0.6, c='blue', edgecolors='none')

        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlim(0, self.Nx)
        ax.set_xlabel(r"Spatial Position $\langle X \rangle$")
        ax.set_ylabel(r"Wannier Center $\theta_y$")
        ax.axhline(0, color='k', ls=':', alpha=0.3)

        # Mark Domain Walls if they exist
        if hasattr(self, 'DW_loc') and self.DW_loc:
            for dw in self.DW_loc:
                ax.axvline(dw, color='red', ls='--', alpha=0.5, label='DW')
            ax.legend()

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Real-Space Spectral Flow")

        plt.tight_layout()

        if filename:
            fig.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.show()
    
    # ------------------------- Gauge-invariant currents -------------------------

    def current_maps_from_hamiltonian(self, H, G, Nx=None, Ny=None):
        '''
        Build nearest-neighbour bond-current maps from a single-particle Hamiltonian and covariance.

        This uses the quadratic-fermion bond-current formula

            J_{i -> j} = 2 Im Tr[h_{ij} C_{ji}],

        where ``h_{ij}`` is the 2x2 orbital block of the one-body Hamiltonian on the
        bond from cell ``i`` to cell ``j`` and ``C = <c^\\dagger c> = (I + G) / 2`` in
        this codebase's covariance convention ``G = 2C - I``.

        Parameters
        ----------
        H : ndarray
            Top-layer one-body Hamiltonian of shape ``(2*Nx*Ny, 2*Nx*Ny)`` or full-layer
            matrix of shape ``(self.Ntot, self.Ntot)``. If full, only the top block is used.
        G : ndarray
            Top-layer covariance in the ``G = 2C - I`` convention with shape
            ``(2*Nx*Ny, 2*Nx*Ny)`` or full-layer covariance of shape ``(self.Ntot, self.Ntot)``.
            If full, only the top block is used.
        Nx, Ny : int, optional
            Lattice dimensions; defaults to the values stored on the instance.

        Returns
        -------
        J_x, J_y : ndarray
            Arrays of shape ``(Nx, Ny)`` giving currents on the +x and +y bonds.
        '''
        Nx = self.Nx if Nx is None else int(Nx)
        Ny = self.Ny if Ny is None else int(Ny)
        Nlayer = 2 * Nx * Ny

        def _coerce_top_block(arr, name):
            arr = np.asarray(arr, dtype=np.complex128)
            if arr.shape == (self.Ntot, self.Ntot):
                arr = arr[:Nlayer, :Nlayer]
            elif arr.shape != (Nlayer, Nlayer):
                raise ValueError(
                    f"Expected {name} to have shape ({Nlayer},{Nlayer}) or "
                    f"({self.Ntot},{self.Ntot}); got {arr.shape}"
                )
            return arr

        H_flat = _coerce_top_block(H, "H")
        G_flat = _coerce_top_block(G, "G")

        # Enforce Hermiticity numerically before extracting bond blocks.
        H_flat = 0.5 * (H_flat + H_flat.conj().T)
        C_flat = 0.5 * (G_flat + np.eye(Nlayer, dtype=np.complex128))
        C_flat = 0.5 * (C_flat + C_flat.conj().T)

        H6 = self._unflatten_top_to_G6(H_flat, Nx=Nx, Ny=Ny)
        C6 = self._unflatten_top_to_G6(C_flat, Nx=Nx, Ny=Ny)

        x_idx = np.arange(Nx)[:, None]
        y_idx = np.arange(Ny)[None, :]
        x_next = (x_idx + 1) % Nx
        y_next = (y_idx + 1) % Ny

        Hx = H6[x_idx, y_idx, :, x_next, y_idx, :]
        Cx = C6[x_next, y_idx, :, x_idx, y_idx, :]
        J_x = 2.0 * np.imag(np.einsum("xyab,xyba->xy", Hx, Cx, optimize=True))

        Hy = H6[x_idx, y_idx, :, x_idx, y_next, :]
        Cy = C6[x_idx, y_next, :, x_idx, y_idx, :]
        J_y = 2.0 * np.imag(np.einsum("xyab,xyba->xy", Hy, Cy, optimize=True))
        return J_x, J_y

    def current_maps_gauge_invariant(self, G, Nx=None, Ny=None):
        '''
        Build gauge-invariant bond-current maps along +x and +y from a covariance.

        Parameters
        ----------
        G : ndarray
            Top-layer covariance of shape (2*Nx*Ny, 2*Nx*Ny) or full-layer covariance
            of shape (self.Ntot, self.Ntot). The top-layer block is used if full.
        Nx, Ny : int, optional
            Lattice dimensions; defaults to the values stored on the instance.

        Returns
        -------
        J_x, J_y : ndarray
            Arrays of shape (Nx, Ny) giving currents on +x and +y bonds.
        '''
        Nx = self.Nx if Nx is None else int(Nx)
        Ny = self.Ny if Ny is None else int(Ny)
        Nlayer = 2 * Nx * Ny

        G = np.asarray(G, dtype=np.complex128)
        if G.shape == (self.Ntot, self.Ntot):
            G_flat = G[:Nlayer, :Nlayer]
        elif G.shape == (Nlayer, Nlayer):
            G_flat = G
        else:
            raise ValueError(f"Expected top-layer ({Nlayer},{Nlayer}) or full ({self.Ntot},{self.Ntot}); got {G.shape}")

        def _as_block(G_flat):
            G2 = 0.5 * (G_flat + np.eye(G_flat.shape[-1], dtype=np.complex128))
            G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order="F")
            return np.transpose(G6, (1, 2, 0, 4, 5, 3))

        block = _as_block(G_flat)
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

    def plot_current_maps(
        self,
        J_x,
        J_y,
        figsize=(18, 5),
        cmap="RdBu_r",
        quiver_cmap="plasma",
        sharey=True,
        vmin = -1,
        vmax = 1
    ):
        '''
        Plot bond-current maps J_x, J_y along with a quiver field.

        Parameters
        ----------
        J_x, J_y : ndarray
            Arrays of shape (Nx, Ny) giving currents along +x and +y bonds.
        figsize : tuple, optional
            Matplotlib figure size for the (1x3) subplot layout.
        cmap : str, optional
            Colormap for the J_x/J_y heatmaps.
        quiver_cmap : str, optional
            Colormap mapping |J| magnitudes in the quiver plot.

        Returns
        -------
        fig, axes : matplotlib Figure and Axes array
            Figure and axes handles for further customization.
        '''
        J_x = np.asarray(J_x, dtype=float)
        J_y = np.asarray(J_y, dtype=float)
        if J_x.shape != J_y.shape:
            raise ValueError(f"J_x shape {J_x.shape} does not match J_y shape {J_y.shape}")

        Nx, Ny = J_x.shape
        extent = (-0.5, Ny - 0.5, -0.5, Nx - 0.5)

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=sharey, constrained_layout=True)

        for ax, (title, data) in zip(axes[:2], [("J_x", J_x), ("J_y", J_y)]):
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                extent=extent,
                vmin = vmin,
                vmax = vmax
            )
            ax.set_title(title)
            ax.set_xlabel(r"$y$")
            ax.set_ylabel(r"$x$")
            fig.colorbar(im, ax=ax, shrink=0.85)

        y_coords, x_coords = np.meshgrid(np.arange(Ny), np.arange(Nx))
        magnitude = np.hypot(J_x, J_y)
        quiv = axes[2].quiver(
            y_coords,
            x_coords,
            J_y,
            J_x,
            magnitude,
            cmap=quiver_cmap,
            angles="xy",
            scale_units="xy",
            scale=None,
            pivot="mid",
        )
        axes[2].set_xlim(extent[0], extent[1])
        axes[2].set_ylim(extent[2], extent[3])
        axes[2].set_xlabel(r"$y$")
        axes[2].set_ylabel(r"$x$")
        axes[2].set_title("Current field $(J_x, J_y)$")
        axes[2].set_aspect("equal")
        fig.colorbar(quiv, ax=axes[2], shrink=0.85, label="|J|")

        return fig, axes

    # -------------------- Translation checks / reshape helpers --------------------

    def _unflatten_top_to_G6(self, G_flat, Nx=None, Ny=None):
        """Reshape a flattened top-layer covariance into (Nx,Ny,2, Nx,Ny,2)."""
        Nx = self.Nx if Nx is None else int(Nx)
        Ny = self.Ny if Ny is None else int(Ny)
        Nlayer = 2 * Nx * Ny
        G_flat = np.asarray(G_flat, dtype=np.complex128)
        if G_flat.shape != (Nlayer, Nlayer):
            raise ValueError(f"Expected flattened top-layer shape ({Nlayer},{Nlayer}); got {G_flat.shape}")
        G6m = G_flat.reshape((2, Nx, Ny, 2, Nx, Ny), order="F")
        return np.transpose(G6m, (1, 2, 0, 4, 5, 3))

    def _flatten_G6_top(self, G6):
        """Flatten G6 with indices (x,y,μ; x',y',ν) to (i,j) with i=μ+2x+2Nx*y (Fortran)."""
        G6 = np.asarray(G6, dtype=np.complex128)
        Nx, Ny, s1, Nx2, Ny2, s2 = G6.shape
        if not ((s1, s2) == (2, 2) and (Nx, Ny) == (Nx2, Ny2)):
            raise ValueError(f"G must have shape (Nx, Ny, 2, Nx, Ny, 2); got {G6.shape}")
        G6m = np.transpose(G6, (2, 0, 1, 5, 3, 4))  # (2,Nx,Ny, 2,Nx,Ny)
        return G6m.reshape(2 * Nx * Ny, 2 * Nx * Ny, order="F")

    def check_y_translation_invariance(self, G, *, Nx=None, Ny=None, tol_shift=1e-10, tol_offdiag=1e-10, hermitize=True):
        """
        Check y-translation invariance and the block structure after the y/y' unitary FFT.

        G can be:
          - G6 of shape (Nx,Ny,2,Nx,Ny,2)
          - flattened top-layer of shape (2*Nx*Ny, 2*Nx*Ny)
          - full-layer of shape (self.Ntot, self.Ntot) (top block is used)
        """
        Nx = self.Nx if Nx is None else int(Nx)
        Ny = self.Ny if Ny is None else int(Ny)
        Nlayer = 2 * Nx * Ny

        G_arr = np.asarray(G, dtype=np.complex128)
        if G_arr.ndim == 2:
            if G_arr.shape == (self.Ntot, self.Ntot):
                G_arr = G_arr[:Nlayer, :Nlayer]
            if G_arr.shape != (Nlayer, Nlayer):
                return {"shape_ok": False}
            G6 = self._unflatten_top_to_G6(G_arr, Nx=Nx, Ny=Ny)
        elif G_arr.ndim == 6:
            G6 = G_arr
            if G6.shape != (Nx, Ny, 2, Nx, Ny, 2):
                return {"shape_ok": False}
        else:
            return {"shape_ok": False}

        if hermitize:
            G6 = 0.5 * (G6 + np.transpose(G6.conj(), (3, 4, 5, 0, 1, 2)))

        # 1) shift invariance along y
        Gy = G6
        Gy_p = np.roll(np.roll(G6, shift=+1, axis=1), shift=+1, axis=4)
        shift_max_abs_diff = float(np.max(np.abs(Gy - Gy_p)))
        shift_pass = (shift_max_abs_diff <= tol_shift)

        # 2) ky block off-diagonal check
        Gk = np.fft.fft(G6, axis=1, norm="ortho")
        Gk = np.fft.ifft(Gk, axis=4, norm="ortho")

        diag_mask = np.eye(Ny, dtype=bool)
        off_mask = ~diag_mask
        block_norms = np.sum(np.abs(Gk) ** 2, axis=(0, 2, 3, 5))
        den = float(block_norms.sum())
        num = float(block_norms[off_mask].sum())
        offdiag_ratio = 0.0 if den == 0 else float(num / den)
        offdiag_pass = (offdiag_ratio <= tol_offdiag)

        return {
            "shape_ok": True,
            "Ny": Ny,
            "shift_max_abs_diff": shift_max_abs_diff,
            "shift_pass": bool(shift_pass),
            "offdiag_ratio": offdiag_ratio,
            "offdiag_pass": bool(offdiag_pass),
        }

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
        child.alpha_1 = self.alpha_1
        child.alpha_2 = self.alpha_2
        child.alpha_top = self.alpha_top
        child.alpha_triv = self.alpha_triv
        child.trial_orbitals = self.trial_orbitals
        child.dw_truncation = self.dw_truncation
        child.dw_interval = getattr(self, "dw_interval", None)
        child.twist_x = float(getattr(self, "twist_x", 0.0))
        child.twist_y = float(getattr(self, "twist_y", 0.0))
        child.filling_frac = self.filling_frac
        child.time_init = self.time_init
        child.DW_loc = getattr(self, "DW_loc", None)
        child._physical_covariance_update_mode = getattr(
            self, "_physical_covariance_update_mode", "rank1"
        )
    
        # ---- share big, read-only arrays (do NOT mutate these in workers) ----
        # NOTE: we intentionally DO NOT copy; workers should treat these as read-only.
        child.Pminus = self.Pminus
        child.Pplus  = self.Pplus
        child.WF_Ap  = self.WF_Ap
        child.WF_Bp  = self.WF_Bp
        child.WF_Am  = self.WF_Am
        child.WF_Bm  = self.WF_Bm
        child.alpha_profile = getattr(self, "alpha_profile", None)
        child.alpha  = getattr(self, "alpha", child.alpha_profile)
    
        # ---- per-worker mutable state ----
        # fresh starting state for each worker (top-layer randomized in your __init__)
        child.G0 = None if self.G0 is None else np.array(self.G0, dtype=np.complex128, copy=True)
        child.G = None
        child.G_list = []
        child.g2_flags = []
        child._I_top_cache = None
        child._local_basis_cache = {}
        child._local_mode_ops_cache = {}
        child._ow_local_cache = {}
        child._ow_local_support_cache = {}
    
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



    def entanglement_contour(self, Gtt, Nx, Ny):
        '''Compute the entanglement contour s(r) for the top-layer covariance.'''
        arr = np.asarray(Gtt, dtype=np.complex128)
        if arr.ndim != 2:
            raise ValueError(f"entanglement_contour expects a 2D covariance; got shape {arr.shape}")

        Nlayer = arr.shape[0]
        I  = np.eye(Nlayer, dtype=np.complex128)
        G2 = 0.5 * (I + arr)

        evals, vecs = np.linalg.eigh(G2)
        evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
        f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))

        # diag(F) = sum_k f_eigs[k] * |vecs[i,k]|^2
        diagF = (np.abs(vecs) ** 2) @ f_eigs
        diagF = diagF.real

        diagF = diagF.reshape(2, Nx, Ny, order="F")
        return diagF.sum(axis=0)                      # sum over μ -> (Nx,Ny)

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
                contours[s, t] = self.entanglement_contour(top_histories[s, t], self.Nx, self.Ny)
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
   
