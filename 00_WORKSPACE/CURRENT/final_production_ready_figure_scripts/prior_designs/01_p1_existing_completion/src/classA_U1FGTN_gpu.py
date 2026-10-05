import hashlib
import json
import os
import sys

import numpy as np
from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency guard
    torch = None

if torch is not None:
    try:
        from .occupied_frame_gpu import (
            BatchedOccupiedFrameState,
            GPU_FRAME_ALGORITHM_VERSION,
        )
    except ImportError:  # Support direct imports from src/fgtn on PYTHONPATH.
        from occupied_frame_gpu import (  # type: ignore
            BatchedOccupiedFrameState,
            GPU_FRAME_ALGORITHM_VERSION,
        )
else:  # pragma: no cover - exercised only in CPU-only environments.
    BatchedOccupiedFrameState = None
    GPU_FRAME_ALGORITHM_VERSION = "padded_append_householder_qr_v2"


class classA_U1FGTN_gpu:
    """
    GPU-focused implementation of the Markov circuit only.

    This class intentionally keeps a narrow surface area: lattice setup,
    OW spinor construction, top-layer Markov/post-selection updates, and
    saving history/final-state data from run_markov_circuit.
    """

    @staticmethod
    def _normalize_state_representation(representation):
        value = str(representation).strip().lower()
        aliases = {
            "auto": "auto",
            "covariance": "covariance",
            "physical_frame": "physical_frame",
            "pure_frame": "physical_frame",
        }
        if value not in aliases:
            raise ValueError(
                "state_representation must be 'auto', 'covariance', or "
                "'physical_frame'."
            )
        return aliases[value]

    def clip_covariance_spectrum(self, G, *, sample_chunk=10, max_correction=1e-6):
        """Project centered covariance eigenvalues to [-1,1], in place.

        This opt-in numerical intervention preserves interior occupations; it is
        not pure-state projection or entrywise clipping. Diagnostics refer to
        C=(G+I)/2, except ``clip_covariance_frobenius`` which refers to G.
        No random numbers are consumed. Large/nonfinite violations still fail.
        """
        if G.ndim != 3 or G.shape[-1] != G.shape[-2] or G.dtype != torch.complex128:
            raise ValueError("spectral clipping requires batched complex128 covariance")
        if int(sample_chunk) <= 0 or not np.isfinite(max_correction) or max_correction <= 0:
            raise ValueError("positive sample_chunk and finite positive max_correction required")
        names = ("clip_pre_min", "clip_pre_max", "clip_max_correction",
                 "clip_covariance_frobenius", "clip_charge_change", "clip_mode_count")
        diagnostics = {name: [] for name in names}
        with torch.inference_mode():
            for start in range(0, G.shape[0], int(sample_chunk)):
                raw = G[start:start + int(sample_chunk)]
                if not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError("spectral clipping: nonfinite covariance")
                if float((raw - raw.mH).abs().max()) > 1e-9:
                    raise FloatingPointError("spectral clipping: Hermiticity residual > 1e-9")
                hermitian = (raw + raw.mH) * .5
                eig, U = torch.linalg.eigh(hermitian)
                delta = eig.clamp(-1., 1.) - eig
                correction = delta.abs().amax(-1) * .5
                if not bool(torch.isfinite(eig).all()) or float(correction.max()) > max_correction:
                    raise FloatingPointError(
                        "spectral clipping requires excessive occupation correction: "
                        f"{float(correction.max()):.17g} > {max_correction:.17g}; state not accepted"
                    )
                # Add only the out-of-range spectral correction. Leave already
                # physical samples bitwise untouched, avoiding needless rebuilds.
                mask = correction > 0
                if bool(mask.any()):
                    u = U[mask]
                    updated = hermitian[mask] + (u * delta[mask].unsqueeze(-2)) @ u.mH
                    raw[mask] = (updated + updated.mH) * .5
                values = ((eig.amin(-1) + 1) * .5, (eig.amax(-1) + 1) * .5,
                          correction, torch.linalg.vector_norm(delta, dim=-1),
                          delta.sum(-1) * .5, (delta != 0).sum(-1))
                for name, value in zip(names, values):
                    diagnostics[name].append(value.detach().cpu().numpy())
        return {name: np.concatenate(rows) for name, rows in diagnostics.items()}

    def _centered_purity_defect(self, centered_covariance):
        centered = torch.as_tensor(
            centered_covariance, dtype=self.dtype, device=self.device
        )
        if centered.ndim == 2:
            centered = centered.unsqueeze(0)
        if centered.ndim != 3 or centered.shape[-1] != centered.shape[-2]:
            raise ValueError("G_init must have shape (N,N) or (samples,N,N).")
        centered = 0.5 * (centered + centered.mH)
        identity = torch.eye(
            centered.shape[-1], dtype=self.dtype, device=self.device
        )
        occupations = torch.linalg.eigvalsh(0.5 * (centered + identity))
        range_defect = torch.maximum(
            (-occupations.amin(dim=1)).clamp_min(0.0),
            (occupations.amax(dim=1) - 1.0).clamp_min(0.0),
        )
        idempotency_defect = torch.minimum(
            occupations.abs(), (1.0 - occupations).abs()
        ).amax(dim=1)
        return torch.maximum(range_defect, idempotency_defect)

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
            return torch.ones_like(dxw, dtype=torch.bool)
        if np.isclose(nsh, 0.5):
            return (
                ((dxw == 0) & (dyw == 0))
                | ((dxw.abs() == 1) & (dyw == 0))
                | ((dxw == 0) & (dyw.abs() == 1))
            )
        return (dxw.abs() <= int(nsh)) & (dyw.abs() <= int(nsh))

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
        n_a_eff = cls._validate_probability(n_a, "n_a")
        p_gain_eff = n_a_eff if p_gain is None else cls._validate_probability(p_gain, "p_gain")
        p_loss_eff = (1.0 - n_a_eff) if p_loss is None else cls._validate_probability(p_loss, "p_loss")
        return n_a_eff, p_gain_eff, p_loss_eff

    def __init__(
        self,
        Nx,
        Ny,
        DW=True,
        nshell=None,
        filling_frac=0.5,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=False,
        triv_region_local_mode=False,
        device=None,
        dtype="complex128",
        backend="auto",
    ):
        if torch is None:
            raise ImportError(
                "classA_U1FGTN_gpu requires PyTorch. Install torch in the active environment first."
            )

        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Ntot = 4 * self.Nx * self.Ny
        self.Nlayer = self.Ntot // 2
        self.nshell = self._normalize_nshell(nshell)
        self.filling_frac = float(filling_frac)
        self.DW = bool(DW)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_top = self.alpha_1
        self.alpha_triv = self.alpha_2
        self.trial_orbitals = str(trial_orbitals)
        self.dw_truncation = bool(dw_truncation)
        self.triv_region_local_mode = bool(triv_region_local_mode)

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_complex_dtype(dtype)
        self.real_dtype = torch.float64 if self.dtype == torch.complex128 else torch.float32
        self.numpy_dtype = np.complex128 if self.dtype == torch.complex128 else np.complex64
        self.backend = self._resolve_backend(backend)
        self._eye_top = torch.eye(self.Nlayer, dtype=self.dtype, device=self.device)
        self._all_top_indices = torch.arange(self.Nlayer, dtype=torch.long, device=self.device)
        self._eye_cache = {self.Nlayer: self._eye_top}
        self._local_site_cache = None

        if self.DW:
            self.create_domain_wall(alpha_1=self.alpha_1, alpha_2=self.alpha_2)
        else:
            self.alpha_profile = np.full((self.Nx, self.Ny), float(self.alpha_1), dtype=np.complex128)
            self.alpha = self.alpha_profile

        self.construct_OW_projectors(
            nshell=self.nshell,
            DW=self.DW,
            trial_orbitals=self.trial_orbitals,
            dw_truncation=self.dw_truncation,
        )
        self._site_local_mode_mask = torch.as_tensor(
            [self._site_uses_local_mode(site) for site in range(self.Nx * self.Ny)],
            dtype=torch.bool,
            device=self.device,
        )

    @staticmethod
    def _resolve_device(device):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False in this runtime.")
        return resolved

    @staticmethod
    def _resolve_complex_dtype(dtype):
        if dtype in (torch.complex64, "complex64", "c64"):
            return torch.complex64
        if dtype in (torch.complex128, "complex128", "c128"):
            return torch.complex128
        raise ValueError("dtype must be one of: 'complex64', 'complex128', torch.complex64, torch.complex128")

    def _resolve_backend(self, backend):
        mode = str(backend).lower()
        allowed = ("auto", "dense", "local")
        if mode not in allowed:
            raise ValueError(f"backend must be one of: {', '.join(allowed)}")
        if mode == "auto":
            return "local" if self.nshell is not None else "dense"
        if mode == "local" and self.nshell is None:
            raise ValueError("backend='local' requires a finite nshell.")
        return mode

    def _eye_of_size(self, size):
        size = int(size)
        eye = self._eye_cache.get(size)
        if eye is None:
            eye = torch.eye(size, dtype=self.dtype, device=self.device)
            self._eye_cache[size] = eye
        return eye

    def _complement_indices(self, support_idx):
        mask = torch.ones(self.Nlayer, dtype=torch.bool, device=self.device)
        mask[support_idx] = False
        return self._all_top_indices[mask]

    def _ensure_outdir(self, path):
        os.makedirs(path, exist_ok=True)
        return path

    def _g_history_outdir(self):
        return self._ensure_outdir(os.path.join("cache", "G_history_samples", f"N{self.Nx}x{self.Ny}"))

    def _g_history_outdir_rel(self):
        return os.path.join("cache", "G_history_samples", f"N{self.Nx}x{self.Ny}")

    def _gpu_cache_root(self, save_dir=None):
        root = "GPU_cache" if save_dir is None else save_dir
        return self._ensure_outdir(root)

    @staticmethod
    def _sanitize_path_component(value):
        if value is None:
            return ""
        text = str(value)
        safe = [ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text]
        return "".join(safe).strip("._")

    def _array_signature(self, value):
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.ascontiguousarray(arr)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha1": hashlib.sha1(arr.view(np.uint8).tobytes()).hexdigest()[:16],
        }

    @staticmethod
    def _normalize_snapshot_cycles(snapshot_cycles, cycles):
        if snapshot_cycles is None:
            return None
        values = [int(cyc) for cyc in snapshot_cycles]
        if not values:
            raise ValueError("snapshot_cycles must contain at least one cycle.")
        if values != sorted(set(values)):
            raise ValueError("snapshot_cycles must be sorted and unique.")
        if values[0] <= 0 or values[-1] > int(cycles):
            raise ValueError("snapshot_cycles entries must be in the range 1..cycles.")
        return values

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

    def _auto_batch_size_for_a100_40gb(
        self,
        *,
        samples,
        cycles,
        store_mode,
        snapshot_cycles=None,
        init_mode="default",
        state_representation="covariance",
        track_choi=False,
        choi_nlayer=None,
    ):
        target_total_bytes = 40 * 1024**3
        usable_fraction = 0.72
        free_memory_fraction = 0.85
        reserve_bytes = 512 * 1024**2

        total_bytes = target_total_bytes
        free_bytes = target_total_bytes
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                with torch.cuda.device(self.device):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                total_bytes = target_total_bytes
                free_bytes = target_total_bytes

        a100_capped_total = min(int(total_bytes), target_total_bytes)
        budget_bytes = min(
            int(a100_capped_total * usable_fraction),
            int(free_bytes * free_memory_fraction),
        )
        budget_bytes = max(0, budget_bytes - reserve_bytes)

        matrix_bytes = int(self.Nlayer) * int(self.Nlayer) * np.dtype(self.numpy_dtype).itemsize
        static_matrix_factor = 4.0 if self.backend == "local" else 6.0
        static_bytes = int(static_matrix_factor * matrix_bytes)
        if str(state_representation) == "physical_frame":
            rank = max(0, min(self.Nlayer, int(round(self.filling_frac * self.Nlayer))))
            capacity = min(self.Nlayer, max(rank, ((rank + 7) // 8) * 8))
            native_frame_bytes = (
                self.Nlayer * capacity * np.dtype(self.numpy_dtype).itemsize
                + np.dtype(np.int64).itemsize
            )
            # Live frame, QR/Householder workspaces, channel contractions, and
            # observer overlap buffers.  This remains conservative while retaining
            # the measured O(Nk), rather than O(N^2), per-trajectory scaling.
            per_sample_bytes = int(8.0 * native_frame_bytes)
            if store_mode == "history":
                per_sample_bytes += int((cycles + 1) * matrix_bytes)
            elif store_mode == "snapshots":
                per_sample_bytes += int(len(snapshot_cycles or ()) * matrix_bytes)
        else:
            per_sample_matrix_factor = 12.0 if self.backend == "local" else 20.0
            if str(init_mode).lower() == "default":
                per_sample_matrix_factor += 4.0
            if store_mode == "history":
                per_sample_matrix_factor += 1.0
            elif store_mode == "snapshots":
                per_sample_matrix_factor += min(2.0, 0.25 * len(snapshot_cycles or ()))
            per_sample_bytes = int(per_sample_matrix_factor * matrix_bytes)
        if track_choi:
            choi_dim = self.Nlayer if choi_nlayer is None else int(choi_nlayer)
            choi_matrix_bytes = choi_dim * choi_dim * np.dtype(self.numpy_dtype).itemsize
            per_sample_bytes += 3 * choi_matrix_bytes
        per_sample_bytes = max(1, per_sample_bytes)
        available_for_samples = max(0, budget_bytes - static_bytes)
        selected = max(1, min(int(samples), available_for_samples // per_sample_bytes))

        info = {
            "strategy": "a100_40gb_estimate",
            "selected_batch_size": int(selected),
            "samples": int(samples),
            "cycles": int(cycles),
            "store_mode": str(store_mode),
            "snapshot_cycles": None if snapshot_cycles is None else [int(c) for c in snapshot_cycles],
            "backend": str(self.backend),
            "dtype": "complex64" if self.dtype == torch.complex64 else "complex128",
            "Nlayer": int(self.Nlayer),
            "state_representation": str(state_representation),
            "track_choi": bool(track_choi),
            "choi_nlayer": None if not track_choi else int(self.Nlayer if choi_nlayer is None else choi_nlayer),
            "target_total_gib": target_total_bytes / 1024**3,
            "detected_total_gib": int(total_bytes) / 1024**3,
            "detected_free_gib": int(free_bytes) / 1024**3,
            "budget_gib": budget_bytes / 1024**3,
            "estimated_static_gib": static_bytes / 1024**3,
            "estimated_per_sample_gib": per_sample_bytes / 1024**3,
            "usable_fraction": float(usable_fraction),
            "free_memory_fraction": float(free_memory_fraction),
            "reserve_gib": reserve_bytes / 1024**3,
        }
        return int(selected), info

    @staticmethod
    def _write_json_atomic(path, payload):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    @staticmethod
    def _save_npy_atomic(path, array):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as fh:
            np.save(fh, np.asarray(array), allow_pickle=False)
        os.replace(tmp_path, path)

    def create_domain_wall(self, alpha_1, alpha_2):
        alpha = np.full((self.Nx, self.Ny), float(alpha_2), dtype=np.complex128)
        half = self.Nx // 2
        w = max(1, self.Nx // 4)
        x0 = max(0, half - w)
        x1 = min(self.Nx, half + w + 1)
        alpha[x0:x1, :] = float(alpha_1)
        self.DW_loc = [int(x0), int(x1 - 1)]
        self.alpha_profile = alpha
        self.alpha = self.alpha_profile

    def _topological_region_mask(self):
        mask = np.zeros((self.Nx, self.Ny), dtype=bool)
        if self.DW and hasattr(self, "DW_loc") and len(self.DW_loc) == 2:
            xL = int(self.DW_loc[0]) % self.Nx
            xR = int(self.DW_loc[1]) % self.Nx
            if xL <= xR:
                mask[xL:xR + 1, :] = True
            else:
                mask[xL:, :] = True
                mask[:xR + 1, :] = True
        else:
            mask[:, :] = np.isclose(np.real(self.alpha_profile), float(self.alpha_1))
        return mask

    def _meas_slab_only_effective(self, meas_slab_only):
        return bool(meas_slab_only and self.DW and self.dw_truncation)

    def active_top_layer_indices(self, meas_slab_only=True):
        """Return full-space canonical indices used by adaptive cycle updates."""
        if not self._meas_slab_only_effective(meas_slab_only):
            return self._all_top_indices.clone()
        topo = self._topological_region_mask()
        indices = [
            mu + 2 * x + 2 * self.Nx * y
            for y in range(self.Ny)
            for x in range(self.Nx)
            if bool(topo[x, y])
            for mu in (0, 1)
        ]
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)

    def _exterior_site_ids(self):
        topo = self._topological_region_mask()
        return [
            int(x) + self.Nx * int(y)
            for x in range(self.Nx)
            for y in range(self.Ny)
            if not bool(topo[x, y])
        ]

    def _site_uses_local_mode(self, site_id):
        if not (self.triv_region_local_mode and self.DW):
            return False
        x = int(site_id) % self.Nx
        y = int(site_id) // self.Nx
        return not bool(self._topological_region_mask()[x, y])

    def _canonical_site_vectors(self, site_id):
        x = int(site_id) % self.Nx
        y = int(site_id) // self.Nx
        idx_a = 2 * x + 2 * self.Nx * y
        idx_b = idx_a + 1
        e_a = torch.zeros((self.Nlayer,), dtype=self.dtype, device=self.device)
        e_b = torch.zeros((self.Nlayer,), dtype=self.dtype, device=self.device)
        e_a[idx_a] = 1.0
        e_b[idx_b] = 1.0
        return e_a, e_b, idx_a, idx_b

    def construct_OW_projectors(self, nshell, DW, trial_orbitals="X", dw_truncation=False):
        Nx, Ny = self.Nx, self.Ny
        trial_orbitals = str(trial_orbitals).upper()
        self.nshell = self._normalize_nshell(nshell)
        self.DW = bool(DW)
        self.trial_orbitals = trial_orbitals
        self.dw_truncation = bool(dw_truncation)
        self._local_site_cache = None
        if not DW:
            self.alpha_profile = self.alpha_1 * np.ones((Nx, Ny), dtype=np.complex128)
        self.alpha = self.alpha_profile

        alpha = torch.as_tensor(np.real(self.alpha_profile), dtype=self.real_dtype, device=self.device)
        kx = 2 * np.pi * torch.fft.fftfreq(Nx, device=self.device, dtype=self.real_dtype)
        ky = 2 * np.pi * torch.fft.fftfreq(Ny, device=self.device, dtype=self.real_dtype)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")

        nx = torch.sin(KX)[:, :, None, None]
        ny = torch.sin(KY)[:, :, None, None]
        nz = alpha[None, None, :, :] - torch.cos(KX)[:, :, None, None] - torch.cos(KY)[:, :, None, None]
        nmag = torch.sqrt(nx * nx + ny * ny + nz * nz).clamp_min(1e-15)

        sx = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        sy = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        I2 = torch.eye(2, dtype=self.dtype, device=self.device)

        hk = (
            nx[..., None, None].to(self.dtype) * sx
            + ny[..., None, None].to(self.dtype) * sy
            + nz[..., None, None].to(self.dtype) * sz
        ) / nmag[..., None, None].to(self.dtype)

        Pminus = 0.5 * (I2 - hk)
        Pplus = 0.5 * (I2 + hk)

        if trial_orbitals == "X":
            tauA = (1 / np.sqrt(2)) * torch.tensor([[1], [1]], dtype=self.dtype, device=self.device)
            tauB = (1 / np.sqrt(2)) * torch.tensor([[1], [-1]], dtype=self.dtype, device=self.device)
        elif trial_orbitals == "Y":
            tauA = (1 / np.sqrt(2)) * torch.tensor([[1], [1j]], dtype=self.dtype, device=self.device)
            tauB = (1 / np.sqrt(2)) * torch.tensor([[1], [-1j]], dtype=self.dtype, device=self.device)
        elif trial_orbitals == "Z":
            tauA = torch.tensor([[1], [0]], dtype=self.dtype, device=self.device)
            tauB = torch.tensor([[0], [1]], dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"trial_orbitals must be 'X', 'Y', or 'Z', got {trial_orbitals!r}")

        Rx_grid = torch.arange(Nx, device=self.device, dtype=self.real_dtype)
        Ry_grid = torch.arange(Ny, device=self.device, dtype=self.real_dtype)
        phase_x = torch.exp(1j * (KX[..., None, None] * Rx_grid[None, None, :, None]))
        phase_y = torch.exp(1j * (KY[..., None, None] * Ry_grid[None, None, None, :]))
        phase = phase_x * phase_y

        dw_region_mask = None
        dw_center_mask = None
        if DW and dw_truncation:
            if not (hasattr(self, "DW_loc") and isinstance(self.DW_loc, (list, tuple)) and len(self.DW_loc) == 2):
                raise ValueError("dw_truncation requires DW=True with valid DW_loc.")
            xL = int(self.DW_loc[0]) % Nx
            xR = int(self.DW_loc[1]) % Nx
            topo_x = torch.zeros(Nx, dtype=torch.bool, device=self.device)
            if xL <= xR:
                topo_x[xL:xR + 1] = True
            else:
                topo_x[xL:] = True
                topo_x[:xR + 1] = True
            topo_region = topo_x[:, None].expand(Nx, Ny)
            topo_center = topo_x[:, None].expand(Nx, Ny)
            dw_region_mask = topo_region[:, :, None, None, None]
            dw_center_mask = topo_center[None, None, None, :, :]

        def make_W(Pband, tau):
            tau_dag = tau[:, 0].conj()
            psi_k = torch.einsum("m,...mn->...n", tau_dag, Pband)

            F0 = phase * psi_k[..., 0]
            F1 = phase * psi_k[..., 1]
            W0 = torch.fft.fft2(F0, dim=(0, 1))
            W1 = torch.fft.fft2(F1, dim=(0, 1))
            W = torch.movedim(torch.stack([W0, W1], dim=-1), -1, 2)

            if nshell is not None:
                x = torch.arange(Nx, device=self.device)[:, None, None, None]
                y = torch.arange(Ny, device=self.device)[None, :, None, None]
                Rx = torch.arange(Nx, device=self.device)[None, None, :, None]
                Ry = torch.arange(Ny, device=self.device)[None, None, None, :]
                dxw = ((x - Rx + Nx // 2) % Nx) - Nx // 2
                dyw = ((y - Ry + Ny // 2) % Ny) - Ny // 2
                mask = self._ow_support_mask(dxw, dyw, nshell)[:, :, None, :, :]
                W = W * mask.to(self.dtype)
            if dw_region_mask is not None:
                region_mask = torch.where(dw_center_mask, dw_region_mask, ~dw_region_mask)
                W = W * region_mask.to(self.dtype)

            denom = torch.sqrt(torch.sum(torch.abs(W) ** 2, dim=(0, 1, 2), keepdim=True)).clamp_min(1e-15)
            return W / denom

        def flatten_centers(W):
            return W.permute(1, 0, 2, 3, 4).contiguous().reshape(2 * Nx * Ny, Nx, Ny)

        self.WF_Ap = flatten_centers(make_W(Pplus, tauA)).contiguous()
        self.WF_Bp = flatten_centers(make_W(Pplus, tauB)).contiguous()
        self.WF_Am = flatten_centers(make_W(Pminus, tauA)).contiguous()
        self.WF_Bm = flatten_centers(make_W(Pminus, tauB)).contiguous()

        self.WF_Ap_sites = self.WF_Ap.permute(2, 1, 0).contiguous().reshape(Nx * Ny, self.Nlayer)
        self.WF_Bp_sites = self.WF_Bp.permute(2, 1, 0).contiguous().reshape(Nx * Ny, self.Nlayer)
        self.WF_Am_sites = self.WF_Am.permute(2, 1, 0).contiguous().reshape(Nx * Ny, self.Nlayer)
        self.WF_Bm_sites = self.WF_Bm.permute(2, 1, 0).contiguous().reshape(Nx * Ny, self.Nlayer)

        if self.backend == "local":
            self._build_local_site_cache()
        print(
            "[info] constructed Wannier projectors "
            f"nshell={self.nshell} "
            f"dw_truncation={bool(self.dw_truncation)} "
            f"backend={self.backend} "
            f"device={self.device} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def _build_local_site_cache(self):
        cache = []
        channel_maps = {
            "Ap": self.WF_Ap_sites,
            "Bp": self.WF_Bp_sites,
            "Am": self.WF_Am_sites,
            "Bm": self.WF_Bm_sites,
        }
        for site_id in range(self.Nx * self.Ny):
            if self._site_uses_local_mode(site_id):
                _, _, idx_a, idx_b = self._canonical_site_vectors(site_id)
                support_idx = torch.as_tensor([idx_a, idx_b], dtype=torch.long, device=self.device)
                comp_idx = self._complement_indices(support_idx)
                cache.append(
                    {
                        "mode": "local_mode",
                        "idx": support_idx,
                        "comp": comp_idx,
                        "A": torch.tensor([1.0, 0.0], dtype=self.dtype, device=self.device),
                        "B": torch.tensor([0.0, 1.0], dtype=self.dtype, device=self.device),
                    }
                )
                continue
            support_mask = torch.zeros(self.Nlayer, dtype=torch.bool, device=self.device)
            for arr in channel_maps.values():
                support_mask |= arr[site_id].abs() > 0
            support_idx = torch.nonzero(support_mask, as_tuple=False).flatten()
            if support_idx.numel() == 0:
                raise ValueError(f"Local backend found empty support at site_id={site_id}.")
            comp_idx = self._complement_indices(support_idx)
            cache.append(
                {
                    "mode": "ow",
                    "idx": support_idx,
                    "comp": comp_idx,
                    "Ap": self.WF_Ap_sites[site_id].index_select(0, support_idx).contiguous(),
                    "Bp": self.WF_Bp_sites[site_id].index_select(0, support_idx).contiguous(),
                    "Am": self.WF_Am_sites[site_id].index_select(0, support_idx).contiguous(),
                    "Bm": self.WF_Bm_sites[site_id].index_select(0, support_idx).contiguous(),
                }
            )
        self._local_site_cache = cache

    def _controller_twist_transport(self, *, gauge, seam_y):
        """Return component transport from each controller center around periodic y."""
        gauge = str(gauge).strip().lower()
        if gauge not in ("seam", "uniform"):
            raise ValueError("twist gauge must be 'seam' or 'uniform'.")
        seam_y = (self.Ny - 1) if seam_y is None else int(seam_y) % self.Ny
        site_ids = torch.arange(
            self.Nx * self.Ny, dtype=torch.long, device=self.device
        )
        center_y = site_ids // self.Nx
        mode_ids = torch.arange(
            self.Nlayer, dtype=torch.long, device=self.device
        )
        component_y = mode_ids // (2 * self.Nx)
        dy = (
            (component_y[None, :] - center_y[:, None] + self.Ny // 2)
            % self.Ny
        ) - self.Ny // 2
        if gauge == "uniform":
            return dy.to(self.real_dtype) / float(self.Ny)
        cut_y = (seam_y + 1) % self.Ny
        center_from_cut = (center_y - cut_y) % self.Ny
        unwrapped = center_from_cut[:, None] + dy
        return (
            (unwrapped >= self.Ny).to(self.real_dtype)
            - (unwrapped < 0).to(self.real_dtype)
        )

    def set_controller_twist(self, phi, *, gauge="seam", seam_y=None):
        """Insert a y-holonomy into every controller orbital.

        The constructor has already formed the complete periodic controller frame before
        applying its support mask.  Multiplying every surviving component by its parallel-
        transport phase therefore implements the same masked twisted frame without
        rebuilding the underlying Wannier transform.  Calls are absolute rather than
        cumulative, so a model can be reused across a complete twist circle.
        """
        phi = float(phi)
        if not np.isfinite(phi):
            raise ValueError("twist phi must be finite.")
        gauge = str(gauge).strip().lower()
        seam_y_eff = (self.Ny - 1) if seam_y is None else int(seam_y) % self.Ny
        new_transport = self._controller_twist_transport(
            gauge=gauge, seam_y=seam_y_eff
        )
        old_phi = float(getattr(self, "controller_twist_phi", 0.0))
        old_gauge = str(getattr(self, "controller_twist_gauge", "seam"))
        old_seam = int(
            getattr(self, "controller_twist_seam_y", self.Ny - 1)
        )
        old_transport = self._controller_twist_transport(
            gauge=old_gauge, seam_y=old_seam
        )
        delta_phase = torch.exp(
            1j * (phi * new_transport - old_phi * old_transport)
        ).to(self.dtype)
        for name in ("WF_Ap_sites", "WF_Bp_sites", "WF_Am_sites", "WF_Bm_sites"):
            setattr(self, name, (getattr(self, name) * delta_phase).contiguous())
        for short_name in ("Ap", "Bp", "Am", "Bm"):
            sites = getattr(self, f"WF_{short_name}_sites")
            frame = (
                sites.reshape(self.Ny, self.Nx, self.Nlayer)
                .permute(2, 1, 0)
                .contiguous()
            )
            setattr(self, f"WF_{short_name}", frame)
        self.controller_twist_phi = phi
        self.controller_twist_gauge = gauge
        self.controller_twist_seam_y = seam_y_eff
        if self.backend == "local":
            self._build_local_site_cache()
        return {
            "phi": phi,
            "gauge": gauge,
            "seam_y": seam_y_eff,
            "holonomy_real": float(np.cos(phi)),
            "holonomy_imag": float(np.sin(phi)),
        }

    def _sequence_helper(self, sequence, meas_slab_only=False):
        if sequence is None:
            sequence = "raster_y"
        if not isinstance(sequence, str):
            raise ValueError("sequence must be a string label.")
        mode = sequence.strip().lower()
        aliases = {"snake_y": "raster_y", "snake_x": "raster_x"}
        mode = aliases.get(mode, mode)
        allowed = ("raster_y", "raster_x", "random", "dw_symmetric_random")
        if mode not in allowed:
            raise ValueError(f"sequence must be one of: {', '.join(allowed)}.")

        coords = []

        if mode == "raster_y":
            coords = [(Rx, Ry) for Rx in range(self.Nx) for Ry in range(self.Ny)]
        elif mode == "raster_x":
            coords = [(Rx, Ry) for Ry in range(self.Ny) for Rx in range(self.Nx)]
        elif mode == "random":
            coords = [(Rx, Ry) for Rx in range(self.Nx) for Ry in range(self.Ny)]
        elif mode == "dw_symmetric_random":
            if not (hasattr(self, "DW_loc") and len(self.DW_loc) >= 2):
                raise ValueError("dw_symmetric_random requires DW_loc to define the slab.")
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_start = int(dw_sorted[-1])
            x_order = list(range(x_start, -1, -1)) + list(range(self.Nx - 1, x_start, -1))
            for Rx in x_order:
                for Ry in range(self.Ny):
                    coords.append((Rx, Ry))

        if self._meas_slab_only_effective(meas_slab_only):
            topo = self._topological_region_mask()
            coords = [(Rx, Ry) for Rx, Ry in coords if bool(topo[Rx, Ry])]

        return {"mode": mode, "coords_for_len": coords}

    def _random_complex_fermion_covariance_batch(self, batch_size, generator=None):
        N = self.Nlayer
        nfill = int(round(self.filling_frac * N))
        nfill = max(0, min(N, nfill))

        real = torch.randn((batch_size, N, N), dtype=self.real_dtype, device=self.device, generator=generator)
        imag = torch.randn((batch_size, N, N), dtype=self.real_dtype, device=self.device, generator=generator)
        mat = torch.complex(real, imag)
        q, r = torch.linalg.qr(mat)
        phases = torch.diagonal(r, dim1=-2, dim2=-1)
        phases = phases / phases.abs().clamp_min(1e-12)
        q = q * phases.conj().unsqueeze(-2)

        occ = torch.full((batch_size, N), -1.0, dtype=self.real_dtype, device=self.device)
        if nfill > 0:
            occ[:, :nfill] = 1.0
            perm_keys = torch.rand((batch_size, N), dtype=self.real_dtype, device=self.device, generator=generator)
            perm = torch.argsort(perm_keys, dim=-1)
            occ = torch.gather(occ, 1, perm)

        D = torch.diag_embed(occ.to(self.dtype))
        return q.conj().transpose(-2, -1) @ D @ q

    def _prepare_initial_batch(self, batch_size, init_mode, G_init=None, sample_offset=0):
        if G_init is not None:
            arr = torch.as_tensor(G_init, dtype=self.dtype)
            if arr.ndim == 2:
                if tuple(arr.shape) == (self.Ntot, self.Ntot):
                    arr = arr[:self.Nlayer, :self.Nlayer]
                elif tuple(arr.shape) != (self.Nlayer, self.Nlayer):
                    raise ValueError(f"G_init shape error: {tuple(arr.shape)}")
                arr = arr.unsqueeze(0).expand(batch_size, -1, -1)
            elif arr.ndim == 3:
                if tuple(arr.shape[1:]) == (self.Ntot, self.Ntot):
                    arr = arr[:, :self.Nlayer, :self.Nlayer]
                elif tuple(arr.shape[1:]) != (self.Nlayer, self.Nlayer):
                    raise ValueError(f"G_init shape error: {tuple(arr.shape)}")
                arr = arr[sample_offset:sample_offset + batch_size]
                if arr.shape[0] != batch_size:
                    raise ValueError("Batched G_init does not contain enough samples.")
            else:
                raise ValueError("G_init must have shape (Nlayer, Nlayer) or (samples, Nlayer, Nlayer).")
            arr = arr.to(device=self.device, dtype=self.dtype)
            return 0.5 * (arr + arr.conj().transpose(-2, -1))

        if init_mode == "default":
            return self._random_complex_fermion_covariance_batch(batch_size=batch_size)
        if init_mode == "maxmix":
            return torch.zeros((batch_size, self.Nlayer, self.Nlayer), dtype=self.dtype, device=self.device)
        raise ValueError("init_mode must be 'default' or 'maxmix'.")

    def _apply_onsite_phase_noise(
        self,
        G,
        *,
        sigma,
        lyapunov_state=None,
    ):
        """Apply the BPJ charge-preserving onsite phase field after one cycle.

        The many-body unitary is exp[4*pi*i*theta*(n-1/2)] with fresh
        theta ~ Uniform[0,sigma) for every trajectory and one-particle mode.  Its
        irrelevant many-body scalar phase drops out of the covariance action.
        """
        sigma = float(sigma)
        frame_native = isinstance(G, BatchedOccupiedFrameState)
        batch_count = G.batch_size if frame_native else int(G.shape[0])
        if sigma == 0.0:
            theta = torch.zeros(
                (batch_count, self.Nlayer),
                dtype=self.real_dtype,
                device=self.device,
            )
            phase = torch.ones_like(theta, dtype=self.dtype)
            return G, theta, phase
        theta = sigma * torch.rand(
            (batch_count, self.Nlayer),
            dtype=self.real_dtype,
            device=self.device,
        )
        phase = torch.exp((4j * np.pi) * theta).to(self.dtype)
        if frame_native:
            G.apply_row_phases(phase)
        else:
            G = phase.unsqueeze(-1) * G * phase.conj().unsqueeze(-2)
            G = 0.5 * (G + G.conj().transpose(-2, -1))
        if lyapunov_state is not None:
            lyapunov_state["frame"] = (
                phase.unsqueeze(-1) * lyapunov_state["frame"]
            )
        return G, theta, phase

    def _solve_regularized_batched(self, A, B, eps=1e-9):
        eye = self._eye_of_size(A.shape[-1])
        if hasattr(torch.linalg, "solve_ex"):
            X, info = torch.linalg.solve_ex(A, B, check_errors=False)
            bad = info != 0
            if not torch.any(bad):
                return X

            bad_idx = torch.nonzero(bad, as_tuple=False).flatten()
            A_bad = A.index_select(0, bad_idx)
            B_bad = B.index_select(0, bad_idx)
            eps_scale = A_bad.abs().sum(dim=-1).amax(dim=-1)
            eps_scale = torch.clamp(eps_scale, min=1.0)
            A_reg = A_bad + (eps * eps_scale)[:, None, None] * eye.unsqueeze(0)
            X_bad, info_bad = torch.linalg.solve_ex(A_reg, B_bad, check_errors=False)

            still_bad = info_bad != 0
            if torch.any(still_bad):
                bad_bad_idx = torch.nonzero(still_bad, as_tuple=False).flatten()
                X_bad[bad_bad_idx] = torch.linalg.pinv(A_reg.index_select(0, bad_bad_idx)) @ B_bad.index_select(0, bad_bad_idx)

            X = X.clone()
            X[bad] = X_bad
            return X

        try:
            return torch.linalg.solve(A, B)
        except RuntimeError:
            eps_scale = A.abs().sum(dim=-1).amax(dim=-1)
            eps_scale = torch.clamp(eps_scale, min=1.0)
            A_reg = A + (eps * eps_scale)[:, None, None] * eye.unsqueeze(0)
            try:
                return torch.linalg.solve(A_reg, B)
            except RuntimeError:
                return torch.linalg.pinv(A_reg) @ B

    def _rank_one_vector_from_projector_batched(self, P):
        P_herm = 0.5 * (P + P.conj().transpose(-2, -1))
        evals, evecs = torch.linalg.eigh(P_herm)
        pos = torch.argmax(evals, dim=-1)
        vals = torch.clamp(torch.gather(evals, -1, pos.unsqueeze(-1)).squeeze(-1), min=0.0)
        if P.ndim == 2:
            return torch.sqrt(vals).to(self.dtype) * evecs[:, int(pos.item())]
        gather_idx = pos[:, None, None].expand(-1, evecs.shape[-2], 1)
        vec = torch.gather(evecs, -1, gather_idx).squeeze(-1)
        return torch.sqrt(vals).to(self.dtype)[:, None] * vec

    def _physical_rank1_resolvent_action_batched(self, G, chi, rhs, sign, eps=1e-9):
        if G.shape[0] == 0:
            return rhs
        batch = int(G.shape[0])
        chi = chi.to(dtype=self.dtype, device=self.device)
        if chi.ndim == 1:
            chi = chi.unsqueeze(0).expand(batch, -1)
        sign = torch.as_tensor(sign, dtype=self.real_dtype, device=self.device).reshape(-1)
        if sign.numel() == 1:
            sign = sign.expand(batch)
        sign_complex = sign.to(self.dtype)
        v = torch.matmul(G, chi.unsqueeze(-1)).squeeze(-1)
        chi_rhs = torch.matmul(chi.conj().unsqueeze(1), rhs).squeeze(1)
        rank_one_scalar = torch.sum(chi.conj() * v, dim=-1)
        denominator = 1.0 + sign_complex * rank_one_scalar
        bad = (~torch.isfinite(denominator)) | (
            torch.abs(denominator).to(self.real_dtype) < 1e-14
        )

        # A host-side ``if torch.any(bad)`` here synchronized CUDA after every
        # measurement channel.  The regularized Sherman--Morrison expression below
        # handles good and exceptional rows together and is exactly the unregularized
        # rank-one inverse when ``bad`` is false.
        abs_chi = torch.abs(chi).to(self.real_dtype)
        abs_v = torch.abs(v).to(self.real_dtype)
        diagonal_abs = torch.abs(
            1.0 + sign_complex[:, None] * v * chi.conj()
        ).to(self.real_dtype)
        row_abs_sum = diagonal_abs + abs_v * (
            torch.sum(abs_chi, dim=-1)[:, None] - abs_chi
        )
        eps_scale = torch.clamp(torch.amax(row_abs_sum, dim=-1), min=1.0)
        alpha_real = torch.where(
            bad,
            1.0 + float(eps) * eps_scale,
            torch.ones_like(eps_scale),
        )
        alpha = alpha_real.to(self.dtype)
        regularized_denominator = alpha + sign_complex * rank_one_scalar
        coefficient = sign_complex / (alpha * regularized_denominator)
        return (
            rhs / alpha[:, None, None]
            - coefficient[:, None, None]
            * v.unsqueeze(-1)
            * chi_rhs.unsqueeze(-2)
        )

    def _physical_rank1_measure_blocks_batched(self, G_ss, G_sr, chi_local, particle):
        p = self._projector_from_vector(chi_local)
        q = self._eye_of_size(G_ss.shape[-1]) - p
        sign_value = 1.0 if bool(particle) else -1.0
        sign = torch.full((G_ss.shape[0],), sign_value, dtype=self.real_dtype, device=self.device)
        if G_sr.shape[-1] == 0:
            Y = G_sr
            pY = Y
            G_sr_new = Y
        else:
            Y = self._physical_rank1_resolvent_action_batched(G_ss, chi_local, G_sr, sign)
            pY = torch.matmul(p.unsqueeze(0), Y)
            G_sr_new = torch.matmul(q.unsqueeze(0), Y)
        G_ss_q = torch.matmul(G_ss, q.unsqueeze(0))
        Z = self._physical_rank1_resolvent_action_batched(G_ss, chi_local, G_ss_q, sign)
        G_ss_new = sign_value * p.unsqueeze(0) + torch.matmul(q.unsqueeze(0), Z)
        G_ss_new = 0.5 * (G_ss_new + G_ss_new.conj().transpose(-2, -1))
        if G_sr.shape[-1] == 0:
            delta_rr = G_sr.new_empty((G_ss.shape[0], 0, 0))
        else:
            delta_rr = -sign_value * torch.matmul(G_sr.conj().transpose(-2, -1), pY)
            delta_rr = 0.5 * (delta_rr + delta_rr.conj().transpose(-2, -1))
        return G_ss_new, G_sr_new, delta_rr

    def _init_lyapunov_state(
        self,
        batch_count,
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
            self._all_top_indices
            if basis_idx is None
            else torch.as_tensor(basis_idx, dtype=torch.long, device=self.device)
        )
        basis_dim = int(basis_idx.numel())
        batch_count = int(batch_count)
        sample_start = int(sample_start)

        if initial_frame is None:
            n_vec = basis_dim if n_vec is None else int(n_vec)
            frame = torch.zeros(
                (batch_count, self.Nlayer, n_vec),
                dtype=self.dtype,
                device=self.device,
            )
            frame[
                :,
                basis_idx[:n_vec],
                torch.arange(n_vec, dtype=torch.long, device=self.device),
            ] = 1.0
        else:
            raw = torch.as_tensor(
                initial_frame, dtype=self.dtype, device=self.device
            )
            if raw.ndim == 2:
                raw = raw.unsqueeze(0).expand(batch_count, -1, -1).clone()
            elif raw.ndim == 3:
                if int(raw.shape[0]) == batch_count:
                    raw = raw.clone()
                elif int(raw.shape[0]) >= sample_start + batch_count:
                    raw = raw[sample_start : sample_start + batch_count].clone()
                else:
                    raise ValueError(
                        "A batched lyapunov_initial_frame must have either "
                        "batch_count rows or enough rows for the sample slice."
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
                    f"lyapunov_nvec={int(n_vec)} disagrees with the initial "
                    f"frame's {inferred_nvec} columns."
                )
            n_vec = inferred_nvec
            if int(raw.shape[1]) == self.Nlayer:
                frame = raw
            elif int(raw.shape[1]) == basis_dim:
                frame = torch.zeros(
                    (batch_count, self.Nlayer, n_vec),
                    dtype=self.dtype,
                    device=self.device,
                )
                frame[:, basis_idx, :] = raw
            else:
                raise ValueError(
                    "lyapunov_initial_frame row dimension must equal the full "
                    "top-layer dimension or the active-basis dimension."
                )
            if not bool(torch.all(torch.isfinite(frame)).item()):
                raise ValueError("lyapunov_initial_frame contains non-finite entries.")
            gram = frame.conj().transpose(-2, -1) @ frame
            target = torch.eye(
                n_vec, dtype=self.dtype, device=self.device
            ).unsqueeze(0).expand(batch_count, -1, -1)
            gram_tol = 1e-6 if self.dtype == torch.complex64 else 1e-9
            if not torch.allclose(gram, target, atol=gram_tol, rtol=gram_tol):
                error = float(torch.amax(torch.abs(gram - target)).item())
                raise ValueError(
                    "lyapunov_initial_frame columns must be orthonormal; "
                    f"maximum Gram error is {error:.3e}."
                )

        if n_vec <= 0 or n_vec > basis_dim:
            raise ValueError(
                f"lyapunov_nvec must satisfy 1 <= n_vec <= {basis_dim}; got {n_vec}"
            )
        failure_mode = str(failure_mode).strip().lower()
        if failure_mode not in ("raise", "censor"):
            raise ValueError(
                "lyapunov_failure_mode must be either 'raise' or 'censor'."
            )
        singular_tol = float(singular_tol)
        if not np.isfinite(singular_tol) or singular_tol <= 0.0:
            raise ValueError(
                "lyapunov_singular_tol must be a positive finite scalar."
            )
        track_restricted_core = bool(
            track_restricted_core or track_record_fisher
        )
        track_record_fisher = bool(track_record_fisher)
        if track_record_fisher and n_vec != 2:
            raise ValueError(
                "lyapunov_track_record_fisher requires exactly two tangent columns."
            )

        state = {
            "frame": frame,
            "log_diag": torch.zeros(
                (batch_count, n_vec), dtype=torch.float64, device=self.device
            ),
            "n_vec": n_vec,
            "basis_idx": basis_idx,
            "null_counts": torch.zeros(
                (batch_count,), dtype=torch.int64, device=self.device
            ),
            "active": torch.ones(
                (batch_count,), dtype=torch.bool, device=self.device
            ),
            "singular_tol": singular_tol,
            "failure_mode": failure_mode,
            "failure_records": [],
            "min_branch_probability": torch.full(
                (batch_count,), torch.inf, dtype=torch.float64, device=self.device
            ),
            "min_abs_born_denominator": torch.full(
                (batch_count,), torch.inf, dtype=torch.float64, device=self.device
            ),
            "invalid_branch_count": torch.zeros(
                (batch_count,), dtype=torch.int64, device=self.device
            ),
            "track_restricted_core": track_restricted_core,
            "track_record_fisher": track_record_fisher,
            "last_r": None,
            "last_cycle_null_mask": torch.zeros(
                (batch_count, n_vec), dtype=torch.bool, device=self.device
            ),
        }
        if track_restricted_core:
            state["core_hat"] = torch.eye(
                n_vec, dtype=self.dtype, device=self.device
            ).unsqueeze(0).expand(batch_count, -1, -1).clone()
            state["core_log_scale"] = torch.zeros(
                (batch_count,), dtype=torch.float64, device=self.device
            )
            state["core_null_count"] = torch.zeros(
                (batch_count,), dtype=torch.int64, device=self.device
            )
        if track_record_fisher:
            state["record_fisher_hat"] = torch.zeros(
                (batch_count, 3, 3), dtype=torch.float64, device=self.device
            )
            state["record_fisher_log_scale"] = torch.full(
                (batch_count,), -torch.inf, dtype=torch.float64, device=self.device
            )
            state["record_fisher_endpoint_count"] = torch.zeros(
                (batch_count,), dtype=torch.int64, device=self.device
            )
            state["record_fisher_infinite"] = torch.zeros(
                (batch_count,), dtype=torch.bool, device=self.device
            )
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
        """Initialize the exact occupied/empty ambient half-Jacobian blocks.

        The physical pure state supplies a complete orthonormal decomposition of
        the cycle-zero active one-particle space.  Subsequent rank-one tangent
        updates act on the concatenated frame, while QR and cumulative cores are
        maintained independently for the occupied and empty column blocks.
        """
        if not isinstance(physical_state, BatchedOccupiedFrameState):
            raise TypeError(
                "lyapunov_basis_mode='pure_occupied_empty' requires "
                "state_representation='physical_frame'."
            )
        if physical_state.batch_size != 1:
            raise ValueError(
                "lyapunov_basis_mode='pure_occupied_empty' requires batch_size=1 "
                "so trajectory-dependent occupied ranks remain exact."
            )
        basis_idx = (
            self._all_top_indices
            if basis_idx is None
            else torch.as_tensor(basis_idx, dtype=torch.long, device=self.device)
        )
        active_dim = int(basis_idx.numel())
        rank = int(physical_state.ranks[0].item())
        occupied_full = physical_state.frame[0, :, :rank]
        occupied_restricted = occupied_full.index_select(0, basis_idx)
        correlation = occupied_restricted @ occupied_restricted.mH
        correlation = 0.5 * (correlation + correlation.mH)
        occupations, eigenvectors = torch.linalg.eigh(correlation)
        purity_defect = torch.minimum(
            occupations.abs(), (1.0 - occupations).abs()
        ).amax()
        if not bool(torch.isfinite(purity_defect).item()) or float(
            purity_defect.item()
        ) > float(purity_tolerance):
            raise ValueError(
                "The active cycle-zero state is not pure enough to define exact "
                "occupied/empty tangent blocks; maximum occupation defect is "
                f"{float(purity_defect.item()):.3e}."
            )
        occupied_mask = occupations > 0.5
        occupied_basis = eigenvectors[:, occupied_mask]
        empty_basis = eigenvectors[:, ~occupied_mask]
        occupied_dim = int(occupied_basis.shape[1])
        empty_dim = int(empty_basis.shape[1])
        if occupied_dim + empty_dim != active_dim:
            raise RuntimeError("Occupied/empty basis did not span the active space.")
        initial_frame = torch.cat((occupied_basis, empty_basis), dim=1)
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
                    torch.eye(
                        occupied_dim, dtype=self.dtype, device=self.device
                    ).unsqueeze(0),
                    torch.eye(
                        empty_dim, dtype=self.dtype, device=self.device
                    ).unsqueeze(0),
                ),
                "block_core_log_scale": (
                    torch.zeros((1,), dtype=torch.float64, device=self.device),
                    torch.zeros((1,), dtype=torch.float64, device=self.device),
                ),
                "block_core_null_count": (
                    torch.zeros((1,), dtype=torch.int64, device=self.device),
                    torch.zeros((1,), dtype=torch.int64, device=self.device),
                ),
                "last_r_blocks": (None, None),
                "initial_active_occupations": occupations,
                "initial_active_purity_defect": purity_defect.to(torch.float64),
                "track_restricted_core": True,
            }
        )
        return state

    def _init_choi_state(self, batch_count, singular_tol, basis_idx=None, failure_mode="raise"):
        batch_count = int(batch_count)
        basis_idx = self._all_top_indices if basis_idx is None else basis_idx.to(dtype=torch.long, device=self.device)
        dim = int(basis_idx.numel())
        zero = torch.zeros((batch_count, dim, dim), dtype=self.dtype, device=self.device)
        identity = self._eye_of_size(dim).unsqueeze(0).expand(batch_count, -1, -1).clone()
        full_to_basis = torch.full((self.Nlayer,), -1, dtype=torch.long, device=self.device)
        full_to_basis[basis_idx] = torch.arange(dim, dtype=torch.long, device=self.device)
        return {
            "LL": zero.clone(),
            "LR": identity,
            "RR": zero,
            "basis_idx": basis_idx,
            "full_to_basis": full_to_basis,
            "singular_tol": float(singular_tol),
            "failure_mode": str(failure_mode),
            "active": torch.ones((batch_count,), dtype=torch.bool, device=self.device),
            "failure_records": [],
            "min_abs_d": float("inf"),
            "min_abs_d_context": None,
        }

    @staticmethod
    def _choi_outer(left, right):
        return left.unsqueeze(-1) * right.conj().unsqueeze(-2)

    def _choi_resolvent_action(self, rhs, v, chi, d, eta1, regularized, eps=1e-9):
        chi_rhs = torch.matmul(chi.conj().unsqueeze(1), rhs).squeeze(1)
        alpha_real = torch.ones((rhs.shape[0],), dtype=self.real_dtype, device=self.device)
        if torch.any(regularized):
            abs_chi = torch.abs(chi).to(self.real_dtype)
            abs_v = torch.abs(v).to(self.real_dtype)
            eta1_complex = eta1.to(self.dtype)
            diag_abs = torch.abs(1.0 - eta1_complex[:, None] * v * chi.conj()).to(self.real_dtype)
            row_abs_sum = diag_abs + abs_v * (torch.sum(abs_chi, dim=-1)[:, None] - abs_chi)
            eps_scale = torch.clamp(torch.amax(row_abs_sum, dim=-1), min=1.0)
            alpha_real[regularized] = 1.0 + float(eps) * eps_scale[regularized]

        alpha = alpha_real.to(self.dtype)
        eta1_complex = eta1.to(self.dtype)
        r = d + eta1_complex
        denom = alpha - eta1_complex * r
        coeff = eta1_complex / (alpha * denom)
        return rhs / alpha[:, None, None] + coeff[:, None, None] * v.unsqueeze(-1) * chi_rhs.unsqueeze(-2)

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
        if choi_state is None or int(sample_offsets.numel()) == 0:
            return
        sample_offsets = sample_offsets.to(dtype=torch.long, device=self.device)
        active = choi_state["active"].index_select(0, sample_offsets)
        if not torch.any(active):
            return
        sample_offsets = sample_offsets[active]
        ll = choi_state["LL"].index_select(0, sample_offsets)
        lr = choi_state["LR"].index_select(0, sample_offsets)
        rr = choi_state["RR"].index_select(0, sample_offsets)
        count = int(sample_offsets.numel())

        chi = chi.to(dtype=self.dtype, device=self.device)
        basis_idx = choi_state.get("basis_idx")
        if basis_idx is not None and int(basis_idx.numel()) != self.Nlayer:
            if support_idx is None:
                chi = chi.index_select(-1, basis_idx)
            else:
                mapped = choi_state["full_to_basis"].index_select(0, support_idx.to(dtype=torch.long, device=self.device))
                if torch.any(mapped < 0):
                    raise ValueError("Choi update received mode support outside the active slab basis.")
                support_idx = mapped
        if chi.ndim == 1:
            chi = chi.unsqueeze(0).expand(count, -1)
        if support_idx is None:
            chi_basis = chi
        else:
            support_idx = support_idx.to(dtype=torch.long, device=self.device)
            chi_basis = torch.zeros((count, rr.shape[-1]), dtype=self.dtype, device=self.device)
            chi_basis[:, support_idx] = chi
        u = torch.matmul(lr, chi_basis.unsqueeze(-1)).squeeze(-1)
        v = torch.matmul(rr, chi_basis.unsqueeze(-1)).squeeze(-1)
        r = torch.sum(chi_basis.conj() * v, dim=-1)

        eta1 = torch.as_tensor(eta1, dtype=self.real_dtype, device=self.device).reshape(-1)
        eta2 = torch.as_tensor(eta2, dtype=self.real_dtype, device=self.device).reshape(-1)
        if eta1.numel() == 1:
            eta1 = eta1.expand(count)
        if eta2.numel() == 1:
            eta2 = eta2.expand(count)
        d = r - eta1.to(self.dtype)
        abs_d = torch.abs(d).to(torch.float64)
        local_min_value, local_min_pos = torch.min(abs_d, dim=0)
        local_min = float(local_min_value.item())
        if local_min < choi_state["min_abs_d"]:
            local_pos = int(local_min_pos.item())
            diagnostic = {} if context is None else dict(context)
            diagnostic.update(
                {
                    "sample_offset": int(sample_offsets[local_pos].item()),
                    "eta1": float(eta1[local_pos].item()),
                    "eta2": float(eta2[local_pos].item()),
                    "d_real": float(d[local_pos].real.item()),
                    "d_imag": float(d[local_pos].imag.item()),
                }
            )
            if diagnostic.get("batch_start") is not None:
                diagnostic["sample_index"] = int(diagnostic["batch_start"]) + diagnostic["sample_offset"]
            choi_state["min_abs_d"] = local_min
            choi_state["min_abs_d_context"] = diagnostic
        invalid = abs_d < choi_state["singular_tol"]
        if torch.any(invalid):
            if choi_state["failure_mode"] == "raise":
                raise FloatingPointError(
                    "Cumulative Choi contraction encountered abs(d) below choi_singular_tol: "
                    f"abs(d)={local_min:.6e}, tol={choi_state['singular_tol']:.6e}, "
                    f"context={choi_state['min_abs_d_context']}"
                )
            failed_pos = torch.nonzero(invalid, as_tuple=False).flatten()
            failed_offsets = sample_offsets.index_select(0, failed_pos)
            for pos, offset in zip(failed_pos.tolist(), failed_offsets.tolist()):
                record = {} if context is None else dict(context)
                record.update(
                    {
                        "stage": "denominator_regularized",
                        "sample_offset": int(offset),
                        "eta1": float(eta1[pos].item()),
                        "eta2": float(eta2[pos].item()),
                        "abs_d": float(abs_d[pos].item()),
                        "d_real": float(d[pos].real.item()),
                        "d_imag": float(d[pos].imag.item()),
                    }
                )
                if record.get("batch_start") is not None:
                    record["sample_index"] = int(record["batch_start"]) + int(offset)
                choi_state["failure_records"].append(record)

        sigma_rl = lr.conj().transpose(-2, -1)
        rq = rr - v.unsqueeze(-1) * chi_basis.conj().unsqueeze(-2)
        x = self._choi_resolvent_action(sigma_rl, v, chi_basis, d, eta1, invalid, eps=1e-9)
        y = self._choi_resolvent_action(rq, v, chi_basis, d, eta1, invalid, eps=1e-9)
        chi_x = torch.matmul(chi_basis.conj().unsqueeze(1), x).squeeze(1)
        chi_y = torch.matmul(chi_basis.conj().unsqueeze(1), y).squeeze(1)
        eta1_complex = eta1.to(self.dtype)
        eta2_complex = eta2.to(self.dtype)
        lrq = lr - u.unsqueeze(-1) * chi_basis.conj().unsqueeze(-2)
        ll_new = ll + eta1_complex[:, None, None] * u.unsqueeze(-1) * chi_x.unsqueeze(-2)
        lr_new = lrq + eta1_complex[:, None, None] * u.unsqueeze(-1) * chi_y.unsqueeze(-2)
        rr_new = (
            eta2_complex[:, None, None] * self._choi_outer(chi_basis, chi_basis)
            + y
            - chi_basis.unsqueeze(-1) * chi_y.unsqueeze(-2)
        )

        choi_state["LL"][sample_offsets] = ll_new
        choi_state["LR"][sample_offsets] = lr_new
        choi_state["RR"][sample_offsets] = rr_new

    def _lyapunov_register_branch(
        self, lyapunov_state, G_sel, sample_offsets, P, particle
    ):
        """Record fixed-branch Born denominators and return valid selected rows."""
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        if P.ndim == 2:
            P_batch = P.unsqueeze(0).expand(G_sel.shape[0], -1, -1)
        else:
            P_batch = P
        sign = 1.0 if bool(particle) else -1.0
        g = torch.einsum("bij,bji->b", G_sel, P_batch).real
        denominator = torch.abs(1.0 + sign * g).to(torch.float64)
        probability = 0.5 * denominator
        active = lyapunov_state["active"].index_select(0, sample_offsets)

        current_d = lyapunov_state[
            "min_abs_born_denominator"
        ].index_select(0, sample_offsets)
        current_p = lyapunov_state[
            "min_branch_probability"
        ].index_select(0, sample_offsets)
        lyapunov_state["min_abs_born_denominator"][sample_offsets] = torch.where(
            active, torch.minimum(current_d, denominator), current_d
        )
        lyapunov_state["min_branch_probability"][sample_offsets] = torch.where(
            active, torch.minimum(current_p, probability), current_p
        )

        finite = torch.isfinite(denominator)
        positive = denominator > float(lyapunov_state["singular_tol"])
        valid = active & finite & positive
        invalid = active & ~valid
        if not bool(torch.any(invalid).item()):
            return valid, g

        bad_positions = torch.nonzero(invalid, as_tuple=False).flatten()
        bad_offsets = sample_offsets.index_select(0, bad_positions)
        lyapunov_state["invalid_branch_count"][bad_offsets] += 1
        for position, offset in zip(
            bad_positions.detach().cpu().tolist(),
            bad_offsets.detach().cpu().tolist(),
        ):
            lyapunov_state["failure_records"].append(
                {
                    "sample_offset": int(offset),
                    "particle": bool(particle),
                    "born_denominator": float(denominator[position].item()),
                    "branch_probability": float(probability[position].item()),
                }
            )
        if lyapunov_state["failure_mode"] == "raise":
            raise FloatingPointError(
                "The requested fixed measurement branch has zero or non-finite "
                "Born probability; sample offsets="
                f"{bad_offsets.detach().cpu().tolist()}, denominators="
                f"{denominator[bad_positions].detach().cpu().tolist()}."
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
        """Accumulate predictable record Fisher information for a 2-mode plane."""
        if lyapunov_state is None or not lyapunov_state.get(
            "track_record_fisher", False
        ):
            return
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        if P.ndim == 2:
            P_batch = P.unsqueeze(0).expand(G_sel.shape[0], -1, -1)
        else:
            P_batch = P
        if valid_mask is None:
            valid_mask = torch.ones(
                (sample_offsets.numel(),), dtype=torch.bool, device=self.device
            )
        valid_mask = valid_mask & lyapunov_state["active"].index_select(
            0, sample_offsets
        )

        frame = lyapunov_state["frame"].index_select(0, sample_offsets)
        if support_idx is not None:
            frame = frame.index_select(1, support_idx)
        core_hat = lyapunov_state["core_hat"].index_select(0, sample_offsets)
        image_hat = torch.matmul(frame, core_hat)
        overlap = torch.matmul(
            image_hat.conj().transpose(-2, -1),
            torch.matmul(P_batch, image_hat),
        )
        d = torch.stack(
            (
                2.0 * overlap[:, 0, 1].real,
                -2.0 * overlap[:, 0, 1].imag,
                (overlap[:, 0, 0] - overlap[:, 1, 1]).real,
            ),
            dim=-1,
        ).to(torch.float64)
        g = torch.einsum("bij,bji->b", G_sel, P_batch).real.to(torch.float64)
        variance = torch.clamp(1.0 - g * g, min=0.0)
        d_norm = torch.linalg.vector_norm(d, dim=-1)
        tol = float(lyapunov_state["singular_tol"])

        endpoint = valid_mask & (variance <= tol)
        if bool(torch.any(endpoint).item()):
            endpoint_positions = torch.nonzero(
                endpoint, as_tuple=False
            ).flatten()
            endpoint_offsets = sample_offsets.index_select(
                0, endpoint_positions
            )
            lyapunov_state["record_fisher_endpoint_count"][endpoint_offsets] += 1
            divergent = d_norm.index_select(0, endpoint_positions) > np.sqrt(tol)
            if bool(torch.any(divergent).item()):
                lyapunov_state["record_fisher_infinite"][
                    endpoint_offsets[divergent]
                ] = True

        gamma = lyapunov_state["core_log_scale"].index_select(
            0, sample_offsets
        )
        accumulate = (
            valid_mask
            & (variance > tol)
            & torch.isfinite(gamma)
            & (d_norm > 0.0)
        )
        if not bool(torch.any(accumulate).item()):
            return
        positions = torch.nonzero(accumulate, as_tuple=False).flatten()
        offsets = sample_offsets.index_select(0, positions)
        d_selected = d.index_select(0, positions)
        increment_hat = d_selected.unsqueeze(-1) * d_selected.unsqueeze(-2)
        increment_log_scale = (
            4.0 * gamma.index_select(0, positions)
            - torch.log(variance.index_select(0, positions))
        )
        old_log_scale = lyapunov_state[
            "record_fisher_log_scale"
        ].index_select(0, offsets)
        new_log_scale = torch.maximum(old_log_scale, increment_log_scale)
        old_weight = torch.where(
            torch.isfinite(old_log_scale),
            torch.exp(old_log_scale - new_log_scale),
            torch.zeros_like(old_log_scale),
        )
        new_weight = torch.exp(increment_log_scale - new_log_scale)
        old_hat = lyapunov_state["record_fisher_hat"].index_select(0, offsets)
        lyapunov_state["record_fisher_hat"][offsets] = (
            old_weight[:, None, None] * old_hat
            + new_weight[:, None, None] * increment_hat
        )
        lyapunov_state["record_fisher_log_scale"][offsets] = new_log_scale

    def _lyapunov_apply_local_selected(
        self,
        lyapunov_state,
        G_sel,
        sample_offsets,
        support_idx,
        comp_idx,
        chi_local,
        particle,
    ):
        if lyapunov_state is None or int(G_sel.shape[0]) == 0:
            return
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        support_idx = support_idx.to(dtype=torch.long, device=self.device)
        comp_idx = comp_idx.to(dtype=torch.long, device=self.device)
        G_ss, G_sr = self._local_support_block(
            G_sel, support_idx, comp_idx
        )
        p = self._projector_from_vector(chi_local)
        q = self._eye_of_size(support_idx.numel()) - p
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
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            return

        offsets = sample_offsets.index_select(0, positions)
        G_ss = G_ss.index_select(0, positions)
        G_sr = G_sr.index_select(0, positions)
        V = lyapunov_state["frame"].index_select(0, offsets)
        V_s = V.index_select(1, support_idx)
        V_r = (
            V.new_empty((V.shape[0], 0, V.shape[-1]))
            if int(comp_idx.numel()) == 0
            else V.index_select(1, comp_idx)
        )
        sign = 1.0 if bool(particle) else -1.0
        p_batch = p.unsqueeze(0)
        solve_mat = self._eye_of_size(
            support_idx.numel()
        ).unsqueeze(0) + sign * torch.matmul(G_ss, p_batch)
        X_s = torch.linalg.solve(solve_mat, V_s)
        pX_s = torch.matmul(p_batch, X_s)
        X_r = (
            V_r
            if int(comp_idx.numel()) == 0
            else V_r
            - sign
            * torch.matmul(G_sr.conj().transpose(-2, -1), pX_s)
        )

        V_new = V.clone()
        V_new[:, support_idx, :] = torch.matmul(q.unsqueeze(0), X_s)
        if int(comp_idx.numel()) > 0:
            V_new[:, comp_idx, :] = X_r
        lyapunov_state["frame"][offsets] = V_new

    def _lyapunov_apply_local_channel(self, lyapunov_state, G, sample_offsets, support_idx, comp_idx, chi_local, particle):
        if lyapunov_state is None:
            return
        if isinstance(particle, bool):
            self._lyapunov_apply_local_selected(
                lyapunov_state,
                G,
                sample_offsets,
                support_idx,
                comp_idx,
                chi_local,
                particle=particle,
            )
            return
        particle = particle.to(dtype=torch.bool, device=self.device)
        idx_occ = torch.nonzero(particle, as_tuple=False).flatten()
        idx_unocc = torch.nonzero(~particle, as_tuple=False).flatten()
        if idx_occ.numel() > 0:
            self._lyapunov_apply_local_selected(
                lyapunov_state,
                G.index_select(0, idx_occ),
                sample_offsets.index_select(0, idx_occ),
                support_idx,
                comp_idx,
                chi_local,
                particle=True,
            )
        if idx_unocc.numel() > 0:
            self._lyapunov_apply_local_selected(
                lyapunov_state,
                G.index_select(0, idx_unocc),
                sample_offsets.index_select(0, idx_unocc),
                support_idx,
                comp_idx,
                chi_local,
                particle=False,
            )

    def _lyapunov_apply_dense_selected(
        self, lyapunov_state, G_sel, sample_offsets, P, particle
    ):
        if lyapunov_state is None or int(G_sel.shape[0]) == 0:
            return
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        P_batch = (
            P.unsqueeze(0).expand(G_sel.shape[0], -1, -1)
            if P.ndim == 2
            else P
        )
        valid, _ = self._lyapunov_register_branch(
            lyapunov_state, G_sel, sample_offsets, P_batch, particle
        )
        self._lyapunov_accumulate_record_fisher(
            lyapunov_state,
            G_sel,
            sample_offsets,
            P_batch,
            valid_mask=valid,
        )
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            return

        offsets = sample_offsets.index_select(0, positions)
        G_valid = G_sel.index_select(0, positions)
        P_valid = P_batch.index_select(0, positions)
        V = lyapunov_state["frame"].index_select(0, offsets)
        eye = self._eye_top.unsqueeze(0)
        sign = 1.0 if bool(particle) else -1.0
        solve_mat = eye + sign * torch.matmul(G_valid, P_valid)
        X = torch.linalg.solve(solve_mat, V)
        V_new = torch.matmul(eye - P_valid, X)
        lyapunov_state["frame"][offsets] = V_new

    def _lyapunov_apply_dense_channel(self, lyapunov_state, G, sample_offsets, P, particle):
        if lyapunov_state is None:
            return
        if isinstance(particle, bool):
            self._lyapunov_apply_dense_selected(lyapunov_state, G, sample_offsets, P, particle=particle)
            return
        particle = particle.to(dtype=torch.bool, device=self.device)
        idx_occ = torch.nonzero(particle, as_tuple=False).flatten()
        idx_unocc = torch.nonzero(~particle, as_tuple=False).flatten()
        if idx_occ.numel() > 0:
            self._lyapunov_apply_dense_selected(
                lyapunov_state,
                G.index_select(0, idx_occ),
                sample_offsets.index_select(0, idx_occ),
                P,
                particle=True,
            )
        if idx_unocc.numel() > 0:
            self._lyapunov_apply_dense_selected(
                lyapunov_state,
                G.index_select(0, idx_unocc),
                sample_offsets.index_select(0, idx_unocc),
                P,
                particle=False,
            )

    def _lyapunov_apply_local_reset(
        self, lyapunov_state, sample_offsets, support_idx, chi_local
    ):
        """Apply the reset differential Q explicitly on selected trajectories."""
        if lyapunov_state is None or int(sample_offsets.numel()) == 0:
            return
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        support_idx = support_idx.to(dtype=torch.long, device=self.device)
        V = lyapunov_state["frame"].index_select(0, sample_offsets).clone()
        q = self._eye_of_size(support_idx.numel()) - self._projector_from_vector(
            chi_local
        )
        V[:, support_idx, :] = torch.matmul(
            q.unsqueeze(0), V.index_select(1, support_idx)
        )
        lyapunov_state["frame"][sample_offsets] = V

    def _lyapunov_apply_dense_reset(
        self, lyapunov_state, sample_offsets, P
    ):
        """Apply the reset differential Q explicitly on selected trajectories."""
        if lyapunov_state is None or int(sample_offsets.numel()) == 0:
            return
        sample_offsets = sample_offsets.to(
            dtype=torch.long, device=self.device
        ).reshape(-1)
        V = lyapunov_state["frame"].index_select(0, sample_offsets)
        P_batch = (
            P.unsqueeze(0).expand(V.shape[0], -1, -1)
            if P.ndim == 2
            else P
        )
        lyapunov_state["frame"][sample_offsets] = torch.matmul(
            self._eye_top.unsqueeze(0) - P_batch, V
        )

    def _lyapunov_end_cycle(self, lyapunov_state, cycle):
        if lyapunov_state is None:
            return None
        if lyapunov_state["failure_mode"] == "raise":
            invalid = lyapunov_state["invalid_branch_count"] > 0
            if bool(torch.any(invalid).item()):
                offsets = torch.nonzero(
                    invalid, as_tuple=False
                ).flatten().detach().cpu().tolist()
                raise FloatingPointError(
                    "The requested fixed measurement branch had zero or non-finite "
                    "Born probability on the sitewise batched path; sample offsets="
                    f"{offsets}."
                )
        if lyapunov_state.get("basis_mode") == "pure_occupied_empty":
            return self._lyapunov_end_cycle_pure_blocks(lyapunov_state, cycle)
        frame = lyapunov_state["frame"]
        q, r = torch.linalg.qr(frame, mode="reduced")
        diag = torch.diagonal(r, dim1=-2, dim2=-1)
        abs_diag = torch.abs(diag).to(torch.float64)
        floor = torch.finfo(abs_diag.dtype).tiny
        active = lyapunov_state["active"]
        scale = torch.clamp(torch.amax(abs_diag, dim=-1), min=1.0)
        rank_tol = (
            torch.finfo(abs_diag.dtype).eps
            * max(int(frame.shape[1]), int(frame.shape[2]))
            * scale
        )
        rank_null_mask = (abs_diag <= rank_tol[:, None]) & active[:, None]
        null_mask = rank_null_mask | ~active[:, None]
        lyapunov_state["null_counts"] += torch.count_nonzero(
            rank_null_mask, dim=-1
        ).to(torch.int64)

        updated_log = lyapunov_state["log_diag"] + torch.log(
            torch.clamp(abs_diag, min=floor)
        )
        lyapunov_state["log_diag"] = torch.where(
            active[:, None],
            updated_log,
            torch.full_like(updated_log, -torch.inf),
        )
        phase = torch.where(
            torch.abs(diag) > 1e-30,
            diag / torch.clamp(torch.abs(diag), min=1e-30),
            torch.ones_like(diag),
        )
        # Positive-diagonal convention: (Q diag phase)(diag phase* R).
        q_gauge = q * phase.unsqueeze(-2)
        r_gauge = phase.conj().unsqueeze(-1) * r
        q_gauge = torch.where(
            active[:, None, None], q_gauge, torch.zeros_like(q_gauge)
        )
        r_gauge = torch.where(
            active[:, None, None], r_gauge, torch.zeros_like(r_gauge)
        )
        lyapunov_state["frame"] = q_gauge
        lyapunov_state["last_r"] = r_gauge
        lyapunov_state["last_cycle_null_mask"] = null_mask

        if lyapunov_state.get("track_restricted_core", False):
            core_raw = torch.matmul(r_gauge, lyapunov_state["core_hat"])
            core_norm = torch.linalg.matrix_norm(
                core_raw, ord="fro", dim=(-2, -1)
            ).to(torch.float64)
            core_valid = active & torch.isfinite(core_norm) & (core_norm > floor)
            if bool(torch.any(core_valid).item()):
                positions = torch.nonzero(
                    core_valid, as_tuple=False
                ).flatten()
                lyapunov_state["core_hat"][positions] = (
                    core_raw.index_select(0, positions)
                    / core_norm.index_select(0, positions)[:, None, None].to(
                        self.dtype
                    )
                )
                lyapunov_state["core_log_scale"][positions] += torch.log(
                    core_norm.index_select(0, positions)
                )
            core_bad = active & ~core_valid
            if bool(torch.any(core_bad).item()):
                positions = torch.nonzero(core_bad, as_tuple=False).flatten()
                lyapunov_state["core_hat"][positions] = 0.0
                lyapunov_state["core_log_scale"][positions] = -torch.inf
                lyapunov_state["core_null_count"][positions] += 1

        spectrum = lyapunov_state["log_diag"] / float(cycle)
        return torch.sort(spectrum, dim=-1).values

    def _lyapunov_end_cycle_pure_blocks(self, lyapunov_state, cycle):
        """QR-stabilize occupied and empty ambient blocks independently."""
        frame = lyapunov_state["frame"]
        active = lyapunov_state["active"]
        floor = torch.finfo(torch.float64).tiny
        q_blocks = []
        r_blocks = []
        null_masks = []
        start = 0
        core_hats = list(lyapunov_state["block_core_hat"])
        core_scales = list(lyapunov_state["block_core_log_scale"])
        core_null_counts = list(lyapunov_state["block_core_null_count"])
        for block_index, block_size in enumerate(lyapunov_state["block_sizes"]):
            stop = start + int(block_size)
            raw_block = frame[:, :, start:stop]
            q, r = torch.linalg.qr(raw_block, mode="reduced")
            diag = torch.diagonal(r, dim1=-2, dim2=-1)
            abs_diag = torch.abs(diag).to(torch.float64)
            scale = torch.clamp(torch.amax(abs_diag, dim=-1), min=1.0)
            rank_tol = (
                torch.finfo(torch.float64).eps
                * max(int(raw_block.shape[1]), int(raw_block.shape[2]))
                * scale
            )
            rank_null = (abs_diag <= rank_tol[:, None]) & active[:, None]
            null_masks.append(rank_null | ~active[:, None])
            lyapunov_state["null_counts"] += torch.count_nonzero(
                rank_null, dim=-1
            ).to(torch.int64)
            updated_log = (
                lyapunov_state["log_diag"][:, start:stop]
                + torch.log(torch.clamp(abs_diag, min=floor))
            )
            lyapunov_state["log_diag"][:, start:stop] = torch.where(
                active[:, None],
                updated_log,
                torch.full_like(updated_log, -torch.inf),
            )
            phase = torch.where(
                torch.abs(diag) > 1e-30,
                diag / torch.clamp(torch.abs(diag), min=1e-30),
                torch.ones_like(diag),
            )
            q_gauge = q * phase.unsqueeze(-2)
            r_gauge = phase.conj().unsqueeze(-1) * r
            q_gauge = torch.where(
                active[:, None, None], q_gauge, torch.zeros_like(q_gauge)
            )
            r_gauge = torch.where(
                active[:, None, None], r_gauge, torch.zeros_like(r_gauge)
            )
            q_blocks.append(q_gauge)
            r_blocks.append(r_gauge)

            core_raw = r_gauge @ core_hats[block_index]
            core_norm = torch.linalg.matrix_norm(
                core_raw, ord="fro", dim=(-2, -1)
            ).to(torch.float64)
            core_valid = active & torch.isfinite(core_norm) & (core_norm > floor)
            if bool(torch.any(core_valid).item()):
                positions = torch.nonzero(core_valid, as_tuple=False).flatten()
                core_hats[block_index][positions] = (
                    core_raw.index_select(0, positions)
                    / core_norm.index_select(0, positions)[:, None, None].to(
                        self.dtype
                    )
                )
                core_scales[block_index][positions] += torch.log(
                    core_norm.index_select(0, positions)
                )
            core_bad = active & ~core_valid
            if bool(torch.any(core_bad).item()):
                positions = torch.nonzero(core_bad, as_tuple=False).flatten()
                core_hats[block_index][positions] = 0.0
                core_scales[block_index][positions] = -torch.inf
                core_null_counts[block_index][positions] += 1
            start = stop

        lyapunov_state["frame"] = torch.cat(q_blocks, dim=-1)
        lyapunov_state["last_r_blocks"] = tuple(r_blocks)
        lyapunov_state["last_r"] = torch.block_diag(
            *[block[0] for block in r_blocks]
        ).unsqueeze(0)
        lyapunov_state["last_cycle_null_mask"] = torch.cat(null_masks, dim=-1)
        lyapunov_state["block_core_hat"] = tuple(core_hats)
        lyapunov_state["block_core_log_scale"] = tuple(core_scales)
        lyapunov_state["block_core_null_count"] = tuple(core_null_counts)
        spectrum = lyapunov_state["log_diag"] / float(cycle)
        return torch.sort(spectrum, dim=-1).values

    def _lyapunov_frame_payload(self, lyapunov_state):
        if lyapunov_state is None:
            return {}
        payload = {
            "lyapunov_frame": lyapunov_state["frame"],
            "lyapunov_qr_r": lyapunov_state["last_r"],
            "lyapunov_log_diag": lyapunov_state["log_diag"],
            "lyapunov_cycle_null_mask": lyapunov_state[
                "last_cycle_null_mask"
            ],
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

    def _reset_lyapunov_accumulator(self, lyapunov_state):
        """Reset finite-window products while preserving the aligned QR frame."""
        lyapunov_state["log_diag"].zero_()
        lyapunov_state["null_counts"].zero_()
        if lyapunov_state.get("basis_mode") == "pure_occupied_empty":
            core_hats = []
            core_scales = []
            core_null_counts = []
            for core, scale, null_count in zip(
                lyapunov_state["block_core_hat"],
                lyapunov_state["block_core_log_scale"],
                lyapunov_state["block_core_null_count"],
            ):
                identity = torch.eye(
                    core.shape[-1], dtype=core.dtype, device=core.device
                ).expand_as(core).clone()
                core_hats.append(identity)
                core_scales.append(torch.zeros_like(scale))
                core_null_counts.append(torch.zeros_like(null_count))
            lyapunov_state["block_core_hat"] = tuple(core_hats)
            lyapunov_state["block_core_log_scale"] = tuple(core_scales)
            lyapunov_state["block_core_null_count"] = tuple(core_null_counts)
        elif lyapunov_state.get("core_hat") is not None:
            core = lyapunov_state["core_hat"]
            lyapunov_state["core_hat"] = torch.eye(
                core.shape[-1], dtype=core.dtype, device=core.device
            ).expand_as(core).clone()
            lyapunov_state["core_log_scale"].zero_()

    def _lyapunov_min_abs_vector_payload(self, lyapunov_state, cycle):
        if lyapunov_state is None:
            return {}
        spectrum = lyapunov_state["log_diag"] / float(cycle)
        min_abs_index = torch.argmin(torch.abs(spectrum), dim=-1)
        gather_index = min_abs_index[:, None, None].expand(
            -1, self.Nlayer, 1
        )
        vector = torch.gather(
            lyapunov_state["frame"], dim=2, index=gather_index
        ).squeeze(-1)
        basis_idx = lyapunov_state.get("basis_idx")
        if basis_idx is not None and int(basis_idx.numel()) != self.Nlayer:
            vector = vector.index_select(1, basis_idx)
        value = torch.gather(
            spectrum, dim=1, index=min_abs_index[:, None]
        ).squeeze(1)
        return {
            "lyapunov_min_abs_vector": vector,
            "lyapunov_min_abs_value": value,
            "lyapunov_min_abs_index": min_abs_index,
            "lyapunov_null_counts": lyapunov_state["null_counts"].clone(),
        }

    def _occ_prob_from_vector_batch(self, G, chi):
        if chi.ndim == 1:
            gv = torch.matmul(G, chi)
            occ = 0.5 * (1.0 + torch.sum(gv * chi.conj(), dim=-1).real)
        else:
            gv = torch.matmul(G, chi.unsqueeze(-1)).squeeze(-1)
            occ = 0.5 * (1.0 + torch.sum(gv * chi.conj(), dim=-1).real)
        return torch.clamp(occ, 0.0, 1.0)

    def _projector_from_vector(self, chi):
        if chi.ndim == 1:
            return chi[:, None] * chi.conj()[None, :]
        return chi.unsqueeze(-1) * chi.conj().unsqueeze(-2)

    def _measure_rank_one_sitewise_batched(self, G, chi, particle):
        """Apply different rank-one measurement channels to every batch row.

        ``_apply_grouped_site_updates`` historically grouped trajectories by their
        current random-schedule site.  For independent permutations, those groups are
        almost always singletons on production lattices, so the GPU effectively ran the
        trajectories serially.  This algebraically equivalent rank-one form accepts one
        normalized controller vector and one realized branch per trajectory and keeps the
        complete trajectory batch in a single CUDA kernel sequence.
        """
        if G.shape[0] == 0:
            return G
        batch = int(G.shape[0])
        chi = chi.to(dtype=self.dtype, device=self.device)
        if chi.ndim == 1:
            chi = chi.unsqueeze(0).expand(batch, -1)
        particle = torch.as_tensor(
            particle, dtype=torch.bool, device=self.device
        ).reshape(-1)
        if particle.numel() == 1:
            particle = particle.expand(batch)
        if int(particle.numel()) != batch:
            raise ValueError("particle must contain one branch value per trajectory")

        sign = torch.where(
            particle,
            torch.ones((batch,), dtype=self.real_dtype, device=self.device),
            -torch.ones((batch,), dtype=self.real_dtype, device=self.device),
        )
        chi_column = chi.unsqueeze(-1)
        chi_row = chi.conj().unsqueeze(-2)
        g_chi = torch.matmul(G, chi_column).squeeze(-1)
        # G Q, with Q = 1 - |chi><chi|, without constructing a dense projector.
        g_q = G - g_chi.unsqueeze(-1) * chi_row
        z = self._physical_rank1_resolvent_action_batched(
            G, chi, g_q, sign
        )
        chi_z = torch.matmul(chi_row, z).squeeze(1)
        measured = (
            sign.to(self.dtype)[:, None, None]
            * chi_column
            * chi_row
            + z
            - chi_column * chi_z.unsqueeze(-2)
        )
        return 0.5 * (measured + measured.conj().transpose(-2, -1))

    def _reset_rank_one_sitewise_batched(self, G, chi, reset_covariance):
        """Replace one different rank-one mode in every trajectory batch row."""
        if G.shape[0] == 0:
            return G
        batch = int(G.shape[0])
        chi = chi.to(dtype=self.dtype, device=self.device)
        if chi.ndim == 1:
            chi = chi.unsqueeze(0).expand(batch, -1)
        reset_covariance = torch.as_tensor(
            reset_covariance, dtype=self.real_dtype, device=self.device
        ).reshape(-1)
        if reset_covariance.numel() == 1:
            reset_covariance = reset_covariance.expand(batch)
        if int(reset_covariance.numel()) != batch:
            raise ValueError(
                "reset_covariance must contain one value per trajectory"
            )

        chi_column = chi.unsqueeze(-1)
        chi_row = chi.conj().unsqueeze(-2)
        g_chi = torch.matmul(G, chi_column).squeeze(-1)
        chi_g = torch.matmul(chi_row, G).squeeze(1)
        scalar = torch.sum(chi.conj() * g_chi, dim=-1)
        projector = chi_column * chi_row
        reset = (
            G
            - chi_column * chi_g.unsqueeze(-2)
            - g_chi.unsqueeze(-1) * chi_row
            + (scalar + reset_covariance.to(self.dtype))[:, None, None]
            * projector
        )
        return 0.5 * (reset + reset.conj().transpose(-2, -1))

    def _lyapunov_apply_rank_one_sitewise(
        self, lyapunov_state, G, chi, particle
    ):
        """Apply the fixed-branch tangent map without dense projectors or solves."""
        if lyapunov_state is None:
            return
        frame_physical = isinstance(G, BatchedOccupiedFrameState)
        batch = G.batch_size if frame_physical else int(G.shape[0])
        chi = chi.to(dtype=self.dtype, device=self.device)
        if chi.ndim == 1:
            chi = chi.unsqueeze(0).expand(batch, -1)
        particle = torch.as_tensor(
            particle, dtype=torch.bool, device=self.device
        ).reshape(-1)
        if particle.numel() == 1:
            particle = particle.expand(batch)
        sign = torch.where(
            particle,
            torch.ones((batch,), dtype=self.real_dtype, device=self.device),
            -torch.ones((batch,), dtype=self.real_dtype, device=self.device),
        )

        if frame_physical:
            coefficients = torch.einsum(
                "bnr,bn->br", G.frame.conj(), chi
            ) * G._active_mask()
            g_chi = 2.0 * torch.einsum(
                "bnr,br->bn", G.frame, coefficients
            ) - chi
        else:
            g_chi = torch.matmul(G, chi.unsqueeze(-1)).squeeze(-1)
        g = torch.sum(chi.conj() * g_chi, dim=-1).real
        raw_denominator = 1.0 + sign * g
        denominator = torch.abs(raw_denominator).to(torch.float64)
        probability = 0.5 * denominator
        active = lyapunov_state["active"]
        lyapunov_state["min_abs_born_denominator"] = torch.where(
            active,
            torch.minimum(
                lyapunov_state["min_abs_born_denominator"], denominator
            ),
            lyapunov_state["min_abs_born_denominator"],
        )
        lyapunov_state["min_branch_probability"] = torch.where(
            active,
            torch.minimum(
                lyapunov_state["min_branch_probability"], probability
            ),
            lyapunov_state["min_branch_probability"],
        )
        valid = (
            active
            & torch.isfinite(denominator)
            & (denominator > float(lyapunov_state["singular_tol"]))
        )
        invalid = active & ~valid
        lyapunov_state["invalid_branch_count"] += invalid.to(torch.int64)
        lyapunov_state["active"] = valid
        self._lyapunov_accumulate_record_fisher_sitewise(
            lyapunov_state, chi, g, valid
        )

        safe_denominator = torch.where(
            valid, raw_denominator, torch.ones_like(raw_denominator)
        ).to(self.dtype)
        frame = lyapunov_state["frame"]
        chi_frame = torch.matmul(
            chi.conj().unsqueeze(1), frame
        ).squeeze(1)
        solved = frame - (
            sign.to(self.dtype) / safe_denominator
        )[:, None, None] * g_chi.unsqueeze(-1) * chi_frame.unsqueeze(-2)
        chi_solved = torch.matmul(
            chi.conj().unsqueeze(1), solved
        ).squeeze(1)
        updated = solved - chi.unsqueeze(-1) * chi_solved.unsqueeze(-2)
        lyapunov_state["frame"] = torch.where(
            valid[:, None, None], updated, torch.zeros_like(updated)
        )

    def _lyapunov_accumulate_record_fisher_sitewise(
        self, lyapunov_state, chi, g, valid_mask
    ):
        if lyapunov_state is None or not lyapunov_state.get(
            "track_record_fisher", False
        ):
            return
        frame = lyapunov_state["frame"]
        core_hat = lyapunov_state["core_hat"]
        image_hat = torch.matmul(frame, core_hat)
        chi_image = torch.matmul(
            chi.conj().unsqueeze(1), image_hat
        ).squeeze(1)
        overlap = chi_image.conj().unsqueeze(2) * chi_image.unsqueeze(1)
        d = torch.stack(
            (
                2.0 * overlap[:, 0, 1].real,
                -2.0 * overlap[:, 0, 1].imag,
                (overlap[:, 0, 0] - overlap[:, 1, 1]).real,
            ),
            dim=-1,
        ).to(torch.float64)
        variance = torch.clamp(1.0 - g.to(torch.float64) ** 2, min=0.0)
        d_norm = torch.linalg.vector_norm(d, dim=-1)
        tol = float(lyapunov_state["singular_tol"])
        endpoint = valid_mask & (variance <= tol)
        lyapunov_state["record_fisher_endpoint_count"] += endpoint.to(torch.int64)
        lyapunov_state["record_fisher_infinite"] |= endpoint & (
            d_norm > np.sqrt(tol)
        )
        gamma = lyapunov_state["core_log_scale"]
        accumulate = (
            valid_mask
            & (variance > tol)
            & torch.isfinite(gamma)
            & (d_norm > 0.0)
        )
        safe_variance = torch.where(
            accumulate, variance, torch.ones_like(variance)
        )
        increment_hat = d.unsqueeze(-1) * d.unsqueeze(-2)
        increment_log_scale = 4.0 * gamma - torch.log(safe_variance)
        old_log_scale = lyapunov_state["record_fisher_log_scale"]
        new_log_scale = torch.maximum(old_log_scale, increment_log_scale)
        old_weight = torch.where(
            torch.isfinite(old_log_scale),
            torch.exp(old_log_scale - new_log_scale),
            torch.zeros_like(old_log_scale),
        )
        new_weight = torch.exp(increment_log_scale - new_log_scale)
        updated_hat = (
            old_weight[:, None, None] * lyapunov_state["record_fisher_hat"]
            + new_weight[:, None, None] * increment_hat
        )
        lyapunov_state["record_fisher_hat"] = torch.where(
            accumulate[:, None, None],
            updated_hat,
            lyapunov_state["record_fisher_hat"],
        )
        lyapunov_state["record_fisher_log_scale"] = torch.where(
            accumulate, new_log_scale, old_log_scale
        )

    def _lyapunov_reset_rank_one_sitewise(
        self, lyapunov_state, chi, reset_mask
    ):
        if lyapunov_state is None:
            return
        chi = chi.to(dtype=self.dtype, device=self.device)
        frame = lyapunov_state["frame"]
        chi_frame = torch.matmul(
            chi.conj().unsqueeze(1), frame
        ).squeeze(1)
        reset = frame - chi.unsqueeze(-1) * chi_frame.unsqueeze(-2)
        apply = (
            torch.as_tensor(reset_mask, dtype=torch.bool, device=self.device)
            & lyapunov_state["active"]
        )
        lyapunov_state["frame"] = torch.where(
            apply[:, None, None], reset, frame
        )

    def _apply_feedback_sitewise_batched(
        self,
        G,
        site_ids,
        *,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        record_observer=None,
        outcome_overrides=None,
        lyapunov_state=None,
        cycle=None,
        update_index=None,
        batch_index=None,
        batch_start=None,
    ):
        """Batched feedback path for independent per-trajectory random sites."""
        batch = int(G.shape[0])
        site_ids = site_ids.to(dtype=torch.long, device=self.device).reshape(-1)
        if int(site_ids.numel()) != batch:
            raise ValueError("site_ids must contain one site per trajectory")
        spinors = {
            "Ap": self.WF_Ap_sites.index_select(0, site_ids),
            "Am": self.WF_Am_sites.index_select(0, site_ids),
            "Bp": self.WF_Bp_sites.index_select(0, site_ids),
            "Bm": self.WF_Bm_sites.index_select(0, site_ids),
        }
        specs = (
            ("Ap", False),
            ("Am", True),
            ("Bp", False),
            ("Bm", True),
        )
        _, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        forced = None
        if outcome_overrides is not None:
            forced = torch.as_tensor(
                outcome_overrides, dtype=torch.bool, device=self.device
            )
            if tuple(forced.shape) != (batch, len(specs)):
                raise ValueError(
                    "outcome_overrides must have shape (batch, 4) on the "
                    "sitewise perfect-correction path"
                )

        probability_rows = []
        outcome_rows = []
        target_rows = []
        reset_rows = []
        for channel_index, (label, target_occupied) in enumerate(specs):
            chi = spinors[label]
            p_occ = self._occ_prob_from_vector_batch(G, chi)
            outcome = (
                torch.rand(
                    (batch,), device=self.device, dtype=self.real_dtype
                )
                < p_occ
                if forced is None
                else forced[:, channel_index]
            )
            self._lyapunov_apply_rank_one_sitewise(
                lyapunov_state, G, chi, outcome
            )
            G_measured = self._measure_rank_one_sitewise_batched(
                G, chi, outcome
            )
            mismatch = outcome != bool(target_occupied)
            self._lyapunov_reset_rank_one_sitewise(
                lyapunov_state, chi, mismatch
            )
            target_sign = 1.0 if target_occupied else -1.0
            if perfect_correction:
                reset_covariance = torch.full(
                    (batch,),
                    target_sign,
                    dtype=self.real_dtype,
                    device=self.device,
                )
            else:
                correction_probability = (
                    p_gain_eff if target_occupied else p_loss_eff
                )
                correction_succeeded = torch.rand(
                    (batch,), device=self.device, dtype=self.real_dtype
                ) < float(correction_probability)
                reset_covariance = torch.where(
                    correction_succeeded,
                    torch.full(
                        (batch,),
                        target_sign,
                        dtype=self.real_dtype,
                        device=self.device,
                    ),
                    torch.full(
                        (batch,),
                        -target_sign,
                        dtype=self.real_dtype,
                        device=self.device,
                    ),
                )
            G_reset = self._reset_rank_one_sitewise_batched(
                G_measured,
                chi,
                reset_covariance,
            )
            G = torch.where(
                mismatch[:, None, None], G_reset, G_measured
            )
            probability_rows.append(p_occ.to(torch.float64))
            outcome_rows.append(outcome)
            target_rows.append(
                torch.full(
                    (batch,),
                    bool(target_occupied),
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            reset_rows.append(
                torch.where(
                    mismatch,
                    reset_covariance,
                    torch.where(
                        outcome,
                        torch.ones_like(p_occ),
                        -torch.ones_like(p_occ),
                    ),
                ).to(torch.float64)
            )

        if record_observer is not None:
            p_occ = torch.stack(probability_rows, dim=1)
            outcomes = torch.stack(outcome_rows, dim=1)
            targets = torch.stack(target_rows, dim=1)
            realized_probability = torch.where(
                outcomes, p_occ, 1.0 - p_occ
            ).clamp(0.0, 1.0)
            tiny = torch.finfo(torch.float64).tiny
            record_observer(
                cycle=int(cycle),
                update_index=int(update_index),
                site_ids=site_ids,
                sample_offsets=torch.arange(
                    batch, dtype=torch.long, device=self.device
                ),
                sample_indices=torch.arange(
                    int(batch_start),
                    int(batch_start) + batch,
                    dtype=torch.long,
                    device=self.device,
                ),
                batch_index=int(batch_index),
                batch_start=int(batch_start),
                batch_count=batch,
                channel_labels=tuple(label for label, _ in specs),
                occupation_probability=p_occ,
                outcome_occupied=outcomes,
                target_occupied=targets,
                transfer=targets.to(torch.int8) - outcomes.to(torch.int8),
                realized_probability=realized_probability,
                conditional_log_probability=torch.log(
                    realized_probability.clamp_min(tiny)
                ),
                target_success_probability=torch.where(
                    targets, p_occ, 1.0 - p_occ
                ),
                reset_covariance=torch.stack(reset_rows, dim=1),
            )
        return G

    def _local_support_block(self, G, support_idx, comp_idx):
        G_ss = G.index_select(1, support_idx).index_select(2, support_idx)
        if comp_idx.numel() == 0:
            G_sr = G_ss.new_empty((G.shape[0], support_idx.numel(), 0))
        else:
            G_sr = G.index_select(1, support_idx).index_select(2, comp_idx)
        return G_ss, G_sr

    def _occ_prob_from_local_support_batch(self, G, support_idx, chi_local):
        G_ss, _ = self._local_support_block(G, support_idx, self._all_top_indices[:0])
        return self._occ_prob_from_vector_batch(G_ss, chi_local)

    def _measure_only_top_layer_local_batched(self, G, support_idx, comp_idx, chi_local, particle=True):
        G_ss, G_sr = self._local_support_block(G, support_idx, comp_idx)
        G_ss_new, G_sr_new, delta_rr = self._physical_rank1_measure_blocks_batched(
            G_ss,
            G_sr,
            chi_local,
            particle=particle,
        )

        G[:, support_idx[:, None], support_idx[None, :]] = G_ss_new
        if comp_idx.numel() == 0:
            return G

        G[:, comp_idx[:, None], comp_idx[None, :]] += delta_rr
        G[:, support_idx[:, None], comp_idx[None, :]] = G_sr_new
        G[:, comp_idx[:, None], support_idx[None, :]] = G_sr_new.conj().transpose(-2, -1)
        return G

    def _prepare_exterior_product_state_batched(
        self,
        G,
        mode="born_conditioned",
        *,
        outcome_observer=None,
        outcome_overrides=None,
        batch_start=0,
    ):
        """Prepare exterior canonical orbitals once, without OW feedback."""
        mode = str(mode).strip().lower()
        if mode not in ("born_conditioned", "forced_occupied"):
            raise ValueError(
                "exterior preparation mode must be 'born_conditioned' or "
                "'forced_occupied'."
            )
        chi_local = torch.ones((1,), dtype=self.dtype, device=self.device)
        overrides = (
            None
            if outcome_overrides is None
            else torch.as_tensor(
                outcome_overrides, dtype=torch.bool, device=self.device
            )
        )
        outcome_rows = []
        orbital_indices = []
        event_index = 0
        for site_id in self._exterior_site_ids():
            _, _, idx_a, idx_b = self._canonical_site_vectors(site_id)
            for idx in (idx_a, idx_b):
                support_idx = torch.as_tensor([idx], dtype=torch.long, device=self.device)
                comp_idx = self._complement_indices(support_idx)
                orbital_indices.append(int(idx))
                if mode == "forced_occupied":
                    occupied = torch.ones(
                        (G.shape[0],), dtype=torch.bool, device=self.device
                    )
                    G = self._measure_only_top_layer_local_batched(
                        G, support_idx, comp_idx, chi_local, particle=True
                    )
                    outcome_rows.append(occupied)
                    event_index += 1
                    continue
                occ_prob = torch.clamp(0.5 * (1.0 + G[:, idx, idx].real), min=0.0, max=1.0)
                occupied = (
                    overrides[:, event_index]
                    if overrides is not None
                    else torch.rand(
                        (G.shape[0],), dtype=self.real_dtype, device=self.device
                    ) < occ_prob
                )
                for particle in (False, True):
                    selected = torch.nonzero(occupied == particle, as_tuple=False).flatten()
                    if selected.numel() == 0:
                        continue
                    G_sel = G.index_select(0, selected).clone()
                    G_sel = self._measure_only_top_layer_local_batched(
                        G_sel, support_idx, comp_idx, chi_local, particle=particle
                    )
                    G[selected] = G_sel
                outcome_rows.append(occupied)
                event_index += 1
        if outcome_observer is not None and outcome_rows:
            outcome_observer(
                orbital_indices=torch.as_tensor(
                    orbital_indices, dtype=torch.long, device=self.device
                ),
                outcome_occupied=torch.stack(outcome_rows, dim=1),
                sample_indices=torch.arange(
                    int(batch_start), int(batch_start) + G.shape[0],
                    dtype=torch.long, device=self.device
                ),
                batch_start=int(batch_start),
                batch_count=int(G.shape[0]),
            )
        return G

    def _ancilla_swap_top_local_batched(
        self,
        G,
        support_idx,
        comp_idx,
        chi_local,
        n_a,
        mismatch_mask,
        target_occupied=None,
        return_ancilla_cov=False,
        ancilla_cov_override=None,
    ):
        if not torch.any(mismatch_mask):
            if return_ancilla_cov:
                empty_idx = torch.empty((0,), dtype=torch.long, device=self.device)
                empty_cov = torch.empty((0,), dtype=self.real_dtype, device=self.device)
                return G, empty_idx, empty_cov
            return G

        batch_idx = torch.nonzero(mismatch_mask, as_tuple=False).flatten()
        G_sel = G.index_select(0, batch_idx)
        G_ss, G_sr = self._local_support_block(G_sel, support_idx, comp_idx)
        p = self._projector_from_vector(chi_local)
        q = self._eye_of_size(support_idx.numel()) - p

        if ancilla_cov_override is not None:
            ancilla_cov = torch.as_tensor(ancilla_cov_override, device=self.device, dtype=self.real_dtype).reshape(-1)
            if ancilla_cov.numel() != batch_idx.numel():
                raise ValueError("ancilla_cov_override length must match the selected mismatch count.")
        elif target_occupied is None:
            ancilla_cov = torch.where(
                torch.rand((batch_idx.numel(),), device=self.device, dtype=self.real_dtype) < float(n_a),
                torch.ones((batch_idx.numel(),), device=self.device, dtype=self.real_dtype),
                -torch.ones((batch_idx.numel(),), device=self.device, dtype=self.real_dtype),
            )
        else:
            ancilla_cov = torch.full(
                (batch_idx.numel(),),
                1.0 if bool(target_occupied) else -1.0,
                device=self.device,
                dtype=self.real_dtype,
            )

        G_sr_new = torch.matmul(q.unsqueeze(0), G_sr)
        G_ss_new = torch.matmul(q.unsqueeze(0), torch.matmul(G_ss, q.unsqueeze(0)))
        G_ss_new = G_ss_new + ancilla_cov[:, None, None].to(self.dtype) * p.unsqueeze(0)
        G_ss_new = 0.5 * (G_ss_new + G_ss_new.conj().transpose(-2, -1))

        G_sel[:, support_idx[:, None], support_idx[None, :]] = G_ss_new
        if comp_idx.numel() > 0:
            G_sel[:, support_idx[:, None], comp_idx[None, :]] = G_sr_new
            G_sel[:, comp_idx[:, None], support_idx[None, :]] = G_sr_new.conj().transpose(-2, -1)
        G[batch_idx] = G_sel
        if return_ancilla_cov:
            return G, batch_idx, ancilla_cov
        return G

    def _measure_only_top_layer_batched(self, G, P, particle=True, chi=None):
        eye = self._eye_top if P.ndim == 2 else self._eye_top.unsqueeze(0)
        Q = eye - P
        chi_vec = self._rank_one_vector_from_projector_batched(P) if chi is None else chi
        sign_value = 1.0 if bool(particle) else -1.0
        sign = torch.full((G.shape[0],), sign_value, dtype=self.real_dtype, device=self.device)
        Z = self._physical_rank1_resolvent_action_batched(G, chi_vec, torch.matmul(G, Q), sign)
        G_upd = sign_value * P + torch.matmul(Q, Z)
        return 0.5 * (G_upd + G_upd.conj().transpose(-2, -1))

    def _ancilla_swap_top_batched(
        self,
        G,
        P,
        n_a,
        mismatch_mask,
        target_occupied=None,
        return_ancilla_cov=False,
        ancilla_cov_override=None,
    ):
        if not torch.any(mismatch_mask):
            if return_ancilla_cov:
                empty_idx = torch.empty((0,), dtype=torch.long, device=self.device)
                empty_cov = torch.empty((0,), dtype=self.real_dtype, device=self.device)
                return G, empty_idx, empty_cov
            return G

        idx = torch.nonzero(mismatch_mask, as_tuple=False).flatten()
        G_sel = G.index_select(0, idx)
        batch_sel = G_sel.shape[0]
        if P.ndim == 2:
            P_batch = P.unsqueeze(0).expand(batch_sel, -1, -1)
        else:
            P_batch = P.index_select(0, idx)
        Q_batch = self._eye_top.unsqueeze(0) - P_batch

        if ancilla_cov_override is not None:
            ancilla_cov = torch.as_tensor(ancilla_cov_override, device=self.device, dtype=self.real_dtype).reshape(-1)
            if ancilla_cov.numel() != batch_sel:
                raise ValueError("ancilla_cov_override length must match the selected mismatch count.")
        elif target_occupied is None:
            ancilla_cov = torch.where(
                torch.rand((batch_sel,), device=self.device, dtype=self.real_dtype) < float(n_a),
                torch.ones((batch_sel,), device=self.device, dtype=self.real_dtype),
                -torch.ones((batch_sel,), device=self.device, dtype=self.real_dtype),
            )
        else:
            ancilla_cov = torch.full(
                (batch_sel,),
                1.0 if bool(target_occupied) else -1.0,
                device=self.device,
                dtype=self.real_dtype,
            )

        G_new = torch.matmul(Q_batch, torch.matmul(G_sel, Q_batch)) + ancilla_cov[:, None, None].to(self.dtype) * P_batch
        G_new = 0.5 * (G_new + G_new.conj().transpose(-2, -1))
        G[idx] = G_new
        if return_ancilla_cov:
            return G, idx, ancilla_cov
        return G

    def _site_spinors(self, site_id):
        if self._site_uses_local_mode(site_id):
            e_a, e_b, _, _ = self._canonical_site_vectors(site_id)
            return e_a, e_b, e_a, e_b
        chi_Ap = self.WF_Ap_sites[site_id]
        chi_Bp = self.WF_Bp_sites[site_id]
        chi_Am = self.WF_Am_sites[site_id]
        chi_Bm = self.WF_Bm_sites[site_id]
        return chi_Ap, chi_Bp, chi_Am, chi_Bm

    def _site_local_payload(self, site_id):
        if self._local_site_cache is None:
            raise RuntimeError("Local support cache is not available for this model.")
        return self._local_site_cache[int(site_id)]

    def cycle_end_born_probabilities_batch(self, G):
        if G.ndim != 3:
            raise ValueError(f"Expected G with shape (B,N,N), got {tuple(G.shape)}")
        batch_count, nrow, ncol = G.shape
        if nrow != ncol:
            raise ValueError(f"Expected square covariance batch, got {tuple(G.shape)}")
        if nrow == self.Ntot:
            G = G[:, :self.Nlayer, :self.Nlayer]
        elif nrow != self.Nlayer:
            raise ValueError(
                f"Expected covariance shape (B,{self.Nlayer},{self.Nlayer}) "
                f"or (B,{self.Ntot},{self.Ntot}), got {tuple(G.shape)}"
            )
        if not torch.isfinite(G).all():
            raise FloatingPointError("Non-finite values encountered in cycle_end_born_probabilities_batch.")

        probs_ap = torch.empty((batch_count, self.Nx * self.Ny), dtype=self.real_dtype, device=self.device)
        probs_bp = torch.empty_like(probs_ap)
        probs_am = torch.empty_like(probs_ap)
        probs_bm = torch.empty_like(probs_ap)

        if self.backend == "local" and self._local_site_cache is not None:
            for site_id, payload in enumerate(self._local_site_cache):
                if payload["mode"] == "local_mode":
                    idx_a = int(payload["idx"][0].item())
                    idx_b = int(payload["idx"][1].item())
                    occ_a = 0.5 * (1.0 + G[:, idx_a, idx_a].real)
                    occ_b = 0.5 * (1.0 + G[:, idx_b, idx_b].real)
                    probs_ap[:, site_id] = torch.clamp(occ_a, 0.0, 1.0)
                    probs_am[:, site_id] = torch.clamp(occ_a, 0.0, 1.0)
                    probs_bp[:, site_id] = torch.clamp(occ_b, 0.0, 1.0)
                    probs_bm[:, site_id] = torch.clamp(occ_b, 0.0, 1.0)
                    continue

                support_idx = payload["idx"]
                probs_ap[:, site_id] = self._occ_prob_from_local_support_batch(G, support_idx, payload["Ap"])
                probs_bp[:, site_id] = self._occ_prob_from_local_support_batch(G, support_idx, payload["Bp"])
                probs_am[:, site_id] = self._occ_prob_from_local_support_batch(G, support_idx, payload["Am"])
                probs_bm[:, site_id] = self._occ_prob_from_local_support_batch(G, support_idx, payload["Bm"])
        else:
            for site_id in range(self.Nx * self.Ny):
                chi_ap, chi_bp, chi_am, chi_bm = self._site_spinors(site_id)
                probs_ap[:, site_id] = self._occ_prob_from_vector_batch(G, chi_ap)
                probs_bp[:, site_id] = self._occ_prob_from_vector_batch(G, chi_bp)
                probs_am[:, site_id] = self._occ_prob_from_vector_batch(G, chi_am)
                probs_bm[:, site_id] = self._occ_prob_from_vector_batch(G, chi_bm)

        probs_ap = probs_ap.view(batch_count, self.Nx, self.Ny)
        probs_bp = probs_bp.view(batch_count, self.Nx, self.Ny)
        probs_am = probs_am.view(batch_count, self.Nx, self.Ny)
        probs_bm = probs_bm.view(batch_count, self.Nx, self.Ny)

        return {
            "N_A_lower": torch.clamp(probs_am, 0.0, 1.0).to(torch.float64),
            "N_B_lower": torch.clamp(probs_bm, 0.0, 1.0).to(torch.float64),
            "one_minus_N_A_upper": torch.clamp(1.0 - probs_ap, 0.0, 1.0).to(torch.float64),
            "one_minus_N_B_upper": torch.clamp(1.0 - probs_bp, 0.0, 1.0).to(torch.float64),
        }

    def _apply_channel_shared_site(
        self,
        G,
        chi,
        expected_occupied,
        sample_offsets=None,
        lyapunov_state=None,
        choi_state=None,
        choi_context=None,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        return_occ_prob=False,
        return_event_data=False,
        forced_outcome_occupied=None,
    ):
        if G.shape[0] == 0:
            if return_event_data:
                empty_prob = torch.empty((0,), device=self.device, dtype=self.real_dtype)
                empty_bool = torch.empty((0,), device=self.device, dtype=torch.bool)
                return G, {
                    "p_occ": empty_prob,
                    "outcome_occupied": empty_bool,
                    "reset_covariance": empty_prob,
                }
            if return_occ_prob:
                return G, torch.empty((0,), device=self.device, dtype=self.real_dtype)
            return G

        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )

        def _sample_reset_cov(count):
            count = int(count)
            if perfect_correction:
                return None
            if bool(expected_occupied):
                success = torch.rand((count,), device=self.device, dtype=self.real_dtype) < p_gain_eff
                return torch.where(
                    success,
                    torch.ones((count,), device=self.device, dtype=self.real_dtype),
                    -torch.ones((count,), device=self.device, dtype=self.real_dtype),
                )
            success = torch.rand((count,), device=self.device, dtype=self.real_dtype) < p_loss_eff
            return torch.where(
                success,
                -torch.ones((count,), device=self.device, dtype=self.real_dtype),
                torch.ones((count,), device=self.device, dtype=self.real_dtype),
            )

        P = self._projector_from_vector(chi)
        p_occ = self._occ_prob_from_vector_batch(G, chi)
        if forced_outcome_occupied is None:
            occ_event = torch.rand(
                (G.shape[0],), device=self.device, dtype=self.real_dtype
            ) < p_occ
        else:
            occ_event = torch.as_tensor(
                forced_outcome_occupied, dtype=torch.bool, device=self.device
            ).reshape(-1)
            if int(occ_event.numel()) != int(G.shape[0]):
                raise ValueError("forced channel outcomes must contain one bit per trajectory")
        if lyapunov_state is not None or choi_state is not None:
            if sample_offsets is None:
                sample_offsets = torch.arange(G.shape[0], dtype=torch.long, device=self.device)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(lyapunov_state, G, sample_offsets, P, occ_event)
        s_in = torch.where(
            occ_event,
            torch.ones_like(p_occ),
            -torch.ones_like(p_occ),
        )
        s_out = s_in.clone()

        idx_occ = torch.nonzero(occ_event, as_tuple=False).flatten()
        idx_unocc = torch.nonzero(~occ_event, as_tuple=False).flatten()

        if idx_occ.numel() > 0:
            G[idx_occ] = self._measure_only_top_layer_batched(
                G.index_select(0, idx_occ), P, particle=True, chi=chi
            )
        if idx_unocc.numel() > 0:
            G[idx_unocc] = self._measure_only_top_layer_batched(
                G.index_select(0, idx_unocc), P, particle=False, chi=chi
            )

        mismatch_mask = occ_event if not expected_occupied else ~occ_event
        if torch.any(mismatch_mask):
            reset_positions = torch.nonzero(
                mismatch_mask, as_tuple=False
            ).flatten()
            if lyapunov_state is not None:
                reset_offsets = sample_offsets.index_select(
                    0, reset_positions
                )
                self._lyapunov_apply_dense_reset(
                    lyapunov_state, reset_offsets, P
                )
            target = expected_occupied if perfect_correction else None
            reset_cov = _sample_reset_cov(torch.count_nonzero(mismatch_mask).item())
            if choi_state is not None:
                G, reset_idx, reset_cov_actual = self._ancilla_swap_top_batched(
                    G,
                    P,
                    n_a=n_a,
                    mismatch_mask=mismatch_mask,
                    target_occupied=target,
                    return_ancilla_cov=True,
                    ancilla_cov_override=reset_cov,
                )
                s_out[reset_idx] = reset_cov_actual
            else:
                G = self._ancilla_swap_top_batched(
                    G,
                    P,
                    n_a=n_a,
                    mismatch_mask=mismatch_mask,
                    target_occupied=target,
                    ancilla_cov_override=reset_cov,
                )
        if choi_state is not None:
            self._choi_apply_rank_one(
                choi_state,
                sample_offsets,
                chi,
                eta1=-s_in,
                eta2=s_out,
                context=choi_context,
            )
        if return_event_data:
            return G, {
                "p_occ": p_occ,
                "outcome_occupied": occ_event,
                "reset_covariance": s_out,
            }
        if return_occ_prob:
            return G, p_occ
        return G

    def _apply_channel_shared_site_local(
        self,
        G,
        site_payload,
        channel_key,
        expected_occupied,
        sample_offsets=None,
        lyapunov_state=None,
        choi_state=None,
        choi_context=None,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        return_occ_prob=False,
        return_event_data=False,
        forced_outcome_occupied=None,
    ):
        if G.shape[0] == 0:
            if return_event_data:
                empty_prob = torch.empty((0,), device=self.device, dtype=self.real_dtype)
                empty_bool = torch.empty((0,), device=self.device, dtype=torch.bool)
                return G, {
                    "p_occ": empty_prob,
                    "outcome_occupied": empty_bool,
                    "reset_covariance": empty_prob,
                }
            if return_occ_prob:
                return G, torch.empty((0,), device=self.device, dtype=self.real_dtype)
            return G

        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )

        def _sample_reset_cov(count):
            count = int(count)
            if perfect_correction:
                return None
            if bool(expected_occupied):
                success = torch.rand((count,), device=self.device, dtype=self.real_dtype) < p_gain_eff
                return torch.where(
                    success,
                    torch.ones((count,), device=self.device, dtype=self.real_dtype),
                    -torch.ones((count,), device=self.device, dtype=self.real_dtype),
                )
            success = torch.rand((count,), device=self.device, dtype=self.real_dtype) < p_loss_eff
            return torch.where(
                success,
                -torch.ones((count,), device=self.device, dtype=self.real_dtype),
                torch.ones((count,), device=self.device, dtype=self.real_dtype),
            )

        support_idx = site_payload["idx"]
        comp_idx = site_payload["comp"]
        chi_local = site_payload[channel_key]

        p_occ = self._occ_prob_from_local_support_batch(G, support_idx, chi_local)
        if forced_outcome_occupied is None:
            occ_event = torch.rand(
                (G.shape[0],), device=self.device, dtype=self.real_dtype
            ) < p_occ
        else:
            occ_event = torch.as_tensor(
                forced_outcome_occupied, dtype=torch.bool, device=self.device
            ).reshape(-1)
            if int(occ_event.numel()) != int(G.shape[0]):
                raise ValueError("forced channel outcomes must contain one bit per trajectory")
        if lyapunov_state is not None or choi_state is not None:
            if sample_offsets is None:
                sample_offsets = torch.arange(G.shape[0], dtype=torch.long, device=self.device)
        if lyapunov_state is not None:
            self._lyapunov_apply_local_channel(
                lyapunov_state,
                G,
                sample_offsets,
                support_idx,
                comp_idx,
                chi_local,
                occ_event,
            )
        s_in = torch.where(
            occ_event,
            torch.ones_like(p_occ),
            -torch.ones_like(p_occ),
        )
        s_out = s_in.clone()

        idx_occ = torch.nonzero(occ_event, as_tuple=False).flatten()
        idx_unocc = torch.nonzero(~occ_event, as_tuple=False).flatten()

        if idx_occ.numel() > 0:
            G[idx_occ] = self._measure_only_top_layer_local_batched(
                G.index_select(0, idx_occ),
                support_idx,
                comp_idx,
                chi_local,
                particle=True,
            )
        if idx_unocc.numel() > 0:
            G[idx_unocc] = self._measure_only_top_layer_local_batched(
                G.index_select(0, idx_unocc),
                support_idx,
                comp_idx,
                chi_local,
                particle=False,
            )

        mismatch_mask = occ_event if not expected_occupied else ~occ_event
        if torch.any(mismatch_mask):
            reset_positions = torch.nonzero(
                mismatch_mask, as_tuple=False
            ).flatten()
            if lyapunov_state is not None:
                reset_offsets = sample_offsets.index_select(
                    0, reset_positions
                )
                self._lyapunov_apply_local_reset(
                    lyapunov_state,
                    reset_offsets,
                    support_idx,
                    chi_local,
                )
            target = expected_occupied if perfect_correction else None
            reset_cov = _sample_reset_cov(torch.count_nonzero(mismatch_mask).item())
            if choi_state is not None:
                G, reset_idx, reset_cov_actual = self._ancilla_swap_top_local_batched(
                    G,
                    support_idx,
                    comp_idx,
                    chi_local,
                    n_a=n_a,
                    mismatch_mask=mismatch_mask,
                    target_occupied=target,
                    return_ancilla_cov=True,
                    ancilla_cov_override=reset_cov,
                )
                s_out[reset_idx] = reset_cov_actual
            else:
                G = self._ancilla_swap_top_local_batched(
                    G,
                    support_idx,
                    comp_idx,
                    chi_local,
                    n_a=n_a,
                    mismatch_mask=mismatch_mask,
                    target_occupied=target,
                    ancilla_cov_override=reset_cov,
                )
        if choi_state is not None:
            self._choi_apply_rank_one(
                choi_state,
                sample_offsets,
                chi_local,
                eta1=-s_in,
                eta2=s_out,
                support_idx=support_idx,
                context=choi_context,
            )
        if return_event_data:
            return G, {
                "p_occ": p_occ,
                "outcome_occupied": occ_event,
                "reset_covariance": s_out,
            }
        if return_occ_prob:
            return G, p_occ
        return G

    def _markov_meas_feedback_shared_site_with_event_data(
        self,
        G,
        site_id,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        sample_offsets=None,
        lyapunov_state=None,
        choi_state=None,
        choi_context=None,
        forced_outcomes=None,
    ):
        """Apply one ordered controller center and expose its realized Born record.

        This is the canonical record-capture path used by production observers.  The
        physical updates are still delegated to the same rank-one channel methods as
        the ordinary Markov path; the returned tensors are read-only diagnostics of
        the draws that were actually applied.
        """

        def channel_context(channel):
            if choi_state is None:
                return None
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = channel
            return payload

        if self.backend == "local":
            site_payload = self._site_local_payload(site_id)
            if site_payload.get("mode") == "local_mode":
                specs = (("A", "A", False), ("B", "B", True))
            else:
                specs = (
                    ("Ap", "Ap", False),
                    ("Am", "Am", True),
                    ("Bp", "Bp", False),
                    ("Bm", "Bm", True),
                )

            event_rows = []
            for channel_index, (label, channel_key, target_occupied) in enumerate(specs):
                G, event = self._apply_channel_shared_site_local(
                    G,
                    site_payload,
                    channel_key,
                    expected_occupied=target_occupied,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                    choi_state=choi_state,
                    choi_context=channel_context(label),
                    n_a=n_a,
                    p_gain=p_gain,
                    p_loss=p_loss,
                    perfect_correction=perfect_correction,
                    return_event_data=True,
                    forced_outcome_occupied=(
                        None
                        if forced_outcomes is None
                        else forced_outcomes[:, channel_index]
                    ),
                )
                event_rows.append(event)
        else:
            chi_ap, chi_bp, chi_am, chi_bm = self._site_spinors(site_id)
            if self._site_uses_local_mode(site_id):
                specs = (("A", chi_ap, False), ("B", chi_bp, True))
            else:
                specs = (
                    ("Ap", chi_ap, False),
                    ("Am", chi_am, True),
                    ("Bp", chi_bp, False),
                    ("Bm", chi_bm, True),
                )

            event_rows = []
            for channel_index, (label, chi, target_occupied) in enumerate(specs):
                G, event = self._apply_channel_shared_site(
                    G,
                    chi,
                    expected_occupied=target_occupied,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                    choi_state=choi_state,
                    choi_context=channel_context(label),
                    n_a=n_a,
                    p_gain=p_gain,
                    p_loss=p_loss,
                    perfect_correction=perfect_correction,
                    return_event_data=True,
                    forced_outcome_occupied=(
                        None
                        if forced_outcomes is None
                        else forced_outcomes[:, channel_index]
                    ),
                )
                event_rows.append(event)

        channel_labels = tuple(spec[0] for spec in specs)
        p_occ = torch.stack([row["p_occ"] for row in event_rows], dim=1).to(torch.float64)
        outcomes = torch.stack([row["outcome_occupied"] for row in event_rows], dim=1)
        targets = torch.as_tensor(
            [spec[2] for spec in specs], dtype=torch.bool, device=self.device
        ).unsqueeze(0).expand(G.shape[0], -1)
        reset_covariance = torch.stack(
            [row["reset_covariance"] for row in event_rows], dim=1
        ).to(torch.float64)
        realized_probability = torch.where(outcomes, p_occ, 1.0 - p_occ)
        realized_probability = torch.clamp(realized_probability, 0.0, 1.0)
        tiny = torch.finfo(torch.float64).tiny
        conditional_log_probability = torch.log(
            realized_probability.clamp_min(tiny)
        )
        transfer = targets.to(torch.int8) - outcomes.to(torch.int8)
        success_probability = torch.where(targets, p_occ, 1.0 - p_occ)

        return G, {
            "channel_labels": channel_labels,
            "occupation_probability": p_occ,
            "outcome_occupied": outcomes,
            "target_occupied": targets,
            "transfer": transfer,
            "realized_probability": realized_probability,
            "conditional_log_probability": conditional_log_probability,
            "target_success_probability": success_probability,
            "reset_covariance": reset_covariance,
        }

    def _markov_meas_feedback_shared_site(
        self,
        G,
        site_id,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        return_success_probs=False,
        return_event_data=False,
        sample_offsets=None,
        lyapunov_state=None,
        choi_state=None,
        choi_context=None,
        forced_outcomes=None,
    ):
        if return_event_data:
            return self._markov_meas_feedback_shared_site_with_event_data(
                G,
                site_id,
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=choi_context,
                forced_outcomes=forced_outcomes,
            )

        def channel_context(channel):
            if choi_state is None:
                return None
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = channel
            return payload

        if self.backend == "local":
            payload = self._site_local_payload(site_id)
            if payload.get("mode") == "local_mode":
                if return_success_probs:
                    raise NotImplementedError(
                        "site_observer success-probability capture requires OW site payloads; "
                        "local_mode sites are not supported."
                    )
                G = self._apply_channel_shared_site_local(
                    G,
                    payload,
                    "A",
                    expected_occupied=False,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                    choi_state=choi_state,
                    choi_context=channel_context("A"),
                    n_a=n_a,
                    p_gain=p_gain,
                    p_loss=p_loss,
                    perfect_correction=perfect_correction,
                )
                G = self._apply_channel_shared_site_local(
                    G,
                    payload,
                    "B",
                    expected_occupied=True,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                    choi_state=choi_state,
                    choi_context=channel_context("B"),
                    n_a=n_a,
                    p_gain=p_gain,
                    p_loss=p_loss,
                    perfect_correction=perfect_correction,
                )
                return G
            G, p_occ_ap = self._apply_channel_shared_site_local(
                G,
                payload,
                "Ap",
                expected_occupied=False,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("Ap"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                return_occ_prob=True,
            )
            G, p_occ_am = self._apply_channel_shared_site_local(
                G,
                payload,
                "Am",
                expected_occupied=True,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("Am"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                return_occ_prob=True,
            )
            G, p_occ_bp = self._apply_channel_shared_site_local(
                G,
                payload,
                "Bp",
                expected_occupied=False,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("Bp"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                return_occ_prob=True,
            )
            G, p_occ_bm = self._apply_channel_shared_site_local(
                G,
                payload,
                "Bm",
                expected_occupied=True,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("Bm"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                return_occ_prob=True,
            )
            if return_success_probs:
                return G, {
                    "s_Ap": (1.0 - p_occ_ap).to(torch.float64),
                    "s_Am": p_occ_am.to(torch.float64),
                    "s_Bp": (1.0 - p_occ_bp).to(torch.float64),
                    "s_Bm": p_occ_bm.to(torch.float64),
                }
            return G

        chi_Ap, chi_Bp, chi_Am, chi_Bm = self._site_spinors(site_id)
        if self._site_uses_local_mode(site_id):
            if return_success_probs:
                raise NotImplementedError(
                    "site_observer success-probability capture requires OW site payloads; "
                    "local_mode sites are not supported."
                )
            G = self._apply_channel_shared_site(
                G,
                chi_Ap,
                expected_occupied=False,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("A"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
            )
            G = self._apply_channel_shared_site(
                G,
                chi_Bp,
                expected_occupied=True,
                sample_offsets=sample_offsets,
                lyapunov_state=lyapunov_state,
                choi_state=choi_state,
                choi_context=channel_context("B"),
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
            )
            return G
        G, p_occ_ap = self._apply_channel_shared_site(
            G,
            chi_Ap,
            expected_occupied=False,
            sample_offsets=sample_offsets,
            lyapunov_state=lyapunov_state,
            choi_state=choi_state,
            choi_context=channel_context("Ap"),
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
            return_occ_prob=True,
        )
        G, p_occ_am = self._apply_channel_shared_site(
            G,
            chi_Am,
            expected_occupied=True,
            sample_offsets=sample_offsets,
            lyapunov_state=lyapunov_state,
            choi_state=choi_state,
            choi_context=channel_context("Am"),
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
            return_occ_prob=True,
        )
        G, p_occ_bp = self._apply_channel_shared_site(
            G,
            chi_Bp,
            expected_occupied=False,
            sample_offsets=sample_offsets,
            lyapunov_state=lyapunov_state,
            choi_state=choi_state,
            choi_context=channel_context("Bp"),
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
            return_occ_prob=True,
        )
        G, p_occ_bm = self._apply_channel_shared_site(
            G,
            chi_Bm,
            expected_occupied=True,
            sample_offsets=sample_offsets,
            lyapunov_state=lyapunov_state,
            choi_state=choi_state,
            choi_context=channel_context("Bm"),
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
            return_occ_prob=True,
        )
        if return_success_probs:
            return G, {
                "s_Ap": (1.0 - p_occ_ap).to(torch.float64),
                "s_Am": p_occ_am.to(torch.float64),
                "s_Bp": (1.0 - p_occ_bp).to(torch.float64),
                "s_Bm": p_occ_bm.to(torch.float64),
            }
        return G

    def _post_selection_shared_site(
        self,
        G,
        site_id,
        sample_offsets=None,
        lyapunov_state=None,
        choi_state=None,
        choi_context=None,
    ):
        if sample_offsets is None and (lyapunov_state is not None or choi_state is not None):
            sample_offsets = torch.arange(G.shape[0], dtype=torch.long, device=self.device)

        def apply_choi(chi, particle, channel, support_idx=None):
            if choi_state is None:
                return
            payload = {} if choi_context is None else dict(choi_context)
            payload["channel"] = channel
            sign = torch.full(
                (G.shape[0],),
                1.0 if particle else -1.0,
                dtype=self.real_dtype,
                device=self.device,
            )
            self._choi_apply_rank_one(
                choi_state,
                sample_offsets,
                chi,
                eta1=-sign,
                eta2=sign,
                support_idx=support_idx,
                context=payload,
            )

        if self.backend == "local":
            payload = self._site_local_payload(site_id)
            if payload.get("mode") == "local_mode":
                if lyapunov_state is not None:
                    self._lyapunov_apply_local_channel(
                        lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["A"], False
                    )
                apply_choi(payload["A"], False, "A", support_idx=payload["idx"])
                G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["A"], particle=False)
                if lyapunov_state is not None:
                    self._lyapunov_apply_local_channel(
                        lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["B"], True
                    )
                apply_choi(payload["B"], True, "B", support_idx=payload["idx"])
                G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["B"], particle=True)
                return G
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Ap"], False
                )
            apply_choi(payload["Ap"], False, "Ap", support_idx=payload["idx"])
            G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["Ap"], particle=False)
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Am"], True
                )
            apply_choi(payload["Am"], True, "Am", support_idx=payload["idx"])
            G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["Am"], particle=True)
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Bp"], False
                )
            apply_choi(payload["Bp"], False, "Bp", support_idx=payload["idx"])
            G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["Bp"], particle=False)
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, payload["idx"], payload["comp"], payload["Bm"], True
                )
            apply_choi(payload["Bm"], True, "Bm", support_idx=payload["idx"])
            G = self._measure_only_top_layer_local_batched(G, payload["idx"], payload["comp"], payload["Bm"], particle=True)
            return G

        chi_Ap, chi_Bp, chi_Am, chi_Bm = self._site_spinors(site_id)
        if self._site_uses_local_mode(site_id):
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(
                    lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Ap), False
                )
            apply_choi(chi_Ap, False, "A")
            G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Ap), particle=False, chi=chi_Ap)
            if lyapunov_state is not None:
                self._lyapunov_apply_dense_channel(
                    lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Bp), True
                )
            apply_choi(chi_Bp, True, "B")
            G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Bp), particle=True, chi=chi_Bp)
            return G
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Ap), False
            )
        apply_choi(chi_Ap, False, "Ap")
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Ap), particle=False, chi=chi_Ap)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Am), True
            )
        apply_choi(chi_Am, True, "Am")
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Am), particle=True, chi=chi_Am)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Bp), False
            )
        apply_choi(chi_Bp, False, "Bp")
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Bp), particle=False, chi=chi_Bp)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Bm), True
            )
        apply_choi(chi_Bm, True, "Bm")
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Bm), particle=True, chi=chi_Bm)
        return G

    def _post_selection_shared_site_with_success_probs(self, G, site_id, sample_offsets=None, lyapunov_state=None):
        if G.shape[0] == 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.float64)
            return G, {"s_Ap": empty, "s_Am": empty, "s_Bp": empty, "s_Bm": empty}
        if sample_offsets is None and lyapunov_state is not None:
            sample_offsets = torch.arange(G.shape[0], dtype=torch.long, device=self.device)

        if self.backend == "local":
            payload = self._site_local_payload(site_id)
            if payload.get("mode") == "local_mode":
                raise NotImplementedError(
                    "site_observer success-probability capture requires OW site payloads; "
                    "local_mode sites are not supported."
                )
            support_idx = payload["idx"]
            comp_idx = payload["comp"]

            p_occ_ap = self._occ_prob_from_local_support_batch(G, support_idx, payload["Ap"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, support_idx, comp_idx, payload["Ap"], False
                )
            G = self._measure_only_top_layer_local_batched(G, support_idx, comp_idx, payload["Ap"], particle=False)
            p_occ_am = self._occ_prob_from_local_support_batch(G, support_idx, payload["Am"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, support_idx, comp_idx, payload["Am"], True
                )
            G = self._measure_only_top_layer_local_batched(G, support_idx, comp_idx, payload["Am"], particle=True)
            p_occ_bp = self._occ_prob_from_local_support_batch(G, support_idx, payload["Bp"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, support_idx, comp_idx, payload["Bp"], False
                )
            G = self._measure_only_top_layer_local_batched(G, support_idx, comp_idx, payload["Bp"], particle=False)
            p_occ_bm = self._occ_prob_from_local_support_batch(G, support_idx, payload["Bm"])
            if lyapunov_state is not None:
                self._lyapunov_apply_local_channel(
                    lyapunov_state, G, sample_offsets, support_idx, comp_idx, payload["Bm"], True
                )
            G = self._measure_only_top_layer_local_batched(G, support_idx, comp_idx, payload["Bm"], particle=True)
            return G, {
                "s_Ap": (1.0 - p_occ_ap).to(torch.float64),
                "s_Am": p_occ_am.to(torch.float64),
                "s_Bp": (1.0 - p_occ_bp).to(torch.float64),
                "s_Bm": p_occ_bm.to(torch.float64),
            }

        chi_Ap, chi_Bp, chi_Am, chi_Bm = self._site_spinors(site_id)
        if self._site_uses_local_mode(site_id):
            raise NotImplementedError(
                "site_observer success-probability capture requires OW site payloads; "
                "local_mode sites are not supported."
            )

        p_occ_ap = self._occ_prob_from_vector_batch(G, chi_Ap)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Ap), False
            )
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Ap), particle=False, chi=chi_Ap)
        p_occ_am = self._occ_prob_from_vector_batch(G, chi_Am)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Am), True
            )
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Am), particle=True, chi=chi_Am)
        p_occ_bp = self._occ_prob_from_vector_batch(G, chi_Bp)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Bp), False
            )
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Bp), particle=False, chi=chi_Bp)
        p_occ_bm = self._occ_prob_from_vector_batch(G, chi_Bm)
        if lyapunov_state is not None:
            self._lyapunov_apply_dense_channel(
                lyapunov_state, G, sample_offsets, self._projector_from_vector(chi_Bm), True
            )
        G = self._measure_only_top_layer_batched(G, self._projector_from_vector(chi_Bm), particle=True, chi=chi_Bm)
        return G, {
            "s_Ap": (1.0 - p_occ_ap).to(torch.float64),
            "s_Am": p_occ_am.to(torch.float64),
            "s_Bp": (1.0 - p_occ_bp).to(torch.float64),
            "s_Bm": p_occ_bm.to(torch.float64),
        }

    def _apply_grouped_site_updates(
        self,
        G,
        site_ids,
        postselect=False,
        postselect_probability=0.0,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        site_observer=None,
        record_observer=None,
        outcome_overrides=None,
        lyapunov_state=None,
        choi_state=None,
        cycle=None,
        update_index=None,
        batch_index=None,
        batch_start=None,
    ):
        if G.shape[0] == 0:
            return G

        postselect_probability = 1.0 if bool(postselect) else float(postselect_probability)
        if postselect_probability < 0.0 or postselect_probability > 1.0:
            raise ValueError("postselect_probability must satisfy 0 <= p <= 1.")

        # Production uses perfect correction, no Choi matrix, and independent random
        # permutations.  In that regime grouping by the current site makes almost every
        # group a singleton and serializes the trajectory batch.  The sitewise rank-one
        # path is algebraically identical but gathers one controller vector per row and
        # keeps the full batch on the GPU.  Specialized/legacy modes retain the grouped
        # implementation below.
        if (
            postselect_probability == 0.0
            and site_observer is None
            and choi_state is None
            and not self.triv_region_local_mode
            and not (
                lyapunov_state is not None
                and lyapunov_state.get("track_record_fisher", False)
            )
        ):
            return self._apply_feedback_sitewise_batched(
                G,
                site_ids,
                n_a=n_a,
                p_gain=p_gain,
                p_loss=p_loss,
                perfect_correction=perfect_correction,
                record_observer=record_observer,
                outcome_overrides=outcome_overrides,
                lyapunov_state=lyapunov_state,
                cycle=cycle,
                update_index=update_index,
                batch_index=batch_index,
                batch_start=batch_start,
            )

        def _validate_success_payload(success_payload, expected_size, context):
            for key, value in success_payload.items():
                if value.ndim != 1 or value.shape[0] != expected_size:
                    raise RuntimeError(f"Unexpected {key} shape from {context}: {tuple(value.shape)}")
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Non-finite values encountered in {context} {key}.")

        def _emit_site_observer(site_ids_group, sample_offsets, success_payload, postselect_mask):
            if site_observer is None:
                return
            group_count = int(site_ids_group.numel())
            _validate_success_payload(success_payload, group_count, "site-observer success capture")
            site_observer(
                cycle=int(cycle),
                site_ids=site_ids_group,
                sample_offsets=sample_offsets,
                sample_indices=sample_offsets + int(batch_start),
                batch_index=int(batch_index),
                batch_start=int(batch_start),
                batch_count=group_count,
                postselect_mask=postselect_mask,
                **success_payload,
            )

        def _emit_record_observer(site_ids_group, sample_offsets, event_payload):
            if record_observer is None:
                return
            record_observer(
                cycle=int(cycle),
                update_index=int(update_index),
                site_ids=site_ids_group,
                sample_offsets=sample_offsets,
                sample_indices=sample_offsets + int(batch_start),
                batch_index=int(batch_index),
                batch_start=int(batch_start),
                batch_count=int(site_ids_group.numel()),
                **event_payload,
            )

        def _apply_one_group(G_group, site_id, site_ids_group, sample_offsets):
            group_count = int(G_group.shape[0])
            forced_group_outcomes = (
                None
                if outcome_overrides is None
                else outcome_overrides.index_select(0, sample_offsets)
            )
            choi_context = {
                "cycle": None if cycle is None else int(cycle),
                "site_id": int(site_id.item()) if torch.is_tensor(site_id) else int(site_id),
                "batch_index": None if batch_index is None else int(batch_index),
                "batch_start": 0 if batch_start is None else int(batch_start),
            }
            if postselect_probability == 1.0:
                if site_observer is None:
                    return self._post_selection_shared_site(
                        G_group,
                        site_id,
                        sample_offsets=sample_offsets,
                        lyapunov_state=lyapunov_state,
                        choi_state=choi_state,
                        choi_context=choi_context,
                    )
                G_out, success_payload = self._post_selection_shared_site_with_success_probs(
                    G_group,
                    site_id,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                )
                postselect_mask = torch.ones((group_count,), dtype=torch.bool, device=self.device)
                _emit_site_observer(site_ids_group, sample_offsets, success_payload, postselect_mask)
                return G_out

            if postselect_probability == 0.0:
                if (
                    site_observer is None
                    and record_observer is None
                    and forced_group_outcomes is None
                ):
                    return self._markov_meas_feedback_shared_site(
                        G_group,
                        site_id,
                        n_a=n_a,
                        p_gain=p_gain,
                        p_loss=p_loss,
                        perfect_correction=perfect_correction,
                        sample_offsets=sample_offsets,
                        lyapunov_state=lyapunov_state,
                        choi_state=choi_state,
                        choi_context=choi_context,
                    )
                if record_observer is not None or forced_group_outcomes is not None:
                    G_out, event_payload = self._markov_meas_feedback_shared_site(
                        G_group,
                        site_id,
                        n_a=n_a,
                        p_gain=p_gain,
                        p_loss=p_loss,
                        perfect_correction=perfect_correction,
                        return_event_data=True,
                        sample_offsets=sample_offsets,
                        lyapunov_state=lyapunov_state,
                        choi_state=choi_state,
                        choi_context=choi_context,
                        forced_outcomes=forced_group_outcomes,
                    )
                    _emit_record_observer(
                        site_ids_group, sample_offsets, event_payload
                    )
                    if site_observer is not None:
                        success_payload = {
                            f"s_{label}": event_payload[
                                "target_success_probability"
                            ][:, channel_index]
                            for channel_index, label in enumerate(
                                event_payload["channel_labels"]
                            )
                        }
                        postselect_mask = torch.zeros(
                            (group_count,), dtype=torch.bool, device=self.device
                        )
                        _emit_site_observer(
                            site_ids_group,
                            sample_offsets,
                            success_payload,
                            postselect_mask,
                        )
                    return G_out
                G_out, success_payload = self._markov_meas_feedback_shared_site(
                    G_group,
                    site_id,
                    n_a=n_a,
                    p_gain=p_gain,
                    p_loss=p_loss,
                    perfect_correction=perfect_correction,
                    return_success_probs=True,
                    sample_offsets=sample_offsets,
                    lyapunov_state=lyapunov_state,
                )
                postselect_mask = torch.zeros((group_count,), dtype=torch.bool, device=self.device)
                _emit_site_observer(site_ids_group, sample_offsets, success_payload, postselect_mask)
                return G_out

            postselect_mask = torch.rand((group_count,), device=self.device, dtype=self.real_dtype) < postselect_probability
            forced_idx = torch.nonzero(postselect_mask, as_tuple=False).flatten()
            born_idx = torch.nonzero(~postselect_mask, as_tuple=False).flatten()
            G_out = G_group.clone()
            combined_payload = None
            if site_observer is not None:
                combined_payload = {
                    key: torch.empty((group_count,), dtype=torch.float64, device=self.device)
                    for key in ("s_Ap", "s_Am", "s_Bp", "s_Bm")
                }

            if forced_idx.numel() > 0:
                G_forced = G_group.index_select(0, forced_idx)
                forced_offsets = sample_offsets.index_select(0, forced_idx)
                if site_observer is None:
                    G_forced = self._post_selection_shared_site(
                        G_forced,
                        site_id,
                        sample_offsets=forced_offsets,
                        lyapunov_state=lyapunov_state,
                        choi_state=choi_state,
                        choi_context=choi_context,
                    )
                else:
                    G_forced, forced_payload = self._post_selection_shared_site_with_success_probs(
                        G_forced,
                        site_id,
                        sample_offsets=forced_offsets,
                        lyapunov_state=lyapunov_state,
                    )
                    for key, value in forced_payload.items():
                        combined_payload[key][forced_idx] = value
                G_out[forced_idx] = G_forced

            if born_idx.numel() > 0:
                G_born = G_group.index_select(0, born_idx)
                born_offsets = sample_offsets.index_select(0, born_idx)
                if site_observer is None:
                    G_born = self._markov_meas_feedback_shared_site(
                        G_born,
                        site_id,
                        n_a=n_a,
                        p_gain=p_gain,
                        p_loss=p_loss,
                        perfect_correction=perfect_correction,
                        sample_offsets=born_offsets,
                        lyapunov_state=lyapunov_state,
                        choi_state=choi_state,
                        choi_context=choi_context,
                    )
                else:
                    G_born, born_payload = self._markov_meas_feedback_shared_site(
                        G_born,
                        site_id,
                        n_a=n_a,
                        p_gain=p_gain,
                        p_loss=p_loss,
                        perfect_correction=perfect_correction,
                        return_success_probs=True,
                        sample_offsets=born_offsets,
                        lyapunov_state=lyapunov_state,
                    )
                    for key, value in born_payload.items():
                        combined_payload[key][born_idx] = value
                G_out[born_idx] = G_born

            if site_observer is not None:
                _emit_site_observer(site_ids_group, sample_offsets, combined_payload, postselect_mask)
            return G_out

        if bool(torch.all(site_ids == site_ids[0])):
            site_id = site_ids[0]
            sample_offsets = torch.arange(G.shape[0], dtype=torch.long, device=self.device)
            return _apply_one_group(G, site_id, site_ids, sample_offsets)

        perm = torch.argsort(site_ids)
        G_sorted = G.index_select(0, perm)
        site_sorted = site_ids.index_select(0, perm)
        change_points = torch.nonzero(site_sorted[1:] != site_sorted[:-1], as_tuple=False).flatten() + 1
        boundaries = torch.cat(
            (
                torch.zeros((1,), dtype=torch.long, device=self.device),
                change_points,
                torch.full((1,), G.shape[0], dtype=torch.long, device=self.device),
            )
        )

        for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            site_id = site_sorted[start]
            G_sub = G_sorted[start:end]
            sample_offsets = perm[start:end]
            G_sub = _apply_one_group(G_sub, site_id, site_sorted[start:end], sample_offsets)
            G_sorted[start:end] = G_sub

        invperm = torch.empty_like(perm)
        invperm[perm] = torch.arange(perm.numel(), device=self.device)
        return G_sorted.index_select(0, invperm)

    def markov_meas_feedback_batch_sites(
        self, G, Rx, Ry, n_a=0.5, p_gain=None, p_loss=None, perfect_correction=False
    ):
        site_ids = Rx.to(dtype=torch.long) + self.Nx * Ry.to(dtype=torch.long)
        return self._apply_grouped_site_updates(
            G,
            site_ids,
            postselect=False,
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
        )

    def post_selection_markov_top_layer_batch_sites(self, G, Rx, Ry):
        site_ids = Rx.to(dtype=torch.long) + self.Nx * Ry.to(dtype=torch.long)
        return self._apply_grouped_site_updates(G, site_ids, postselect=True)

    def markov_meas_feedback_batched(
        self, G, Rx, Ry, n_a=0.5, p_gain=None, p_loss=None, perfect_correction=False
    ):
        site_ids = torch.full((G.shape[0],), int(Rx) + self.Nx * int(Ry), dtype=torch.long, device=self.device)
        return self._apply_grouped_site_updates(
            G,
            site_ids,
            postselect=False,
            n_a=n_a,
            p_gain=p_gain,
            p_loss=p_loss,
            perfect_correction=perfect_correction,
        )

    def post_selection_markov_top_layer_batched(self, G, Rx, Ry):
        site_ids = torch.full((G.shape[0],), int(Rx) + self.Nx * int(Ry), dtype=torch.long, device=self.device)
        return self._apply_grouped_site_updates(G, site_ids, postselect=True)

    def _prepare_schedule_state(self, mode, coords_for_len):
        site_ids = [int(Rx) + self.Nx * int(Ry) for Rx, Ry in coords_for_len]
        state = {
            "mode": mode,
            "base_site_ids": torch.as_tensor(site_ids, dtype=torch.long, device=self.device),
        }

        if mode == "dw_symmetric_random":
            dw_sorted = sorted(list(set(self.DW_loc)))
            x_start = int(dw_sorted[-1])
            x_order = list(range(x_start, -1, -1)) + list(range(self.Nx - 1, x_start, -1))
            allowed_ry_by_x = {}
            for Rx, Ry in coords_for_len:
                allowed_ry_by_x.setdefault(int(Rx), []).append(int(Ry))
            x_order_filtered = [int(Rx) for Rx in x_order if Rx in allowed_ry_by_x]
            state["x_order_filtered"] = x_order_filtered
            state["allowed_ry_by_x"] = {
                Rx: torch.as_tensor(vals, dtype=torch.long, device=self.device)
                for Rx, vals in allowed_ry_by_x.items()
            }
        return state

    def _sequence_schedule_batch(self, schedule_state, batch_count):
        mode = schedule_state["mode"]
        base_site_ids = schedule_state["base_site_ids"]
        total_sites = int(base_site_ids.numel())

        if mode in ("raster_y", "raster_x"):
            return base_site_ids.unsqueeze(0).expand(batch_count, -1)

        if mode == "random":
            perm_keys = torch.rand((batch_count, total_sites), device=self.device, dtype=self.real_dtype)
            perm = torch.argsort(perm_keys, dim=1)
            return base_site_ids.unsqueeze(0).expand(batch_count, -1).gather(1, perm)

        if mode == "dw_symmetric_random":
            schedules = torch.empty((batch_count, total_sites), device=self.device, dtype=torch.long)
            pos = 0
            for Rx in schedule_state["x_order_filtered"]:
                y_vals = schedule_state["allowed_ry_by_x"][Rx]
                count = int(y_vals.numel())
                perm_keys = torch.rand((batch_count, count), device=self.device, dtype=self.real_dtype)
                perm = torch.argsort(perm_keys, dim=1)
                y_order = y_vals.unsqueeze(0).expand(batch_count, -1).gather(1, perm)
                schedules[:, pos:pos + count] = int(Rx) + self.Nx * y_order
                pos += count
            return schedules

        raise ValueError(f"Unsupported sequence mode: {mode}")

    def _cache_key(
        self,
        *,
        cycles,
        samples,
        init_mode,
        n_a,
        seq,
        store_mode,
        snapshot_cycles,
        ps,
        psp,
        pc,
        dtype_name,
        backend_name,
        mslab,
        state_representation,
        frame_init_prepared=False,
        G_init_prepared=False,
        feedback_suffix="",
    ):
        nsh = "None" if self.nshell is None else str(self.nshell)
        snapshot_tag = "" if snapshot_cycles is None else "_snap" + "-".join(str(cyc) for cyc in snapshot_cycles)
        prepared_tag = (
            "_frameprepared1"
            if frame_init_prepared
            else "_gprepared1"
            if G_init_prepared
            else ""
        )
        key = (
            f"N{self.Nx}x{self.Ny}"
            f"_C{int(cycles)}"
            f"_S{int(samples)}"
            f"_nsh{nsh}"
            f"_DW{int(bool(self.DW))}"
            f"_alpha_top{self.alpha_top}"
            f"_alpha_triv{self.alpha_triv}"
            f"_trial-{self.trial_orbitals}"
            f"_dwtrunc{int(bool(self.dw_truncation))}"
            f"_init-{init_mode}"
            f"_n_a{n_a}"
            f"{feedback_suffix}"
            f"_seq-{seq}"
            f"{snapshot_tag}"
            f"_ps{int(ps)}"
            f"_psp{float(psp):g}"
            f"_pc{int(pc)}"
            f"_mslab{int(bool(mslab))}"
            f"_dtype-{dtype_name}"
            f"_backend-{backend_name}"
            f"_repr-{state_representation}"
            f"{prepared_tag}"
            "_engine-frame-v2"
            "_markov_circuit_gpu"
        )
        if store_mode == "final":
            key += "_final"
        elif store_mode == "snapshots":
            key += "_snapshots"
        return key

    def _run_mean_replacement_circuit(
        self,
        *,
        cycles,
        samples,
        init_mode,
        G_init,
        sequence,
        meas_slab_only,
        batch_size,
        frozen_schedule,
        cycle_observer,
        native_cycle_observer,
        lyapunov_frame_observer,
        lyapunov_nvec,
        lyapunov_start_cycle,
        lyapunov_track_restricted_core,
        lyapunov_singular_tol,
        lyapunov_failure_mode,
        return_data,
    ):
        """Exact fixed-schedule mean replacement, called only through run_markov_circuit."""
        cycles, samples = int(cycles), int(samples)
        if cycles <= 0 or samples <= 0:
            raise ValueError("mean replacement requires positive cycles and samples")
        batch_size = samples if batch_size is None else int(batch_size)
        if batch_size != samples:
            raise ValueError("mean replacement currently requires one complete shard batch")
        seq_info = self._sequence_helper(sequence, meas_slab_only=bool(meas_slab_only))
        coords = seq_info["coords_for_len"]
        sites_per_cycle = len(coords)
        if frozen_schedule is None:
            schedule_state = self._prepare_schedule_state(seq_info["mode"], coords)
            schedules = torch.stack(
                [self._sequence_schedule_batch(schedule_state, samples) for _ in range(cycles)], dim=1
            )
        else:
            schedules = torch.as_tensor(frozen_schedule, dtype=torch.long, device=self.device)
            expected = (samples, cycles, sites_per_cycle)
            if tuple(schedules.shape) != expected:
                raise ValueError(f"mean frozen_schedule must have shape {expected}")
        active_indices = self.active_top_layer_indices(meas_slab_only=bool(meas_slab_only))
        lyapunov_state = None
        if lyapunov_frame_observer is not None:
            nvec = 16 if lyapunov_nvec is None else int(lyapunov_nvec)
            lyapunov_state = self._init_lyapunov_state(
                batch_count=samples,
                n_vec=nvec,
                basis_idx=active_indices if self._meas_slab_only_effective(meas_slab_only) else None,
                initial_frame=None,
                sample_start=0,
                track_restricted_core=bool(lyapunov_track_restricted_core),
                track_record_fisher=False,
                singular_tol=float(lyapunov_singular_tol),
                failure_mode=str(lyapunov_failure_mode),
            )

        def replace(batch, chi, eta):
            chi = chi / torch.linalg.vector_norm(chi).clamp_min(1e-15)
            gchi = torch.matmul(batch, chi[:, None]).squeeze(-1)
            chig = torch.matmul(chi.conj()[None, None, :], batch).squeeze(1)
            scalar = torch.sum(chi.conj()[None, :] * gchi, dim=1)
            projector = chi[:, None] * chi.conj()[None, :]
            return (
                batch
                - chi[None, :, None] * chig[:, None, :]
                - gchi[:, :, None] * chi.conj()[None, None, :]
                + (scalar + float(eta))[:, None, None] * projector[None]
            )

        with torch.inference_mode():
            G = self._prepare_initial_batch(samples, init_mode, G_init=G_init, sample_offset=0)
            if self._meas_slab_only_effective(meas_slab_only):
                G = self._prepare_exterior_product_state_batched(
                    G, mode="born_conditioned"
                )
            if cycle_observer is not None:
                cycle_observer(cycle=0, G=G, batch_index=0, batch_start=0, batch_count=samples)
            if native_cycle_observer is not None:
                native_cycle_observer(
                    cycle=0, state=G, batch_index=0, batch_start=0, batch_count=samples
                )
            for cycle in range(1, cycles + 1):
                for update_index in range(sites_per_cycle):
                    site_word = schedules[:, cycle - 1, update_index]
                    for site_id_tensor in torch.unique(site_word, sorted=True):
                        site_id = int(site_id_tensor.item())
                        selected = torch.nonzero(site_word == site_id, as_tuple=False).flatten()
                        batch = G.index_select(0, selected)
                        chi_ap, chi_bp, chi_am, chi_bm = self._site_spinors(site_id)
                        if self._site_uses_local_mode(site_id):
                            specs = ((chi_ap, -1.0), (chi_bp, 1.0))
                        else:
                            specs = (
                                (chi_ap, -1.0),
                                (chi_am, 1.0),
                                (chi_bp, -1.0),
                                (chi_bm, 1.0),
                            )
                        for chi, eta in specs:
                            batch = replace(batch, chi, eta)
                            if lyapunov_state is not None and cycle >= int(lyapunov_start_cycle):
                                frame_chi = (
                                    chi.index_select(0, active_indices)
                                    if int(lyapunov_state["frame"].shape[1]) != self.Nlayer
                                    else chi
                                )
                                P = frame_chi[:, None] * frame_chi.conj()[None, :]
                                self._lyapunov_apply_dense_reset(lyapunov_state, selected, P)
                        G[selected] = 0.5 * (batch + batch.conj().transpose(-2, -1))
                if lyapunov_state is not None and cycle >= int(lyapunov_start_cycle):
                    spectrum = self._lyapunov_end_cycle(
                        lyapunov_state, cycle - int(lyapunov_start_cycle) + 1
                    )
                    lyapunov_frame_observer(
                        cycle=cycle,
                        lyapunov_cycle=cycle - int(lyapunov_start_cycle) + 1,
                        spectra=spectrum,
                        batch_index=0,
                        batch_start=0,
                        batch_count=samples,
                        **self._lyapunov_frame_payload(lyapunov_state),
                    )
                if cycle_observer is not None:
                    cycle_observer(cycle=cycle, G=G, batch_index=0, batch_start=0, batch_count=samples)
                if native_cycle_observer is not None:
                    native_cycle_observer(
                        cycle=cycle,
                        state=G,
                        batch_index=0,
                        batch_start=0,
                        batch_count=samples,
                    )
        metadata = {
            "mean_replacement": True,
            "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
            "samples": samples,
            "cycles": cycles,
            "schedule_signature": self._array_signature(frozen_schedule),
        }
        if return_data:
            metadata["G_final"] = G.detach().cpu().numpy()
        return metadata

    def mean_replacement_channels(self, *, meas_slab_only=False, site_ids=None):
        """Return the canonical ordered local channels used by the exact mean map.

        This public, read-only description supports generic mean-channel validation and
        matrix-free generator analysis.  The returned spinors are copies;
        mutating them cannot alter the controller frame held by the dynamics object.
        """
        sequence = self._sequence_helper("random", meas_slab_only=bool(meas_slab_only))
        allowed = [int(x) + self.Nx * int(y) for x, y in sequence["coords_for_len"]]
        selected_sites = allowed if site_ids is None else [int(value) for value in site_ids]
        unknown = sorted(set(selected_sites).difference(allowed))
        if unknown:
            raise ValueError(f"site_ids contain centers outside the active schedule: {unknown}")
        channels = []
        for site_id in selected_sites:
            chi_ap, chi_bp, chi_am, chi_bm = self._site_spinors(site_id)
            if self._site_uses_local_mode(site_id):
                specs = (("A", chi_ap, 0), ("B", chi_bp, 1))
            else:
                specs = (
                    ("Ap", chi_ap, 0),
                    ("Am", chi_am, 1),
                    ("Bp", chi_bp, 0),
                    ("Bm", chi_bm, 1),
                )
            for channel_index, (label, chi, target) in enumerate(specs):
                normalized = chi / torch.linalg.vector_norm(chi).clamp_min(1e-15)
                channels.append(
                    {
                        "site_id": int(site_id),
                        "channel_index": int(channel_index),
                        "label": label,
                        "target_occupation": int(target),
                        "eta": float(2 * int(target) - 1),
                        "chi": normalized.detach().clone(),
                    }
                )
        return channels

    def _prepare_initial_frame_batch(
        self,
        batch_size,
        *,
        init_mode,
        G_init=None,
        frame_init=None,
        frame_ranks=None,
        sample_offset=0,
        purity_tolerance=1e-9,
    ):
        if frame_init is not None:
            tensor = torch.as_tensor(
                frame_init, dtype=self.dtype, device=self.device
            )
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1).clone()
                ranks = frame_ranks
            elif tensor.ndim == 3:
                tensor = tensor[sample_offset : sample_offset + batch_size]
                ranks = (
                    None
                    if frame_ranks is None
                    else torch.as_tensor(
                        frame_ranks, dtype=torch.long, device=self.device
                    )[sample_offset : sample_offset + batch_size]
                )
            else:
                raise ValueError(
                    "frame_init must have shape (N,k) or (samples,N,Rcap)."
                )
            return BatchedOccupiedFrameState.from_frame(
                tensor,
                ranks=ranks,
                device=self.device,
                dtype=self.dtype,
            )
        if G_init is not None:
            centered = self._prepare_initial_batch(
                batch_size, init_mode, G_init=G_init, sample_offset=sample_offset
            )
            return BatchedOccupiedFrameState.from_centered_covariance(
                centered, purity_tolerance=purity_tolerance
            )
        rank = int(round(self.filling_frac * self.Nlayer))
        return BatchedOccupiedFrameState.random_pure(
            batch_size,
            self.Nlayer,
            rank,
            device=self.device,
            dtype=self.dtype,
        )

    def _prepare_exterior_product_frame_batched(
        self,
        state,
        *,
        mode="born_conditioned",
        outcome_observer=None,
        outcome_overrides=None,
        batch_start=0,
    ):
        if not isinstance(state, BatchedOccupiedFrameState):
            raise TypeError("state must be a BatchedOccupiedFrameState.")
        mode = str(mode).strip().lower()
        if mode not in ("born_conditioned", "forced_occupied"):
            raise ValueError(
                "exterior preparation mode must be 'born_conditioned' or "
                "'forced_occupied'."
            )
        exterior = []
        if self.DW and hasattr(self, "DW_loc"):
            x_min, x_max = sorted(int(x) for x in self.DW_loc)
            exterior = [
                (x, y)
                for x in range(self.Nx)
                for y in range(self.Ny)
                if not (x_min <= x <= x_max)
            ]
        overrides = (
            None
            if outcome_overrides is None
            else torch.as_tensor(
                outcome_overrides, dtype=torch.bool, device=self.device
            )
        )
        outcome_rows = []
        orbital_indices = []
        event_index = 0
        for x, y in exterior:
            for orbital_index in (0, 1):
                index = orbital_index + 2 * x + 2 * self.Nx * y
                orbital = torch.zeros(
                    (state.batch_size, self.Nlayer),
                    dtype=self.dtype,
                    device=self.device,
                )
                orbital[:, index] = 1.0
                probability = state.occupation_probability(orbital)
                occupied = (
                    torch.ones_like(probability, dtype=torch.bool)
                    if mode == "forced_occupied"
                    else overrides[:, event_index]
                    if overrides is not None
                    else torch.rand_like(probability) < probability
                )
                state.project_occupied(orbital, occupied)
                state.project_empty(orbital, ~occupied)
                outcome_rows.append(occupied)
                orbital_indices.append(int(index))
                event_index += 1
        if outcome_observer is not None and outcome_rows:
            outcome_observer(
                orbital_indices=torch.as_tensor(
                    orbital_indices, dtype=torch.long, device=self.device
                ),
                outcome_occupied=torch.stack(outcome_rows, dim=1),
                sample_indices=torch.arange(
                    int(batch_start), int(batch_start) + state.batch_size,
                    dtype=torch.long, device=self.device
                ),
                batch_start=int(batch_start),
                batch_count=int(state.batch_size),
            )
        return state

    def _apply_frame_site_updates(
        self,
        state,
        site_ids,
        *,
        postselect_probability=0.0,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        perfect_correction=False,
        site_observer=None,
        record_observer=None,
        outcome_overrides=None,
        lyapunov_state=None,
        choi_state=None,
        cycle=None,
        update_index=None,
        batch_index=None,
        batch_start=None,
    ):
        """Apply one ordered OW site word to a padded occupied-frame batch."""

        if not isinstance(state, BatchedOccupiedFrameState):
            raise TypeError("state must be a BatchedOccupiedFrameState.")
        site_ids = torch.as_tensor(
            site_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if site_ids.numel() != state.batch_size:
            raise ValueError("site_ids must contain one site per trajectory.")
        _, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        forced_postselect = (
            torch.rand(
                state.batch_size, dtype=self.real_dtype, device=self.device
            )
            < float(postselect_probability)
        )
        overrides = None
        if outcome_overrides is not None:
            overrides = torch.as_tensor(
                outcome_overrides, dtype=torch.bool, device=self.device
            )

        local_mask = self._site_local_mode_mask.index_select(0, site_ids)

        def run_family(family_mask, specs, orbitals):
            if not torch.any(family_mask):
                return
            probability_rows = []
            outcome_rows = []
            target_rows = []
            reset_rows = []
            success_rows = []
            for channel_index, (label, expected_occupied) in enumerate(specs):
                orbital = orbitals[label]
                p_occ = state.occupation_probability(orbital)
                born_mask = family_mask & ~forced_postselect
                if overrides is None:
                    outcome = torch.rand_like(p_occ) < p_occ
                else:
                    if channel_index >= overrides.shape[1]:
                        raise ValueError("frozen_outcomes has too few channel columns.")
                    outcome = overrides[:, channel_index]
                outcome = torch.where(
                    forced_postselect,
                    torch.full_like(outcome, bool(expected_occupied)),
                    outcome,
                )
                self._lyapunov_apply_rank_one_sitewise(
                    lyapunov_state, state, orbital, outcome
                )
                mismatch = born_mask & (outcome != bool(expected_occupied))
                if perfect_correction:
                    correction_succeeded = mismatch
                    correction_log_probability = torch.zeros_like(p_occ)
                else:
                    probability = p_gain_eff if expected_occupied else p_loss_eff
                    correction_succeeded = mismatch & (
                        torch.rand_like(p_occ) < float(probability)
                    )
                    selected_correction_probability = torch.where(
                        correction_succeeded,
                        torch.full_like(p_occ, float(probability)),
                        torch.full_like(p_occ, 1.0 - float(probability)),
                    )
                    correction_log_probability = torch.where(
                        mismatch,
                        torch.log(
                            selected_correction_probability.clamp_min(
                                torch.finfo(self.real_dtype).tiny
                            )
                        ),
                        torch.zeros_like(p_occ),
                    )
                final_target = torch.where(
                    correction_succeeded,
                    torch.full_like(outcome, bool(expected_occupied)),
                    outcome,
                )
                self._lyapunov_reset_rank_one_sitewise(
                    lyapunov_state, orbital, mismatch
                )
                changed = family_mask & (final_target != outcome)
                state.gain(orbital, changed & final_target)
                state.loss(orbital, changed & ~final_target)
                projected = family_mask & ~changed
                state.project_occupied(orbital, projected & outcome)
                state.project_empty(orbital, projected & ~outcome)
                realized_measurement_probability = torch.where(
                    outcome, p_occ, 1.0 - p_occ
                )
                state.log_weight += torch.where(
                    born_mask,
                    torch.log(
                        realized_measurement_probability.clamp_min(
                            torch.finfo(self.real_dtype).tiny
                        )
                    )
                    + correction_log_probability,
                    torch.zeros_like(p_occ),
                )
                if choi_state is not None:
                    sample_offsets = torch.arange(
                        state.batch_size, dtype=torch.long, device=self.device
                    )
                    self._choi_apply_rank_one(
                        choi_state,
                        sample_offsets,
                        orbital,
                        eta1=-torch.where(
                            outcome,
                            torch.ones_like(p_occ),
                            -torch.ones_like(p_occ),
                        ),
                        eta2=torch.where(
                            final_target,
                            torch.ones_like(p_occ),
                            -torch.ones_like(p_occ),
                        ),
                        context={
                            "cycle": None if cycle is None else int(cycle),
                            "channel": str(label),
                            "batch_index": None if batch_index is None else int(batch_index),
                            "batch_start": None if batch_start is None else int(batch_start),
                        },
                    )

                probability_rows.append(p_occ.to(torch.float64))
                success_rows.append(
                    (p_occ if expected_occupied else 1.0 - p_occ).to(torch.float64)
                )
                outcome_rows.append(outcome)
                target_rows.append(
                    torch.full_like(outcome, bool(expected_occupied))
                )
                reset_rows.append(
                    torch.where(
                        final_target,
                        torch.ones_like(p_occ),
                        -torch.ones_like(p_occ),
                    ).to(torch.float64)
                )

            if record_observer is not None:
                rows = torch.nonzero(family_mask, as_tuple=False).flatten()
                probabilities = torch.stack(probability_rows, dim=1).index_select(0, rows)
                outcomes = torch.stack(outcome_rows, dim=1).index_select(0, rows)
                targets = torch.stack(target_rows, dim=1).index_select(0, rows)
                realized = torch.where(
                    outcomes, probabilities, 1.0 - probabilities
                ).clamp(0.0, 1.0)
                tiny = torch.finfo(torch.float64).tiny
                record_observer(
                    cycle=int(cycle),
                    update_index=int(update_index),
                    site_ids=site_ids.index_select(0, rows),
                    sample_offsets=rows,
                    sample_indices=rows + int(batch_start),
                    batch_index=int(batch_index),
                    batch_start=int(batch_start),
                    batch_count=int(rows.numel()),
                    channel_labels=tuple(label for label, _ in specs),
                    occupation_probability=probabilities,
                    outcome_occupied=outcomes,
                    target_occupied=targets,
                    transfer=targets.to(torch.int8) - outcomes.to(torch.int8),
                    realized_probability=realized,
                    conditional_log_probability=torch.log(
                        realized.clamp_min(tiny)
                    ),
                    target_success_probability=torch.stack(
                        success_rows, dim=1
                    ).index_select(0, rows),
                    reset_covariance=torch.stack(reset_rows, dim=1).index_select(
                        0, rows
                    ),
                )
            if site_observer is not None:
                rows = torch.nonzero(family_mask, as_tuple=False).flatten()
                site_observer(
                    cycle=int(cycle),
                    site_ids=site_ids.index_select(0, rows),
                    sample_offsets=rows,
                    sample_indices=rows + int(batch_start),
                    batch_index=int(batch_index),
                    batch_start=int(batch_start),
                    batch_count=int(rows.numel()),
                    postselect_mask=forced_postselect.index_select(0, rows),
                    **{
                        f"s_{label}": success_rows[index].index_select(0, rows)
                        for index, (label, _) in enumerate(specs)
                    },
                )

        if torch.any(~local_mask):
            ow_orbitals = {
                "Ap": self.WF_Ap_sites.index_select(0, site_ids),
                "Am": self.WF_Am_sites.index_select(0, site_ids),
                "Bp": self.WF_Bp_sites.index_select(0, site_ids),
                "Bm": self.WF_Bm_sites.index_select(0, site_ids),
            }
            run_family(
                ~local_mask,
                (("Ap", False), ("Am", True), ("Bp", False), ("Bm", True)),
                ow_orbitals,
            )
        if torch.any(local_mask):
            orbital_a = torch.zeros(
                (state.batch_size, self.Nlayer), dtype=self.dtype, device=self.device
            )
            orbital_b = torch.zeros_like(orbital_a)
            x = site_ids % self.Nx
            y = site_ids // self.Nx
            orbital_a[
                torch.arange(state.batch_size, device=self.device),
                2 * x + 2 * self.Nx * y,
            ] = 1.0
            orbital_b[
                torch.arange(state.batch_size, device=self.device),
                1 + 2 * x + 2 * self.Nx * y,
            ] = 1.0
            run_family(local_mask, (("A", False), ("B", True)), {"A": orbital_a, "B": orbital_b})
        return state

    def run_markov_circuit(
        self,
        G_history=True,
        progress=True,
        cycles=None,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=False,
        samples=None,
        init_mode="default",
        G_init=None,
        frame_init=None,
        frame_ranks=None,
        initial_purity_tolerance=1e-9,
        save=True,
        save_init=True,
        save_history_stride=None,
        snapshot_cycles=None,
        save_suffix=None,
        n_a=0.5,
        p_gain=None,
        p_loss=None,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=None,
        save_dir=None,
        resume=False,
        merge_on_finish=False,
        return_data=True,
        run_dir_style="cache_key",
        state_representation="auto",
        cycle_observer=None,
        native_cycle_observer=None,
        site_observer=None,
        record_observer=None,
        noise_observer=None,
        onsite_phase_noise_sigma=0.0,
        frozen_schedule=None,
        frozen_outcomes=None,
        lyapunov_observer=None,
        lyapunov_nvec=None,
        lyapunov_frame_observer=None,
        lyapunov_initial_frame=None,
        lyapunov_start_cycle=1,
        lyapunov_track_restricted_core=False,
        lyapunov_track_record_fisher=False,
        lyapunov_singular_tol=1e-12,
        lyapunov_failure_mode="raise",
        track_choi=False,
        choi_observer=None,
        choi_observer_cycles=None,
        choi_singular_tol=1e-10,
        choi_failure_mode="raise",
        skip_batch_indices=None,
        mean_replacement=False,
        return_native_state=False,
        require_no_covariance_materialization=False,
        frame_reorthonormalize_interval=1,
        cycle_observer_cycles=None,
        native_cycle_observer_cycles=None,
        exterior_outcome_observer=None,
        frozen_exterior_outcomes=None,
        lyapunov_observer_cycles=None,
        lyapunov_basis_mode="canonical",
        lyapunov_accumulator_reset_cycles=None,
        frame_init_prepared=False,
        G_init_prepared=False,
        covariance_spectral_clip=False,
        covariance_clip_sample_chunk=10,
        covariance_clip_max_correction=1e-6,
        covariance_clip_observer=None,
    ):
        """Execute batched adaptive dynamics through the canonical GPU runner.

        With ``meas_slab_only=True``, ``DW=True``, and ``dw_truncation=True``,
        the exterior is projected sequentially in the canonical unit-cell/orbital
        basis before the cycle-zero observer, and OW updates subsequently visit
        only slab cells. Ordinary and partial postselection runs use Born outcomes
        for this preparation; full forced postselection selects occupied orbitals.
        If ``frame_init_prepared=True``, ``frame_init`` is already the endpoint
        of that hard-wall exterior preparation and the preparation is not
        repeated.  This supports exact continuation from a saved native frame.
        Likewise, ``G_init_prepared=True`` marks an explicit covariance
        ``G_init`` as already exterior-prepared and supports exact continuation
        of mixed covariance trajectories without consuming the exterior Born
        draws again.  Both flags default to false and preserve the historical
        initialization behavior.
        ``lyapunov_accumulator_reset_cycles`` resets QR logs and stabilized cores
        after observers at those cycles while preserving the aligned tangent frame.

        ``covariance_spectral_clip=True`` projects occupations to [0,1] after
        each complete physical cycle, before snapshots/observers/continuation.
        This changes the numerical trajectory and requires a separately versioned
        caller output. It supports explicit covariance, save=False acquisition
        only; tangent/Choi/mean-replacement paths cannot silently use this map.
        ``covariance_clip_observer`` receives per-sample correction diagnostics.
        """
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
        frame_init_prepared = bool(frame_init_prepared)
        G_init_prepared = bool(G_init_prepared)
        if frame_init_prepared and frame_init is None:
            raise ValueError("frame_init_prepared=True requires frame_init.")
        if G_init_prepared and G_init is None:
            raise ValueError("G_init_prepared=True requires G_init.")
        if frame_init_prepared and G_init_prepared:
            raise ValueError(
                "frame_init_prepared and G_init_prepared are mutually exclusive."
            )
        init_mode = str(init_mode).strip().lower()
        if init_mode not in ("default", "maxmix"):
            raise ValueError("init_mode must be 'default' or 'maxmix'.")
        lyapunov_basis_mode = str(lyapunov_basis_mode).strip().lower()
        if lyapunov_basis_mode not in ("canonical", "pure_occupied_empty"):
            raise ValueError(
                "lyapunov_basis_mode must be 'canonical' or "
                "'pure_occupied_empty'."
            )
        reset_cycles = tuple(
            sorted({int(value) for value in (lyapunov_accumulator_reset_cycles or ())})
        )
        if any(value < int(lyapunov_start_cycle) or value >= int(cycles) for value in reset_cycles):
            raise ValueError(
                "lyapunov_accumulator_reset_cycles must lie between "
                "lyapunov_start_cycle and cycles-1"
            )
        if bool(mean_replacement) and reset_cycles:
            raise ValueError(
                "lyapunov_accumulator_reset_cycles are not supported by mean replacement"
            )
        initial_purity_defects = None
        if bool(mean_replacement):
            detected_initial_pure = False
            resolution_reason = "mean_replacement_is_mixed"
        elif frame_init is not None:
            detected_initial_pure = True
            resolution_reason = "explicit_frame_init"
        elif G_init is not None:
            initial_tensor = torch.as_tensor(
                G_init, dtype=self.dtype, device=self.device
            )
            if initial_tensor.shape[-2:] == (self.Ntot, self.Ntot):
                initial_tensor = initial_tensor[..., : self.Nlayer, : self.Nlayer]
            initial_purity_defects_tensor = self._centered_purity_defect(initial_tensor)
            initial_purity_defects = (
                initial_purity_defects_tensor.detach().cpu().tolist()
            )
            pure_mask = initial_purity_defects_tensor <= initial_purity_tolerance
            if bool(torch.any(pure_mask)) and not bool(torch.all(pure_mask)):
                raise ValueError(
                    "G_init contains a heterogeneous pure/mixed batch; split it "
                    "into separate run_markov_circuit calls."
                )
            detected_initial_pure = bool(torch.all(pure_mask))
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
            state_representation_resolved = (
                "physical_frame" if detected_initial_pure else "covariance"
            )
        else:
            state_representation_resolved = state_representation_requested
        if state_representation_resolved == "physical_frame" and not detected_initial_pure:
            raise ValueError(
                "state_representation='physical_frame' requires a pure initial state."
            )
        explicit_covariance_override = bool(
            state_representation_requested == "covariance" and detected_initial_pure
        )
        frame_native = state_representation_resolved == "physical_frame"
        if covariance_spectral_clip:
            if frame_native or mean_replacement or track_choi or lyapunov_observer is not None or lyapunov_frame_observer is not None:
                raise ValueError("covariance spectral clipping does not support frame/mean/tangent/Choi dynamics")
            if save or state_representation_requested != "covariance":
                raise ValueError("covariance spectral clipping requires explicit covariance and save=False; caller owns versioned outputs")
            if int(covariance_clip_sample_chunk) <= 0 or not np.isfinite(covariance_clip_max_correction) or covariance_clip_max_correction <= 0:
                raise ValueError("invalid covariance clipping chunk/correction limit")
        elif covariance_clip_observer is not None:
            raise ValueError("covariance_clip_observer requires covariance_spectral_clip=True")
        if G_init_prepared and frame_native:
            raise ValueError(
                "G_init_prepared=True requires covariance state representation."
            )
        initial_state_prepared = bool(frame_init_prepared or G_init_prepared)
        if lyapunov_basis_mode == "pure_occupied_empty" and not frame_native:
            raise ValueError(
                "lyapunov_basis_mode='pure_occupied_empty' requires a pure "
                "physical-frame trajectory."
            )
        require_no_covariance_materialization = bool(
            require_no_covariance_materialization
        )
        return_native_state = bool(return_native_state)
        frame_reorthonormalize_interval = int(frame_reorthonormalize_interval)
        if frame_reorthonormalize_interval <= 0:
            raise ValueError("frame_reorthonormalize_interval must be positive.")
        covariance_materializations = []
        if return_native_state and not frame_native:
            raise ValueError(
                "return_native_state=True requires a physical-frame trajectory."
            )
        if return_native_state and (G_history or snapshot_cycles is not None or save):
            raise ValueError(
                "return_native_state=True requires G_history=False, no snapshot_cycles, "
                "and save=False."
            )
        if bool(mean_replacement):
            incompatible = {
                "postselect": bool(postselect) or float(postselect_probability) != 0.0,
                "record_observer": record_observer is not None,
                "frozen_outcomes": frozen_outcomes is not None,
                "noise": float(onsite_phase_noise_sigma) != 0.0 or noise_observer is not None,
                "choi": bool(track_choi) or choi_observer is not None,
                "G_init_prepared": G_init_prepared,
            }
            bad = [key for key, value in incompatible.items() if value]
            if bad:
                raise ValueError(f"mean_replacement is incompatible with {bad}")
            mean_result = self._run_mean_replacement_circuit(
                cycles=cycles,
                samples=1 if samples is None else samples,
                init_mode=init_mode,
                G_init=G_init,
                sequence=sequence,
                meas_slab_only=meas_slab_only,
                batch_size=batch_size,
                frozen_schedule=frozen_schedule,
                cycle_observer=cycle_observer,
                native_cycle_observer=native_cycle_observer,
                lyapunov_frame_observer=lyapunov_frame_observer,
                lyapunov_nvec=lyapunov_nvec,
                lyapunov_start_cycle=lyapunov_start_cycle,
                lyapunov_track_restricted_core=lyapunov_track_restricted_core,
                lyapunov_singular_tol=lyapunov_singular_tol,
                lyapunov_failure_mode=lyapunov_failure_mode,
                return_data=return_data,
            )
            mean_result.update(
                {
                    "state_representation_requested": state_representation_requested,
                    "state_representation": "covariance",
                    "state_representation_resolved": "covariance",
                    "state_representation_resolution_reason": resolution_reason,
                    "initial_purity_tolerance": float(initial_purity_tolerance),
                    "initial_purity_defects": initial_purity_defects,
                    "explicit_covariance_override": explicit_covariance_override,
                    "frame_algorithm_version": None,
                    "covariance_materialization_count": 0,
                    "covariance_materializations": [],
                    "require_no_covariance_materialization": bool(
                        require_no_covariance_materialization
                    ),
                    "native_cycle_observer": native_cycle_observer is not None,
                }
            )
            return mean_result
        if resume and not save:
            raise ValueError("resume=True requires save=True")
        if run_dir_style not in ("cache_key", "compact"):
            raise ValueError("run_dir_style must be either 'cache_key' or 'compact'.")
        track_choi = bool(track_choi)
        if choi_observer is not None and not track_choi:
            raise ValueError("choi_observer requires track_choi=True.")
        if choi_observer_cycles is not None and choi_observer is None:
            raise ValueError("choi_observer_cycles requires choi_observer.")
        if track_choi and site_observer is not None:
            raise ValueError("track_choi cannot currently be combined with site_observer.")
        onsite_phase_noise_sigma = float(onsite_phase_noise_sigma)
        if not np.isfinite(onsite_phase_noise_sigma) or onsite_phase_noise_sigma < 0.0:
            raise ValueError("onsite_phase_noise_sigma must be a finite nonnegative scalar.")
        if onsite_phase_noise_sigma > 0.0 and track_choi:
            raise ValueError(
                "BPJ onsite phase noise is not yet implemented for the reduced Choi tracker."
            )
        choi_singular_tol = float(choi_singular_tol)
        if not np.isfinite(choi_singular_tol) or choi_singular_tol <= 0.0:
            raise ValueError("choi_singular_tol must be a positive finite scalar.")
        choi_failure_mode = str(choi_failure_mode).strip().lower()
        if choi_failure_mode not in ("raise", "censor"):
            raise ValueError("choi_failure_mode must be either 'raise' or 'censor'.")
        if track_choi and resume:
            raise ValueError("track_choi is not resume-aware; start a fresh Choi-observed run.")

        p_gain_requested = p_gain
        p_loss_requested = p_loss
        n_a, p_gain_eff, p_loss_eff = self._resolve_feedback_probabilities(
            n_a=n_a, p_gain=p_gain, p_loss=p_loss
        )
        feedback_key_suffix = (
            ""
            if p_gain_requested is None and p_loss_requested is None
            else f"_pg{p_gain_eff:g}_pl{p_loss_eff:g}"
        )

        cycles = 5 if cycles is None else int(cycles)
        lyapunov_enabled = (
            lyapunov_observer is not None
            or lyapunov_frame_observer is not None
        )
        lyapunov_start_cycle = int(lyapunov_start_cycle)
        if lyapunov_enabled and not (1 <= lyapunov_start_cycle <= cycles):
            raise ValueError(
                "lyapunov_start_cycle is the first propagated cycle and must "
                f"lie in 1..{cycles}; got {lyapunov_start_cycle}."
            )
        lyapunov_singular_tol = float(lyapunov_singular_tol)
        if (
            not np.isfinite(lyapunov_singular_tol)
            or lyapunov_singular_tol <= 0.0
        ):
            raise ValueError(
                "lyapunov_singular_tol must be a positive finite scalar."
            )
        lyapunov_failure_mode = str(
            lyapunov_failure_mode
        ).strip().lower()
        if lyapunov_failure_mode not in ("raise", "censor"):
            raise ValueError(
                "lyapunov_failure_mode must be either 'raise' or 'censor'."
            )
        lyapunov_track_restricted_core = bool(
            lyapunov_track_restricted_core
            or lyapunov_track_record_fisher
        )
        lyapunov_track_record_fisher = bool(
            lyapunov_track_record_fisher
        )
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
            or lyapunov_track_restricted_core
            or lyapunov_track_record_fisher
        ) and not lyapunov_enabled:
            raise ValueError(
                "Custom tangent frames and restricted diagnostics require a "
                "lyapunov_observer or lyapunov_frame_observer."
            )
        if lyapunov_enabled and resume:
            raise ValueError(
                "Lyapunov tracking is not resume-aware; start a fresh run."
            )
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
        if record_observer is not None and postselect_probability != 0.0:
            raise ValueError(
                "record_observer currently records realized Born branches and "
                "requires postselect_probability=0."
            )
        if (frozen_schedule is None) != (frozen_outcomes is None):
            raise ValueError(
                "frozen_schedule and frozen_outcomes must be provided together."
            )
        frozen_record_replay = frozen_schedule is not None
        if frozen_record_replay and postselect_probability != 0.0:
            raise ValueError(
                "frozen-record replay requires postselect_probability=0."
            )
        effective_postselect = postselect_probability == 1.0
        site_update_batching = (
            "padded_variable_rank_frame_v2"
            if frame_native
            else
            "sitewise_rank1_full_trajectory_batch_v1"
            if (
                postselect_probability == 0.0
                and site_observer is None
                and not track_choi
                and not self.triv_region_local_mode
                and not lyapunov_track_record_fisher
            )
            else "grouped_site_reference_v1"
        )
        meas_slab_only_requested = bool(meas_slab_only)
        meas_slab_only_effective = self._meas_slab_only_effective(meas_slab_only_requested)
        if frozen_exterior_outcomes is not None and not meas_slab_only_effective:
            raise ValueError(
                "frozen_exterior_outcomes is only valid for an effective hard slab."
            )
        active_top_layer_indices = self.active_top_layer_indices(meas_slab_only=meas_slab_only_requested)
        active_nlayer = int(active_top_layer_indices.numel())
        lyapunov_basis_dim = (
            active_nlayer if meas_slab_only_effective else self.Nlayer
        )
        if lyapunov_initial_frame is not None:
            frame_shape = getattr(lyapunov_initial_frame, "shape", None)
            if frame_shape is None:
                frame_shape = np.asarray(lyapunov_initial_frame).shape
            lyapunov_initial_frame_shape = [
                int(value) for value in frame_shape
            ]
        else:
            lyapunov_initial_frame_shape = None
        if lyapunov_initial_frame is not None and lyapunov_nvec is None:
            lyapunov_nvec_eff = int(
                lyapunov_initial_frame_shape[-1]
            )
        else:
            lyapunov_nvec_eff = (
                lyapunov_basis_dim
                if lyapunov_nvec is None
                else int(lyapunov_nvec)
            )
        if lyapunov_enabled and (
            lyapunov_nvec_eff <= 0
            or lyapunov_nvec_eff > lyapunov_basis_dim
        ):
            raise ValueError(
                "lyapunov_nvec must satisfy 1 <= lyapunov_nvec <= "
                f"{lyapunov_basis_dim}; got {lyapunov_nvec_eff}"
            )
        samples = 1 if samples is None else int(samples)
        if samples <= 0:
            raise ValueError("samples must be a positive integer")
        if effective_postselect:
            if samples != 1:
                print("[info] postselect_probability=1 overrides samples; using samples=1.")
            samples = 1

        snapshot_cycles_norm = self._normalize_snapshot_cycles(snapshot_cycles, cycles)
        choi_observer_cycles_norm = self._normalize_choi_observer_cycles(choi_observer_cycles, cycles)
        cycle_observer_cycles_norm = self._normalize_choi_observer_cycles(
            cycle_observer_cycles, cycles
        )
        native_cycle_observer_cycles_norm = self._normalize_choi_observer_cycles(
            native_cycle_observer_cycles, cycles
        )
        lyapunov_observer_cycles_norm = self._normalize_choi_observer_cycles(
            lyapunov_observer_cycles, cycles
        )
        if cycle_observer_cycles_norm is not None and cycle_observer is None:
            raise ValueError("cycle_observer_cycles requires cycle_observer.")
        if (
            native_cycle_observer_cycles_norm is not None
            and native_cycle_observer is None
        ):
            raise ValueError(
                "native_cycle_observer_cycles requires native_cycle_observer."
            )
        if lyapunov_observer_cycles_norm is not None and not lyapunov_enabled:
            raise ValueError(
                "lyapunov_observer_cycles requires a Lyapunov observer."
            )
        if choi_observer is not None and choi_observer_cycles_norm is None:
            choi_observer_cycles_norm = [cycles]
        if snapshot_cycles_norm is not None and G_history:
            raise ValueError("snapshot_cycles requires G_history=False.")
        store_mode_for_batch = "snapshots" if snapshot_cycles_norm is not None else ("history" if G_history else "final")

        batch_size_auto_info = None
        if effective_postselect:
            batch_size = 1
            batch_size_mode = "postselect"
        elif batch_size is None:
            batch_size, batch_size_auto_info = self._auto_batch_size_for_a100_40gb(
                samples=samples,
                cycles=cycles,
                store_mode=store_mode_for_batch,
                snapshot_cycles=snapshot_cycles_norm,
                init_mode=init_mode,
                state_representation=state_representation_resolved,
                track_choi=track_choi,
                choi_nlayer=active_nlayer if meas_slab_only_effective else self.Nlayer,
            )
            batch_size_mode = "auto_a100_40gb"
            print(
                "[info] auto-selected GPU batch_size="
                f"{batch_size} for N{self.Nx}x{self.Ny}, samples={samples}, cycles={cycles}, "
                f"store_mode={store_mode_for_batch}, backend={self.backend}, dtype={self.dtype}",
                flush=True,
            )
        else:
            batch_size = int(batch_size)
            batch_size_mode = "explicit"
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if lyapunov_basis_mode == "pure_occupied_empty" and batch_size != 1:
            raise ValueError(
                "pure_occupied_empty exact block propagation requires batch_size=1."
            )
        if skip_batch_indices is None:
            skip_batch_index_set = set()
        else:
            skip_batch_index_set = {int(idx) for idx in skip_batch_indices}
            if any(idx < 0 for idx in skip_batch_index_set):
                raise ValueError("skip_batch_indices must contain nonnegative integers only")

        def _materialize_gpu(state, *, reason, cycle, batch_index, batch_start):
            if not isinstance(state, BatchedOccupiedFrameState):
                return state
            if require_no_covariance_materialization:
                raise RuntimeError(
                    "require_no_covariance_materialization=True forbids the requested "
                    f"GPU covariance reconstruction ({reason}, cycle={cycle}, "
                    f"batch={batch_index})."
                )
            covariance_materializations.append(
                {
                    "reason": str(reason),
                    "cycle": int(cycle),
                    "batch_index": int(batch_index),
                    "batch_start": int(batch_start),
                }
            )
            return state.centered_covariance(reason=str(reason))

        def _emit_native_cycle_observer(*, cycle, state, batch_index, batch_start, batch_count):
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
                            "batch_index": int(batch_index),
                            "batch_start": int(batch_start),
                            "requested_by": "native_cycle_observer",
                        }
                    )
                if require_no_covariance_materialization:
                    raise RuntimeError(
                        "require_no_covariance_materialization=True was violated by "
                        f"native_cycle_observer at cycle={cycle}, batch={batch_index}."
                    )

        with torch.inference_mode():
            seq_info = self._sequence_helper(sequence, meas_slab_only=meas_slab_only_requested)
            sequence_mode = seq_info["mode"]
            coords_for_len = seq_info["coords_for_len"]
            schedule_state = self._prepare_schedule_state(sequence_mode, coords_for_len)
            snapshot_cycle_set = set(snapshot_cycles_norm or ())
            choi_observer_cycle_set = set(choi_observer_cycles_norm or ())

            sites_per_cycle = len(coords_for_len)
            frozen_schedule_tensor = None
            frozen_outcomes_tensor = None
            frozen_exterior_outcomes_tensor = None
            if frozen_record_replay:
                frozen_schedule_tensor = torch.as_tensor(
                    frozen_schedule, dtype=torch.long, device=self.device
                )
                frozen_outcomes_tensor = torch.as_tensor(
                    frozen_outcomes, dtype=torch.bool, device=self.device
                )
                expected_schedule_shape = (samples, cycles, sites_per_cycle)
                expected_outcome_shape = (samples, cycles, sites_per_cycle, 4)
                if tuple(frozen_schedule_tensor.shape) != expected_schedule_shape:
                    raise ValueError(
                        "frozen_schedule must have shape "
                        f"{expected_schedule_shape}; got {tuple(frozen_schedule_tensor.shape)}"
                    )
                if tuple(frozen_outcomes_tensor.shape) != expected_outcome_shape:
                    raise ValueError(
                        "frozen_outcomes must have shape "
                        f"{expected_outcome_shape}; got {tuple(frozen_outcomes_tensor.shape)}"
                    )
                expected_site_ids = torch.as_tensor(
                    [int(x + self.Nx * y) for x, y in coords_for_len],
                    dtype=torch.long,
                    device=self.device,
                )
                sorted_expected = torch.sort(expected_site_ids).values
                sorted_words = torch.sort(frozen_schedule_tensor, dim=-1).values
                if not bool(
                    torch.all(
                        sorted_words == sorted_expected[None, None, :]
                    ).item()
                ):
                    raise ValueError(
                        "every frozen schedule word must be a permutation of the active centers"
                    )
            if frozen_exterior_outcomes is not None:
                frozen_exterior_outcomes_tensor = torch.as_tensor(
                    frozen_exterior_outcomes,
                    dtype=torch.bool,
                    device=self.device,
                )
                expected_exterior_shape = (
                    samples,
                    2 * len(self._exterior_site_ids()),
                )
                if tuple(frozen_exterior_outcomes_tensor.shape) != expected_exterior_shape:
                    raise ValueError(
                        "frozen_exterior_outcomes must have shape "
                        f"{expected_exterior_shape}; got "
                        f"{tuple(frozen_exterior_outcomes_tensor.shape)}"
                    )
            dtype_name = "c64" if self.dtype == torch.complex64 else "c128"
            store_mode = store_mode_for_batch
            save_root = self._gpu_cache_root(save_dir=save_dir) if save else None

            key = self._cache_key(
                cycles=cycles,
                samples=samples,
                init_mode=init_mode,
                n_a=n_a,
                seq=sequence_mode,
                store_mode=store_mode,
                snapshot_cycles=snapshot_cycles_norm,
                ps=effective_postselect,
                psp=postselect_probability,
                pc=perfect_correction,
                dtype_name=dtype_name,
                backend_name=self.backend,
                mslab=meas_slab_only_effective,
                state_representation=state_representation_resolved,
                frame_init_prepared=frame_init_prepared,
                G_init_prepared=G_init_prepared,
                feedback_suffix=feedback_key_suffix,
            )
            suffix_tag = self._sanitize_path_component(save_suffix)
            run_config = {
                "Nx": self.Nx,
                "Ny": self.Ny,
                "cycles": cycles,
                "samples": samples,
                "batch_size": batch_size,
                "batch_size_mode": batch_size_mode,
                "store_mode": store_mode,
                "alpha_top": float(np.real(self.alpha_top)),
                "alpha_triv": float(np.real(self.alpha_triv)),
                "trial_orbitals": self.trial_orbitals,
                "dw_truncation": bool(self.dw_truncation),
                "meas_slab_only_requested": meas_slab_only_requested,
                "meas_slab_only_effective": meas_slab_only_effective,
                "active_top_layer_indices": active_top_layer_indices.detach().cpu().tolist(),
                "Nlayer_slab": active_nlayer,
                "exterior_preparation": (
                    "skipped_prepared_frame"
                    if meas_slab_only_effective and frame_init_prepared
                    else "skipped_prepared_covariance"
                    if meas_slab_only_effective and G_init_prepared
                    else
                    "forced_occupied_onsite_before_cycle_0"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned_onsite_before_cycle_0"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_mode": (
                    "already_prepared"
                    if meas_slab_only_effective and initial_state_prepared
                    else
                    "forced_occupied"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_performed": bool(
                    meas_slab_only_effective and not initial_state_prepared
                ),
                "exterior_preparation_basis": (
                    "canonical_unit_cell_orbital"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_orbital_count": (
                    2 * len(self._exterior_site_ids())
                    if meas_slab_only_effective
                    else 0
                ),
                "init_mode": init_mode,
                "n_a": float(n_a),
                "p_gain": None if p_gain_requested is None else float(p_gain_requested),
                "p_loss": None if p_loss_requested is None else float(p_loss_requested),
                "p_gain_effective": float(p_gain_eff),
                "p_loss_effective": float(p_loss_eff),
                "canonical_dynamics_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
                "sequence": sequence_mode,
                "postselect": bool(effective_postselect),
                "postselect_probability": float(postselect_probability),
                "perfect_correction": bool(perfect_correction),
                "dtype": dtype_name,
                "backend": self.backend,
                "save_init": bool(save_init),
                "save_history_stride": None if save_history_stride is None else int(save_history_stride),
                "snapshot_cycles": snapshot_cycles_norm,
                "physical_covariance_update": "rank1_resolvent_v1",
                "site_update_batching": site_update_batching,
                "track_choi": track_choi,
                "choi_observer_cycles": choi_observer_cycles_norm,
                "choi_singular_tol": choi_singular_tol,
                "choi_failure_mode": choi_failure_mode,
                "choi_initialization": "Sigma_LL=0,Sigma_LR=I,Sigma_RR=0",
                "choi_formula": "regularized_resolvent_rank_one_v2",
                "choi_basis": "reduced_topological_slab" if meas_slab_only_effective else "full_top_layer",
                "lyapunov_observer": lyapunov_observer is not None,
                "record_observer": record_observer is not None,
                "record_formula": (
                    "ordered_site_channel_born_record_v1"
                    if record_observer is not None
                    else None
                ),
                "onsite_phase_noise_sigma": float(onsite_phase_noise_sigma),
                "onsite_phase_noise_formula": (
                    "BPJ_iid_uniform_onsite_U1_between_cycles_v1"
                    if onsite_phase_noise_sigma > 0.0
                    else None
                ),
                "noise_observer": noise_observer is not None,
                "frozen_record_replay": bool(frozen_record_replay),
                "frozen_schedule_signature": self._array_signature(
                    frozen_schedule
                ),
                "frozen_outcomes_signature": self._array_signature(
                    frozen_outcomes
                ),
                "frozen_exterior_outcomes_signature": self._array_signature(
                    frozen_exterior_outcomes
                ),
                "exterior_outcome_observer": exterior_outcome_observer is not None,
                "lyapunov_frame_observer": (
                    lyapunov_frame_observer is not None
                ),
                "lyapunov_basis_mode": lyapunov_basis_mode,
                "lyapunov_observer_cycles": lyapunov_observer_cycles_norm,
                "lyapunov_nvec": int(lyapunov_nvec_eff),
                "lyapunov_initial_frame_shape": (
                    lyapunov_initial_frame_shape
                ),
                "lyapunov_initial_frame_signature": self._array_signature(
                    lyapunov_initial_frame
                ),
                "lyapunov_start_cycle": int(lyapunov_start_cycle),
                "lyapunov_observation_cycles": int(
                    cycles - lyapunov_start_cycle + 1
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
                "nshell": self.nshell,
                "DW": bool(self.DW),
                "filling_frac": float(self.filling_frac),
                "suffix_tag": suffix_tag,
                "G_init_signature": self._array_signature(G_init),
                "frame_init_signature": self._array_signature(frame_init),
                "frame_ranks_signature": self._array_signature(frame_ranks),
                "frame_init_prepared": frame_init_prepared,
                "G_init_prepared": G_init_prepared,
                "state_representation_requested": state_representation_requested,
                "state_representation_resolved": state_representation_resolved,
                "state_representation_resolution_reason": resolution_reason,
                "initial_purity_tolerance": float(initial_purity_tolerance),
                "initial_purity_defects": initial_purity_defects,
                "explicit_covariance_override": explicit_covariance_override,
                "frame_algorithm_version": (
                    GPU_FRAME_ALGORITHM_VERSION if frame_native else None
                ),
                "require_no_covariance_materialization": bool(
                    require_no_covariance_materialization
                ),
                "native_cycle_observer": native_cycle_observer is not None,
                "return_native_state": bool(return_native_state),
            }
            run_id = hashlib.sha1(json.dumps(run_config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
            run_label = key if not suffix_tag else f"{key}_{suffix_tag}"
            run_dir_name = f"run_{run_id}" if run_dir_style == "compact" else f"{run_label}_{run_id}"
            run_dir = os.path.join(save_root, run_dir_name) if save else None
            manifest_path = os.path.join(run_dir, "manifest.json") if save else None

            collect_history_cpu = bool(G_history and (not save) and return_data)
            collect_final_cpu = bool((not G_history) and snapshot_cycles_norm is None and (not save) and return_data)
            collect_snapshots_cpu = bool((snapshot_cycles_norm is not None) and (not save) and return_data)
            histories_cpu = [] if collect_history_cpu else None
            finals_cpu = [] if collect_final_cpu else None
            snapshots_cpu = [] if collect_snapshots_cpu else None
            num_batches = (samples + batch_size - 1) // batch_size
            shard_records = {}
            completed_batches = set()
            choi_diagnostics = [] if track_choi else None
            lyapunov_diagnostics = [] if lyapunov_enabled else None
            frame_diagnostics = [] if frame_native else None

            if save:
                self._ensure_outdir(run_dir)
                if resume and os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    if manifest.get("config") != run_config:
                        raise ValueError(
                            f"Existing manifest at {manifest_path} does not match the current run configuration."
                        )
                    for batch_key, record in manifest.get("shards", {}).items():
                        batch_idx = int(batch_key)
                        shard_path = os.path.join(run_dir, record["filename"])
                        if os.path.exists(shard_path):
                            shard_records[batch_idx] = record
                            completed_batches.add(batch_idx)
                else:
                    manifest = None

                if manifest is None:
                    manifest = {
                        "version": 1,
                        "run_id": run_id,
                        "key": key,
                        "storage": {
                            "run_dir_style": run_dir_style,
                            "run_dir_name": run_dir_name,
                        },
                        "config": run_config,
                        "num_batches": num_batches,
                        "store_mode": store_mode,
                        "completed_batches": [],
                        "shards": {},
                        "batch_size_auto_info": batch_size_auto_info,
                    }
                    self._write_json_atomic(manifest_path, manifest)

            non_tty_progress = bool(progress and not sys.stdout.isatty())
            outer_pbar = (
                tqdm(
                    total=samples,
                    desc="Markov GPU batches",
                    unit="sample",
                    leave=True,
                    file=sys.stdout,
                    dynamic_ncols=False,
                )
                if progress
                else None
            )

            for batch_idx, batch_start in enumerate(range(0, samples, batch_size)):
                batch_count = min(batch_size, samples - batch_start)
                shard_filename = f"batch_{batch_idx:05d}.npy" if not suffix_tag else f"batch_{batch_idx:05d}_{suffix_tag}.npy"
                shard_path = os.path.join(run_dir, shard_filename) if save else None

                if non_tty_progress:
                    print(
                        f"[progress] batch {batch_idx + 1}/{num_batches} "
                        f"samples {batch_start}:{batch_start + batch_count}",
                        flush=True,
                    )

                if batch_idx in skip_batch_index_set:
                    if outer_pbar is not None:
                        outer_pbar.update(batch_count)
                    continue

                if save and batch_idx in completed_batches:
                    shard_records[batch_idx] = {
                        "filename": shard_filename,
                        "sample_start": batch_start,
                        "sample_stop": batch_start + batch_count,
                    }
                    if outer_pbar is not None:
                        outer_pbar.update(batch_count)
                    continue

                if frame_native:
                    G_top = self._prepare_initial_frame_batch(
                        batch_count,
                        init_mode=init_mode,
                        G_init=G_init,
                        frame_init=frame_init,
                        frame_ranks=frame_ranks,
                        sample_offset=batch_start,
                        purity_tolerance=initial_purity_tolerance,
                    )
                else:
                    G_top = self._prepare_initial_batch(
                        batch_size=batch_count,
                        init_mode=init_mode,
                        G_init=G_init,
                        sample_offset=batch_start,
                    )
                frozen_schedule_batch = (
                    None
                    if frozen_schedule_tensor is None
                    else frozen_schedule_tensor[
                        batch_start : batch_start + batch_count
                    ]
                )
                frozen_outcomes_batch = (
                    None
                    if frozen_outcomes_tensor is None
                    else frozen_outcomes_tensor[
                        batch_start : batch_start + batch_count
                    ]
                )
                if meas_slab_only_effective and not initial_state_prepared:
                    exterior_overrides_batch = (
                        None
                        if frozen_exterior_outcomes_tensor is None
                        else frozen_exterior_outcomes_tensor[
                            batch_start : batch_start + batch_count
                        ]
                    )
                    if frame_native:
                        G_top = self._prepare_exterior_product_frame_batched(
                            G_top,
                            mode=(
                                "forced_occupied"
                                if effective_postselect
                                else "born_conditioned"
                            ),
                            outcome_observer=exterior_outcome_observer,
                            outcome_overrides=exterior_overrides_batch,
                            batch_start=batch_start,
                        )
                    else:
                        G_top = self._prepare_exterior_product_state_batched(
                            G_top,
                            mode=(
                                "forced_occupied"
                                if effective_postselect
                                else "born_conditioned"
                            ),
                            outcome_observer=exterior_outcome_observer,
                            outcome_overrides=exterior_overrides_batch,
                            batch_start=batch_start,
                        )

                history_batch = [] if (G_history and (save or return_data)) else None
                snapshot_batch = [] if (snapshot_cycles_norm is not None and (save or return_data)) else None
                last_saved_cycle = 0
                if history_batch is not None and save_init:
                    initial_output = (
                        _materialize_gpu(
                            G_top,
                            reason="history",
                            cycle=0,
                            batch_index=batch_idx,
                            batch_start=batch_start,
                        )
                        if frame_native
                        else G_top
                    )
                    history_batch.append(initial_output.detach().cpu())
                # Native start avoids reinitializing the physical trajectory
                # or exterior merely to begin the tangent observation window.
                lyapunov_state = None
                choi_state = (
                    self._init_choi_state(
                        batch_count=batch_count,
                        singular_tol=choi_singular_tol,
                        basis_idx=active_top_layer_indices if meas_slab_only_effective else None,
                        failure_mode=choi_failure_mode,
                    )
                    if track_choi
                    else None
                )
                if cycle_observer is not None and cycle_observer_cycles_norm is None:
                    observed = (
                        _materialize_gpu(
                            G_top,
                            reason="cycle_observer",
                            cycle=0,
                            batch_index=batch_idx,
                            batch_start=batch_start,
                        )
                        if frame_native
                        else G_top
                    )
                    cycle_observer(
                        cycle=0,
                        G=observed,
                        batch_index=batch_idx,
                        batch_start=batch_start,
                        batch_count=batch_count,
                    )
                if (
                    native_cycle_observer is not None
                    and native_cycle_observer_cycles_norm is None
                ):
                    _emit_native_cycle_observer(
                        cycle=0,
                        state=G_top,
                        batch_index=batch_idx,
                        batch_start=batch_start,
                        batch_count=batch_count,
                    )

                static_schedules = None
                if sequence_mode in ("raster_y", "raster_x"):
                    static_schedules = self._sequence_schedule_batch(schedule_state, batch_count)
                emit_site_observer = (
                    site_observer is not None
                    and not effective_postselect
                    and bool(perfect_correction)
                )
                emit_record_observer = (
                    record_observer is not None and not effective_postselect
                )

                cycle_pbar = (
                    tqdm(
                        total=cycles,
                        desc=f"Markov cycles batch {batch_idx + 1}/{num_batches}",
                        unit="cycle",
                        leave=False,
                        file=sys.stdout,
                        dynamic_ncols=False,
                    )
                    if progress
                    else None
                )
                for cyc in range(1, cycles + 1):
                    if lyapunov_enabled and cyc == lyapunov_start_cycle:
                        tangent_basis_idx = (
                            active_top_layer_indices
                            if meas_slab_only_effective
                            else None
                        )
                        if lyapunov_basis_mode == "pure_occupied_empty":
                            lyapunov_state = (
                                self._init_pure_occupied_empty_lyapunov_state(
                                    G_top,
                                    basis_idx=tangent_basis_idx,
                                    singular_tol=lyapunov_singular_tol,
                                    failure_mode=lyapunov_failure_mode,
                                    purity_tolerance=initial_purity_tolerance,
                                )
                            )
                        else:
                            lyapunov_state = self._init_lyapunov_state(
                                batch_count=batch_count,
                                n_vec=lyapunov_nvec_eff,
                                basis_idx=tangent_basis_idx,
                                initial_frame=lyapunov_initial_frame,
                                sample_start=batch_start,
                                track_restricted_core=(
                                    lyapunov_track_restricted_core
                                ),
                                track_record_fisher=(
                                    lyapunov_track_record_fisher
                                ),
                                singular_tol=lyapunov_singular_tol,
                                failure_mode=lyapunov_failure_mode,
                            )
                    schedules = (
                        None
                        if frozen_schedule_batch is None
                        else frozen_schedule_batch[:, cyc - 1, :]
                    )
                    if schedules is None:
                        schedules = static_schedules
                    if schedules is None:
                        schedules = self._sequence_schedule_batch(schedule_state, batch_count)
                    for site_idx in range(sites_per_cycle):
                        site_ids = schedules[:, site_idx]
                        outcome_overrides = (
                            None
                            if frozen_outcomes_batch is None
                            else frozen_outcomes_batch[
                                :, cyc - 1, site_idx, :
                            ]
                        )
                        if frame_native:
                            G_top = self._apply_frame_site_updates(
                                G_top,
                                site_ids,
                                postselect_probability=postselect_probability,
                                n_a=n_a,
                                p_gain=p_gain_eff,
                                p_loss=p_loss_eff,
                                perfect_correction=perfect_correction,
                                site_observer=(
                                    site_observer if emit_site_observer else None
                                ),
                                record_observer=(
                                    record_observer
                                    if emit_record_observer
                                    else None
                                ),
                                outcome_overrides=outcome_overrides,
                                lyapunov_state=lyapunov_state,
                                choi_state=choi_state,
                                cycle=cyc,
                                update_index=site_idx,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )
                        elif effective_postselect:
                            G_top = self._apply_grouped_site_updates(
                                G_top,
                                site_ids,
                                postselect=True,
                                lyapunov_state=lyapunov_state,
                                choi_state=choi_state,
                                cycle=cyc,
                                update_index=site_idx,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )
                        else:
                            G_top = self._apply_grouped_site_updates(
                                G_top,
                                site_ids,
                                postselect=False,
                                postselect_probability=postselect_probability,
                                n_a=n_a,
                                p_gain=p_gain_eff,
                                p_loss=p_loss_eff,
                                perfect_correction=perfect_correction,
                                site_observer=site_observer if emit_site_observer else None,
                                record_observer=(
                                    record_observer
                                    if emit_record_observer
                                    else None
                                ),
                                outcome_overrides=outcome_overrides,
                                lyapunov_state=lyapunov_state,
                                choi_state=choi_state,
                                cycle=cyc,
                                update_index=site_idx,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )

                    if onsite_phase_noise_sigma > 0.0 or noise_observer is not None:
                        G_top, noise_theta, noise_phase = self._apply_onsite_phase_noise(
                            G_top,
                            sigma=onsite_phase_noise_sigma,
                            lyapunov_state=lyapunov_state,
                        )
                        if noise_observer is not None:
                            noise_observer(
                                cycle=int(cyc),
                                theta=noise_theta,
                                phase=noise_phase,
                                sigma=float(onsite_phase_noise_sigma),
                                sample_indices=torch.arange(
                                    batch_start,
                                    batch_start + batch_count,
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                batch_index=int(batch_idx),
                                batch_start=int(batch_start),
                                batch_count=int(batch_count),
                            )

                    if covariance_spectral_clip:
                        clip_diagnostics = self.clip_covariance_spectrum(
                            G_top, sample_chunk=covariance_clip_sample_chunk,
                            max_correction=covariance_clip_max_correction,
                        )
                        if covariance_clip_observer is not None:
                            covariance_clip_observer(cycle=int(cyc), diagnostics=clip_diagnostics,
                                batch_index=batch_idx, batch_start=batch_start, batch_count=batch_count)

                    if snapshot_batch is not None and cyc in snapshot_cycle_set:
                        snapshot_state = (
                            _materialize_gpu(
                                G_top,
                                reason="snapshot",
                                cycle=cyc,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )
                            if frame_native
                            else G_top
                        )
                        snapshot_batch.append(snapshot_state.detach().cpu())

                    if history_batch is not None:
                        if save_history_stride is None or (cyc % int(max(1, save_history_stride))) == 0:
                            history_state = (
                                _materialize_gpu(
                                    G_top,
                                    reason="history",
                                    cycle=cyc,
                                    batch_index=batch_idx,
                                    batch_start=batch_start,
                                )
                                if frame_native
                                else G_top
                            )
                            history_batch.append(history_state.detach().cpu())
                            last_saved_cycle = cyc

                    if frame_native and cyc % frame_reorthonormalize_interval == 0:
                        gram_residual = G_top.reorthonormalize()
                        hard = 2e-4 if self.dtype == torch.complex64 else 1e-9
                        if torch.any(~torch.isfinite(gram_residual)) or torch.any(
                            gram_residual > hard
                        ):
                            raise FloatingPointError(
                                "GPU occupied-frame Gram residual exceeded the runtime ceiling."
                            )

                    if cycle_observer is not None and (
                        cycle_observer_cycles_norm is None
                        or cyc in set(cycle_observer_cycles_norm)
                    ):
                        observed = (
                            _materialize_gpu(
                                G_top,
                                reason="cycle_observer",
                                cycle=cyc,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )
                            if frame_native
                            else G_top
                        )
                        cycle_observer(
                            cycle=cyc,
                            G=observed,
                            batch_index=batch_idx,
                            batch_start=batch_start,
                            batch_count=batch_count,
                        )
                    if native_cycle_observer is not None and (
                        native_cycle_observer_cycles_norm is None
                        or cyc in set(native_cycle_observer_cycles_norm)
                    ):
                        _emit_native_cycle_observer(
                            cycle=cyc,
                            state=G_top,
                            batch_index=batch_idx,
                            batch_start=batch_start,
                            batch_count=batch_count,
                        )
                    if lyapunov_state is not None:
                        elapsed_lyapunov_cycle = int(
                            cyc - lyapunov_start_cycle + 1
                        )
                        spectra = self._lyapunov_end_cycle(
                            lyapunov_state, elapsed_lyapunov_cycle
                        )
                        lyapunov_extra = (
                            self._lyapunov_min_abs_vector_payload(
                                lyapunov_state, elapsed_lyapunov_cycle
                            )
                            if cyc == cycles
                            else {}
                        )
                        emit_lyapunov = (
                            lyapunov_observer_cycles_norm is None
                            or cyc in set(lyapunov_observer_cycles_norm)
                        )
                        if lyapunov_observer is not None and emit_lyapunov:
                            lyapunov_observer(
                                cycle=elapsed_lyapunov_cycle,
                                spectra=spectra,
                                G=G_top,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                                batch_count=batch_count,
                                **lyapunov_extra,
                            )
                        if lyapunov_frame_observer is not None and emit_lyapunov:
                            lyapunov_frame_observer(
                                cycle=int(cyc),
                                lyapunov_cycle=elapsed_lyapunov_cycle,
                                spectra=spectra,
                                G=G_top,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                                batch_count=batch_count,
                                **self._lyapunov_frame_payload(
                                    lyapunov_state
                                ),
                            )
                        if int(cyc) in reset_cycles:
                            self._reset_lyapunov_accumulator(lyapunov_state)
                    if choi_observer is not None and cyc in choi_observer_cycle_set:
                        choi_observer_payload = {
                            "cycle": cyc,
                            "G": G_top,
                            "sigma_ll": choi_state["LL"],
                            "sigma_lr": choi_state["LR"],
                            "sigma_rr": choi_state["RR"],
                            "batch_index": batch_idx,
                            "batch_start": batch_start,
                            "batch_count": batch_count,
                            "min_abs_d": choi_state["min_abs_d"],
                            "min_abs_d_context": choi_state["min_abs_d_context"],
                            "choi_active_mask": choi_state["active"],
                            "choi_failure_records": tuple(choi_state["failure_records"]),
                        }
                        if meas_slab_only_effective:
                            choi_observer_payload.update(
                                {
                                    "active_top_layer_indices": choi_state["basis_idx"],
                                    "full_nlayer": self.Nlayer,
                                }
                            )
                        choi_observer_result = choi_observer(
                            **choi_observer_payload,
                        )
                        if choi_observer_result is not None:
                            if choi_failure_mode != "censor":
                                raise RuntimeError(
                                    "choi_observer requested trajectory censoring while choi_failure_mode is not 'censor'."
                                )
                            deactivate = choi_observer_result.get("deactivate_sample_offsets", ())
                            reasons = choi_observer_result.get("failure_records", ())
                            if len(deactivate):
                                deactivate = torch.as_tensor(deactivate, dtype=torch.long, device=self.device)
                                choi_state["active"][deactivate] = False
                            choi_state["failure_records"].extend(dict(record) for record in reasons)

                    if cycle_pbar is not None:
                        cycle_pbar.update(1)

                if cycle_pbar is not None:
                    cycle_pbar.close()

                if history_batch is not None and save_history_stride is not None and last_saved_cycle != cycles:
                    final_history_state = (
                        _materialize_gpu(
                            G_top,
                            reason="history_final_stride",
                            cycle=cycles,
                            batch_index=batch_idx,
                            batch_start=batch_start,
                        )
                        if frame_native
                        else G_top
                    )
                    history_batch.append(final_history_state.detach().cpu())
                if snapshot_batch is not None and len(snapshot_batch) != len(snapshot_cycles_norm):
                    raise RuntimeError(
                        f"Collected {len(snapshot_batch)} snapshots, expected {len(snapshot_cycles_norm)}."
                    )
                if track_choi:
                    choi_diagnostics.append(
                        {
                            "batch_index": int(batch_idx),
                            "batch_start": int(batch_start),
                            "batch_count": int(batch_count),
                            "min_abs_d": float(choi_state["min_abs_d"]),
                            "min_abs_d_context": choi_state["min_abs_d_context"],
                            "choi_active_final": choi_state["active"].detach().cpu().tolist(),
                            "choi_failure_records": list(choi_state["failure_records"]),
                        }
                    )

                if lyapunov_state is not None:
                    lyapunov_diagnostics.append(
                        {
                            "batch_index": int(batch_idx),
                            "batch_start": int(batch_start),
                            "batch_count": int(batch_count),
                            "lyapunov_cycles": int(
                                cycles - lyapunov_start_cycle + 1
                            ),
                            "null_counts": lyapunov_state[
                                "null_counts"
                            ].detach().cpu().tolist(),
                            "active_final": lyapunov_state[
                                "active"
                            ].detach().cpu().tolist(),
                            "min_branch_probability": lyapunov_state[
                                "min_branch_probability"
                            ].detach().cpu().tolist(),
                            "min_abs_born_denominator": lyapunov_state[
                                "min_abs_born_denominator"
                            ].detach().cpu().tolist(),
                            "invalid_branch_count": lyapunov_state[
                                "invalid_branch_count"
                            ].detach().cpu().tolist(),
                            "failure_records": list(
                                lyapunov_state["failure_records"]
                            ),
                        }
                    )

                if frame_native:
                    frame_diagnostics.append(
                        {
                            "batch_index": int(batch_idx),
                            "batch_start": int(batch_start),
                            "batch_count": int(batch_count),
                            "ranks": G_top.ranks.detach().cpu().tolist(),
                            "min_ranks": G_top.min_ranks.detach().cpu().tolist(),
                            "max_ranks": G_top.max_ranks.detach().cpu().tolist(),
                            "capacity": int(G_top.capacity),
                            "native_state_bytes": int(G_top.native_state_bytes()),
                            "gram_residual": G_top.gram_residual().detach().cpu().tolist(),
                        }
                    )

                if snapshot_cycles_norm is not None:
                    if save:
                        batch_array = np.asarray(torch.stack(snapshot_batch, dim=1).numpy(), dtype=self.numpy_dtype)
                        self._save_npy_atomic(shard_path, batch_array)
                    elif collect_snapshots_cpu:
                        batch_array = np.asarray(torch.stack(snapshot_batch, dim=1).numpy(), dtype=self.numpy_dtype)
                        snapshots_cpu.append(batch_array)
                elif G_history:
                    if save:
                        batch_array = np.asarray(torch.stack(history_batch, dim=1).numpy(), dtype=self.numpy_dtype)
                        self._save_npy_atomic(shard_path, batch_array)
                    elif collect_history_cpu:
                        batch_array = np.asarray(torch.stack(history_batch, dim=1).numpy(), dtype=self.numpy_dtype)
                        histories_cpu.append(batch_array)
                else:
                    if save:
                        final_state = (
                            _materialize_gpu(
                                G_top,
                                reason="legacy_final_return_or_save",
                                cycle=cycles,
                                batch_index=batch_idx,
                                batch_start=batch_start,
                            )
                            if frame_native
                            else G_top
                        )
                        batch_array = np.asarray(final_state.detach().cpu().numpy(), dtype=self.numpy_dtype)
                        self._save_npy_atomic(shard_path, batch_array)
                    elif collect_final_cpu:
                        if return_native_state:
                            finals_cpu.append(G_top.snapshot(cpu=True))
                        else:
                            final_state = (
                                _materialize_gpu(
                                    G_top,
                                    reason="legacy_final_return_or_save",
                                    cycle=cycles,
                                    batch_index=batch_idx,
                                    batch_start=batch_start,
                                )
                                if frame_native
                                else G_top
                            )
                            batch_array = np.asarray(final_state.detach().cpu().numpy(), dtype=self.numpy_dtype)
                            finals_cpu.append(batch_array)

                if save:
                    shard_records[batch_idx] = {
                        "filename": shard_filename,
                        "sample_start": batch_start,
                        "sample_stop": batch_start + batch_count,
                        "shape": [batch_count, len(snapshot_cycles_norm), self.Nlayer, self.Nlayer] if snapshot_cycles_norm is not None else ([batch_count, len(history_batch) if history_batch is not None else self.Nlayer, self.Nlayer] if G_history else [batch_count, self.Nlayer, self.Nlayer]),
                        "dtype": str(self.numpy_dtype),
                    }
                    manifest["shards"][str(batch_idx)] = shard_records[batch_idx]
                    manifest["completed_batches"] = sorted(shard_records.keys())
                    self._write_json_atomic(manifest_path, manifest)

                if outer_pbar is not None:
                    outer_pbar.update(batch_count)

            if outer_pbar is not None:
                outer_pbar.close()

            if save:
                missing = [idx for idx in range(num_batches) if idx not in shard_records]
                if missing:
                    raise RuntimeError(f"Missing shard files for batches: {missing}")

            merged_save_path = None
            choi_result_metadata = {
                "site_update_batching": site_update_batching,
                "choi_tracked": track_choi,
                "choi_observer_cycles": choi_observer_cycles_norm,
                "choi_diagnostics": choi_diagnostics,
                "choi_failure_mode": choi_failure_mode,
                "lyapunov_tracked": bool(lyapunov_enabled),
                "lyapunov_basis_mode": lyapunov_basis_mode,
                "lyapunov_observer_cycles": lyapunov_observer_cycles_norm,
                "lyapunov_start_cycle": int(lyapunov_start_cycle),
                "lyapunov_accumulator_reset_cycles": list(reset_cycles),
                "lyapunov_observation_cycles": int(
                    cycles - lyapunov_start_cycle + 1
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
                "meas_slab_only_requested": meas_slab_only_requested,
                "meas_slab_only_effective": meas_slab_only_effective,
                "active_top_layer_indices": active_top_layer_indices.detach().cpu().numpy(),
                "Nlayer_slab": active_nlayer,
                "exterior_preparation": (
                    "skipped_prepared_frame"
                    if meas_slab_only_effective and frame_init_prepared
                    else "skipped_prepared_covariance"
                    if meas_slab_only_effective and G_init_prepared
                    else
                    "forced_occupied_onsite_before_cycle_0"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned_onsite_before_cycle_0"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_mode": (
                    "already_prepared"
                    if meas_slab_only_effective and initial_state_prepared
                    else
                    "forced_occupied"
                    if meas_slab_only_effective and effective_postselect
                    else "born_conditioned"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_performed": bool(
                    meas_slab_only_effective and not initial_state_prepared
                ),
                "exterior_preparation_basis": (
                    "canonical_unit_cell_orbital"
                    if meas_slab_only_effective
                    else None
                ),
                "exterior_preparation_orbital_count": (
                    2 * len(self._exterior_site_ids())
                    if meas_slab_only_effective
                    else 0
                ),
                "site_update_batching": site_update_batching,
                "state_representation_requested": state_representation_requested,
                "state_representation": state_representation_resolved,
                "state_representation_resolved": state_representation_resolved,
                "state_representation_resolution_reason": resolution_reason,
                "initial_purity_tolerance": float(initial_purity_tolerance),
                "initial_purity_defects": initial_purity_defects,
                "explicit_covariance_override": explicit_covariance_override,
                "frame_algorithm_version": (
                    GPU_FRAME_ALGORITHM_VERSION if frame_native else None
                ),
                "frame_diagnostics": frame_diagnostics,
                "frame_init_prepared": frame_init_prepared,
                "G_init_prepared": G_init_prepared,
                "covariance_materialization_count": len(
                    covariance_materializations
                ),
                "covariance_materializations": list(covariance_materializations),
                "require_no_covariance_materialization": bool(
                    require_no_covariance_materialization
                ),
                "native_cycle_observer": native_cycle_observer is not None,
                "return_native_state": bool(return_native_state),
            }
            if save and not return_data:
                return {
                    "samples": samples,
                    "batch_size": batch_size,
                    "batch_size_mode": batch_size_mode,
                    "batch_size_auto_info": batch_size_auto_info,
                    "T": len(snapshot_cycles_norm) if snapshot_cycles_norm is not None else (None if G_history else (cycles + 1)),
                    "snapshot_cycles": snapshot_cycles_norm,
                    "save_path": run_dir,
                    "run_dir": run_dir,
                    "manifest_path": manifest_path,
                    "merged_save_path": None,
                    "postselect_probability": float(postselect_probability),
                    **choi_result_metadata,
                }

            if (not save) and (not return_data):
                return {
                    "samples": samples,
                    "batch_size": batch_size,
                    "batch_size_mode": batch_size_mode,
                    "batch_size_auto_info": batch_size_auto_info,
                    "T": len(snapshot_cycles_norm) if snapshot_cycles_norm is not None else (None if G_history else (cycles + 1)),
                    "snapshot_cycles": snapshot_cycles_norm,
                    "save_path": None,
                    "run_dir": None,
                    "manifest_path": None,
                    "merged_save_path": None,
                    "postselect_probability": float(postselect_probability),
                    **choi_result_metadata,
                }

            if snapshot_cycles_norm is not None:
                if save:
                    shards = []
                    for batch_idx in range(num_batches):
                        shard_path = os.path.join(run_dir, shard_records[batch_idx]["filename"])
                        shards.append(np.asarray(np.load(shard_path, mmap_mode=None, allow_pickle=False), dtype=self.numpy_dtype))
                    G_snapshots = np.concatenate(shards, axis=0)
                    if merge_on_finish:
                        merged_name = "merged_snapshots.npy" if not suffix_tag else f"merged_snapshots_{suffix_tag}.npy"
                        merged_save_path = os.path.join(run_dir, merged_name)
                        self._save_npy_atomic(merged_save_path, G_snapshots)
                else:
                    G_snapshots = np.concatenate(snapshots_cpu, axis=0)
                return {
                    "G_snapshots": G_snapshots,
                    "G_snapshots_avg": np.mean(G_snapshots, axis=0),
                    "samples": samples,
                    "batch_size": batch_size,
                    "batch_size_mode": batch_size_mode,
                    "batch_size_auto_info": batch_size_auto_info,
                    "T": G_snapshots.shape[1],
                    "snapshot_cycles": snapshot_cycles_norm,
                    "save_path": merged_save_path if merged_save_path is not None else run_dir,
                    "run_dir": run_dir,
                    "manifest_path": manifest_path,
                    "merged_save_path": merged_save_path,
                    "postselect_probability": float(postselect_probability),
                    **choi_result_metadata,
                }

            if G_history:
                if save:
                    shards = []
                    for batch_idx in range(num_batches):
                        shard_path = os.path.join(run_dir, shard_records[batch_idx]["filename"])
                        shards.append(np.asarray(np.load(shard_path, mmap_mode=None, allow_pickle=False), dtype=self.numpy_dtype))
                    G_hist = np.concatenate(shards, axis=0)
                    if merge_on_finish:
                        merged_name = "merged_history.npy" if not suffix_tag else f"merged_history_{suffix_tag}.npy"
                        merged_save_path = os.path.join(run_dir, merged_name)
                        self._save_npy_atomic(merged_save_path, G_hist)
                else:
                    G_hist = np.concatenate(histories_cpu, axis=0)
                return {
                    "G_hist": G_hist,
                    "G_hist_avg": np.mean(G_hist, axis=0),
                    "samples": samples,
                    "batch_size": batch_size,
                    "batch_size_mode": batch_size_mode,
                    "batch_size_auto_info": batch_size_auto_info,
                    "T": G_hist.shape[1],
                    "snapshot_cycles": None,
                    "save_path": merged_save_path if merged_save_path is not None else run_dir,
                    "run_dir": run_dir,
                    "manifest_path": manifest_path,
                    "merged_save_path": merged_save_path,
                    "postselect_probability": float(postselect_probability),
                    **choi_result_metadata,
                }

            if save:
                shards = []
                for batch_idx in range(num_batches):
                    shard_path = os.path.join(run_dir, shard_records[batch_idx]["filename"])
                    shards.append(np.asarray(np.load(shard_path, mmap_mode=None, allow_pickle=False), dtype=self.numpy_dtype))
                G_final = np.concatenate(shards, axis=0)
                if merge_on_finish:
                    merged_name = "merged_final.npy" if not suffix_tag else f"merged_final_{suffix_tag}.npy"
                    merged_save_path = os.path.join(run_dir, merged_name)
                    self._save_npy_atomic(merged_save_path, G_final)
            elif return_native_state:
                return {
                    "native_final": (
                        finals_cpu[0]
                        if len(finals_cpu) == 1
                        else finals_cpu
                    ),
                    "samples": samples,
                    "batch_size": batch_size,
                    "batch_size_mode": batch_size_mode,
                    "batch_size_auto_info": batch_size_auto_info,
                    "T": cycles + 1,
                    "snapshot_cycles": None,
                    "save_path": None,
                    "run_dir": None,
                    "manifest_path": None,
                    "merged_save_path": None,
                    "postselect_probability": float(postselect_probability),
                    **choi_result_metadata,
                }
            else:
                G_final = np.concatenate(finals_cpu, axis=0)
            return {
                "G_final": G_final,
                "G_final_avg": np.mean(G_final, axis=0),
                "covariance_spectral_clip": bool(covariance_spectral_clip),
                "covariance_clip_max_correction": float(covariance_clip_max_correction),
                "samples": samples,
                "batch_size": batch_size,
                "batch_size_mode": batch_size_mode,
                "batch_size_auto_info": batch_size_auto_info,
                "T": cycles + 1,
                "snapshot_cycles": None,
                "save_path": merged_save_path if merged_save_path is not None else run_dir,
                "run_dir": run_dir,
                "manifest_path": manifest_path,
                "merged_save_path": merged_save_path,
                "postselect_probability": float(postselect_probability),
                **choi_result_metadata,
            }
