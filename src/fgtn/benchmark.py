import gc
import time

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency guard
    torch = None


def autotune_batch_size(
    model,
    *,
    candidates,
    cycles=1,
    postselect=False,
    postselect_probability=0.0,
    perfect_correction=False,
    sequence='raster_y',
    init_mode='default',
    n_a=0.5,
    max_memory_fraction=0.80,
):
    if torch is None:
        raise ImportError('autotune_batch_size requires PyTorch in the active environment.')

    results = []
    best = None
    best_fallback = None
    total_memory = None
    if torch.cuda.is_available():
        total_memory = float(torch.cuda.get_device_properties(model.device).total_memory)

    for batch_size in candidates:
        effective_postselect_probability = 1.0 if bool(postselect) else float(postselect_probability)
        trial_samples = 1 if effective_postselect_probability == 1.0 else int(batch_size)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(model.device)
            t0 = time.perf_counter()
            model.run_markov_circuit(
                G_history=False,
                progress=False,
                cycles=int(cycles),
                postselect=bool(postselect),
                postselect_probability=float(postselect_probability),
                perfect_correction=bool(perfect_correction),
                samples=int(trial_samples),
                init_mode=init_mode,
                save=False,
                save_init=False,
                n_a=float(n_a),
                sequence=sequence,
                batch_size=int(batch_size),
                return_data=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            samples_per_sec = trial_samples / max(elapsed, 1e-12)
            peak_alloc = None
            peak_reserved = None
            peak_fraction = None
            if torch.cuda.is_available():
                peak_alloc = float(torch.cuda.max_memory_allocated(model.device))
                peak_reserved = float(torch.cuda.max_memory_reserved(model.device))
                if total_memory and total_memory > 0:
                    peak_fraction = max(peak_alloc, peak_reserved) / total_memory
            within_headroom = (
                True if peak_fraction is None else (peak_fraction <= float(max_memory_fraction))
            )
            rec = {
                'batch_size': int(batch_size),
                'trial_samples': int(trial_samples),
                'elapsed_sec': float(elapsed),
                'samples_per_sec': float(samples_per_sec),
                'peak_memory_allocated_bytes': peak_alloc,
                'peak_memory_reserved_bytes': peak_reserved,
                'peak_memory_fraction': peak_fraction,
                'max_memory_fraction': float(max_memory_fraction),
                'status': 'ok' if within_headroom else 'headroom_exceeded',
            }
            results.append(rec)
            if best_fallback is None or rec['samples_per_sec'] > best_fallback['samples_per_sec']:
                best_fallback = rec
            if within_headroom and (best is None or rec['samples_per_sec'] > best['samples_per_sec']):
                best = rec
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'out of memory' not in msg and 'cuda error' not in msg:
                raise
            results.append({
                'batch_size': int(batch_size),
                'trial_samples': int(trial_samples),
                'status': 'oom',
                'error': str(exc),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            gc.collect()

    if best is None:
        if best_fallback is None:
            raise RuntimeError('All candidate batch sizes failed during autotuning.')
        best = best_fallback

    return {
        'best_batch_size': int(best['batch_size']),
        'results': results,
    }
