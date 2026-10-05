# Max-mix many-body Lyapunov v2 data

This is the checksum-verified local copy of the completed v2 A100 campaign.
It belongs to the `04_maxmix_manybody_lyapunov_pilot` bundle that generated it.

## Contents

- `results/Ny20` through `results/Ny40`: 20 five-trajectory tasks per
  circumference, each stored as one NPZ and one completion JSON.
- `campaign_config.v2.json`: the exact locked configuration whose SHA-256 is
  recorded by every completion JSON.
- `DOWNLOAD_MANIFEST.json`: Drive provenance, source hashes, aggregate counts,
  and the post-download validation summary.

The data cover 160 tasks and 800 independent trajectories. Every NPZ byte count
and SHA-256 was checked against its paired completion JSON after download. The
source files in the owning bundle also match the source hashes embedded in all
160 completion records.

There was no outer archive to unpack on Drive. Each `.npz` is itself a standard
ZIP-based NumPy container. Those files are intentionally retained in their
original compressed form because the completion records bind their exact bytes
and the campaign analysis consumes them directly. Expanding them into `.npy`
files would create a redundant, noncanonical copy.

## Local analysis

From this bundle directory, run:

```bash
python analyze_campaign.py \
  --config gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/campaign_config.v2.json \
  --results-root gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2 \
  --output-root analysis_outputs/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2
```

The Drive source remains unchanged at
<https://drive.google.com/drive/folders/1D6DZ69-shgbjKo2cQY3EzLh1UEhBI_r9>.
