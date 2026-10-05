# Archive manifests

Each JSON file in this directory is a file-level SHA-256 inventory produced by
`scripts/archive_cold_data.py`.  A source manifest proves what was intended for archival;
it does not prove that an external copy exists.  Only a separate receipt with
`status="verified"` establishes that every destination file was independently read and
matched.

The archive tool never deletes source data and writes `source_deletion_authorized=false`
in every verification receipt.  Deletion remains a separate user-approved action.

## Completed source manifests (2026-08-17)

All nine cold-data source inventories are complete: 4,263 files and
850,138,800,945 bytes (791.75 GiB).  The SHA-256 values below identify the manifest
files themselves; each JSON manifest contains the per-file hashes for its dataset.

```text
5037a05c4fc705addb7429c1854d2fb3ffb21c67c349e04bbc218dc05ecefab1  cache_G_history_samples.json
a6b2eb482c6829d2cf9f5164f529fb3f66cc2cad4ae86967d8d669b663fce70e  choi_covariance_cpu_data.json
d73397e8620a4371b6197469d6384a0cac513cbbd45149ebeb3a9df941de0211  colab_charge_fluctuations_cpu_data.json
0a7a5fb588004a25c60c88e438984f0109d02712a3348f08f24f24a872af198b  colab_charge_fluctuations_gpu_data.json
cfb06e9f0642fe6c12846ed5549f89c688526b84549e76c33f78ad413ae17cd9  colab_small_system_testing_analysis_outputs.json
a1240ef82a223556d6d1e4d04d960cab0a213012914035f686001b59984dc87e  colab_small_system_testing_gpu_data.json
0aef60d12c1915d9ff14f10a2fd9f74a336139419c78f6577ec3943dc0b39499  dw_convergence.json
8af514a7df6f5c9bc8930544400dbdee4f6bac67fd3de8235ff860e5431f36cd  experiments.json
7fc62351fdf51495fb5279da800a2b9b3a6809a9309639166e29dd44390745f0  lyapunov_analysis_v2_cache.json
```

`scripts/archive_cold_data.py` refuses to overwrite manifests, receipts, partial
copies, or destination files.
