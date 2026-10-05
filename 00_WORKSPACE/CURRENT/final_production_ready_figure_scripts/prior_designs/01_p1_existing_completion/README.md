# Exact completion of the existing P1 campaign

This folder is a frozen copy of the numerical source used by the transferred
`production_25sample_v1` P1 archives.  It exists only to verify and finish the original
240-shard P1 matrix in `MyDrive/classA_final_production_outputs/01_bulk_width_gate`.

Open `finish_existing_P1.ipynb` in an A100 Colab runtime.  The notebook defaults to a
read-only receipt/checksum report.  After it reports the expected verified and pending
counts, set `RUN_MISSING_SHARDS=True` and rerun the launch cell.  The runner refuses a
changed audit, engine, sample count, malformed receipt, checksum failure, duplicate slot,
or case mismatch.  It never enumerates or launches W1.

Do not modify this frozen folder when changing the new lean campaign.  Its audit identity
is deliberately the original `d23d...` 25-trajectory production identity.
