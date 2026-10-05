# Final-production Drive recovery, 2026-09-01

This directory is the read-only local recovery copy made immediately before
quarantining the latest final-production bundle designs. Files retain their
Google Drive folder layout and original bytes; archives were not unpacked or
rewritten.

## Preserved collections

| Collection | Drive folder ID | Files | Bytes | Preserved result |
|---|---|---:|---:|---|
| `production_25sample_p1_chern_v4` | `1VJvd0PN1rULvWtnX69KNaI_bbyL0ZSGO` | 8 | 249,512 | Completed A100 L64 qualification archive, receipt, preflight, logs, and stale lease evidence; no production shard archive |
| `production_25sample_h1_endpoint_packet_v4` | `16vXdo9-TVM7i3Hj7vXxGBBzvgtEQL9U4` | 13 | 81,523,041 | Completed A100 qualification archive, receipt, status, preflight, migration ledger, and logs |
| `production_25sample_h1_endpoint_packet_v3` | `1R0TfqTZQii5vdERkWLSoxNC1VKLXvHcE` | 37 | 951,223,754 | Twelve archive/receipt pairs, eight receipt-only entries, preflight, and logs |

Total: 58 files and 1,032,996,307 bytes.

## Validation

- Every downloaded file matched the server-reported byte count before it was
  promoted from its temporary download name.
- All 14 `tar.gz` files passed gzip and tar structure validation.
- All 14 archive/receipt pairs matched their recorded SHA-256 values.
- The eight H1-v3 receipt-only entries are preserved as evidence and are not
  represented as completed archives.
- `CHECKSUMS.sha256` records the SHA-256 digest of every recovered file.

The Google Drive source parent is `classA_final_production_outputs`, folder ID
`1jieMw9hCLonr208TYMnojD0YmrHJ1wpm`. The source collections remain on Drive;
this recovery did not delete or rename them.
