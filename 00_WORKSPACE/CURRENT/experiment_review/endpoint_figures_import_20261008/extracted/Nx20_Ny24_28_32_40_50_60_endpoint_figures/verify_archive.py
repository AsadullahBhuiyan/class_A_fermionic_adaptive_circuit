"""Verify every archived payload against SHA256SUMS.json."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root/'SHA256SUMS.json').read_text())
for name, expected in records.items():
    path = root/name
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise SystemExit(f'Missing file or size mismatch: {name}')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected['sha256']:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f'Verified {len(records)} packaged files.')
