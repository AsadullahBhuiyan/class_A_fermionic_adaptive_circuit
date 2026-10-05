#!/usr/bin/env python3
"""Build the exact independent-job handoff for the frame-native v4 campaign."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "_shared_src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from bundle_layout import bundle_path, bundle_relative_path  # noqa: E402
from b1_controller_frame import b1_cases  # noqa: E402
from campaign_cases import expand_cases  # noqa: E402
from production_runtime import (  # noqa: E402
    AUDIT_SHA256,
    PRODUCTION_OUTPUT_COLLECTION,
    SAMPLING_REVISION,
)


ACTIVE = (
    "01_bulk_width_gate",
    "02_pure_wall_master",
    "03_chirality_replay",
    "04_maxmix_master",
    "05_scans_and_controls",
    "06_b1_controller_frame",
)


def load_config(bundle: str) -> dict:
    return json.loads(
        (bundle_path(ROOT, bundle) / "production_config.json").read_text(
            encoding="utf-8"
        )
    )


def cases_for(bundle: str) -> list[dict]:
    config = load_config(bundle)
    if bundle == "06_b1_controller_frame":
        return b1_cases(config)
    return expand_cases(config, m3_wall_sigma=None)


def main() -> None:
    jobs = []
    ordinal = 0
    for bundle in ACTIVE:
        for case in cases_for(bundle):
            shard_count = 1 if case.get("kind") == "deterministic_descendant" else 2
            for shard in range(shard_count):
                ordinal += 1
                command = (
                    f"python -u {bundle_relative_path(bundle)}/run_bundle.py --mode production "
                    f"--drive-root /shared/drive --case-id {case['case_id']} "
                    f"--shard-index {shard}"
                )
                jobs.append(
                    {
                        "ordinal": ordinal,
                        "bundle": bundle,
                        "output_bundle": load_config(bundle).get(
                            "output_bundle", bundle
                        ),
                        "campaign": case.get("campaign", "B1"),
                        "case_id": case["case_id"],
                        "shard_index": shard,
                        "global_sample_indices": (
                            [0] if case.get("kind") == "deterministic_descendant"
                            else list(range(5 * shard, 5 * shard + 5))
                        ),
                        "command": command,
                    }
                )
    totals = Counter(job["bundle"] for job in jobs)
    payload = {
        "schema_version": 1,
        "status": "ready_base_queue_before_m3_wall_expansion",
        "sampling_revision": SAMPLING_REVISION,
        "audit_sha256": AUDIT_SHA256,
        "production_output_collection": PRODUCTION_OUTPUT_COLLECTION,
        "bundle_output_overrides": {
            bundle: load_config(bundle).get("output_bundle", bundle)
            for bundle in ACTIVE
            if load_config(bundle).get("output_bundle", bundle) != bundle
        },
        "fixed_geometry": {"Nx": 20, "Ny_scaling": [20, 30, 40, 50, 60]},
        "base_job_count_including_two_h3_descendants": len(jobs),
        "base_ordinary_stochastic_shards": len(jobs) - 2,
        "dynamic_m3_wall_rule": {
            "after_gate": True,
            "protocols": 2,
            "Ny": [20, 30, 40, 50, 60],
            "noise_values": "the exact 3--5 values in m3_bulk_gate.json:wall_sigma_bracket",
            "shards_per_case": 2,
            "additional_shards": {"minimum": 60, "maximum": 100},
        },
        "final_ordinary_stochastic_shards": {"minimum": 458, "maximum": 498},
        "per_bundle_base_jobs": dict(sorted(totals.items())),
        "rules": [
            "Each listed case/shard pair is an independent array job.",
            "Run one report-only resume scan before dispatching the array.",
            "Do not schedule H3 until both declared 20x40 parent shard-zero archives verify.",
            "Generate M3 wall jobs only from the accepted gate JSON; never guess the bracket.",
            "A prior shard is reusable only when receipt, checksum, engine, case, seed, audit compatibility, shard index, and global sample indices all match.",
        ],
        "jobs": jobs,
    }
    output = ROOT / "collaborator_handoff_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[built] {output.name}: {len(jobs)} base jobs")


if __name__ == "__main__":
    main()
