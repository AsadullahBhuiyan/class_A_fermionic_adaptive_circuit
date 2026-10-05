#!/usr/bin/env python3
"""Verify completed Drive campaign files after they are copied into the repo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "PROJECT_ADMIN"
    / "drive_import_manifests"
    / "completed_campaigns_20260908.remote.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_join(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    return candidate


def completion_result_fields(payload: dict[str, Any]) -> tuple[str | None, int | None, str | None]:
    nested = payload.get("result")
    if not isinstance(nested, dict):
        nested = {}
    name = payload.get("result_filename", nested.get("name"))
    size = payload.get("result_bytes", nested.get("bytes"))
    digest = payload.get("result_sha256", nested.get("sha256"))
    return (
        name if isinstance(name, str) else None,
        int(size) if isinstance(size, int) else None,
        digest if isinstance(digest, str) else None,
    )


def verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> str | None:
    if not path.is_file():
        return "missing"
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        return f"bytes:{actual_bytes}!={expected_bytes}"
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        return f"sha256:{actual_sha256}!={expected_sha256}"
    return None


def verify_campaign(campaign: dict[str, Any]) -> dict[str, Any]:
    destination = safe_join(REPO_ROOT, campaign["canonical_local_destination"])
    problems: list[str] = []
    verified_pairs = 0
    verified_extras = 0

    for record in campaign["records"]:
        completion = safe_join(destination, record["relative_completion_path"])
        result = safe_join(destination, record["relative_result_path"])

        completion_problem = verify_file(
            completion,
            expected_bytes=int(record["completion_bytes"]),
            expected_sha256=record["completion_sha256"],
        )
        if completion_problem is not None:
            problems.append(f"{completion.relative_to(REPO_ROOT)}: {completion_problem}")
            continue

        result_problem = verify_file(
            result,
            expected_bytes=int(record["result_bytes"]),
            expected_sha256=record["result_sha256"],
        )
        if result_problem is not None:
            problems.append(f"{result.relative_to(REPO_ROOT)}: {result_problem}")
            continue

        try:
            payload = json.loads(completion.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{completion.relative_to(REPO_ROOT)}: invalid JSON ({exc})")
            continue
        name, size, digest = completion_result_fields(payload)
        expected_identity = (
            result.name,
            int(record["result_bytes"]),
            record["result_sha256"],
        )
        if (name, size, digest) != expected_identity:
            problems.append(
                f"{completion.relative_to(REPO_ROOT)}: result identity does not match inventory"
            )
            continue
        verified_pairs += 1

    for record in campaign.get("extra_json_files", []):
        path = safe_join(destination, record["relative_path"])
        problem = verify_file(
            path,
            expected_bytes=int(record["bytes"]),
            expected_sha256=record["sha256"],
        )
        if problem is not None:
            problems.append(f"{path.relative_to(REPO_ROOT)}: {problem}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{path.relative_to(REPO_ROOT)}: invalid JSON ({exc})")
            continue
        verified_extras += 1

    return {
        "campaign": campaign["campaign"],
        "destination": str(destination),
        "expected_pairs": int(campaign["result_completion_pairs"]),
        "verified_pairs": verified_pairs,
        "expected_extra_json": len(campaign.get("extra_json_files", [])),
        "verified_extra_json": verified_extras,
        "complete": not problems,
        "problem_count": len(problems),
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--campaign", action="append", help="Limit verification to this manifest campaign key")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report incomplete imports without returning a nonzero status",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.campaign or [])
    campaigns = [
        campaign
        for campaign in manifest["campaigns"]
        if not selected or campaign["campaign"] in selected
    ]
    unknown = selected - {campaign["campaign"] for campaign in campaigns}
    if unknown:
        parser.error(f"unknown campaign keys: {sorted(unknown)}")

    reports = [verify_campaign(campaign) for campaign in campaigns]
    summary = {
        "manifest": str(args.manifest.resolve()),
        "campaigns_checked": len(reports),
        "campaigns_complete": sum(report["complete"] for report in reports),
        "expected_pairs": sum(report["expected_pairs"] for report in reports),
        "verified_pairs": sum(report["verified_pairs"] for report in reports),
        "problem_count": sum(report["problem_count"] for report in reports),
        "campaigns": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.allow_missing or summary["problem_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
