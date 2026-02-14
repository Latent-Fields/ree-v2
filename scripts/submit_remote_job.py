#!/usr/bin/env python3
"""Submit remote job specs to backend (dry-run supported)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {
    "schema_version",
    "job_id",
    "experiment_type",
    "condition",
    "seeds",
    "config_hash",
    "source_commit",
    "contract_lock_sha256",
    "jepa_source_lock_sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    if not args.job_spec_dir.exists():
        print(f"FAIL: job spec directory does not exist: {args.job_spec_dir}")
        return 1

    spec_files = sorted(args.job_spec_dir.glob("*.json"))
    if not spec_files:
        print(f"PASS: no remote job specs found in {args.job_spec_dir}")
        return 0

    issues: list[str] = []
    for spec_path in spec_files:
        spec = load_json(spec_path)
        missing = sorted(REQUIRED_KEYS - set(spec.keys()))
        if missing:
            issues.append(f"{spec_path}: missing keys {missing}")
            continue
        if spec.get("schema_version") != "ree_remote_job_spec/v1":
            issues.append(f"{spec_path}: schema_version must be ree_remote_job_spec/v1")
            continue

        if args.dry_run:
            print(f"dry-run submit OK: {spec_path.name} -> backend=<not configured>")
        else:
            submitted_dir = REPO_ROOT / "jobs" / "submitted"
            submitted_dir.mkdir(parents=True, exist_ok=True)
            marker_path = submitted_dir / f"{spec['job_id']}.submitted.json"
            with marker_path.open("w", encoding="utf-8") as handle:
                json.dump({"status": "SUBMITTED", "job_spec": spec_path.name}, handle, indent=2)
                handle.write("\n")
            print(f"submitted: {spec_path.name}")

    if issues:
        print(f"FAIL: submit validation failed for {len(issues)} spec(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(f"PASS: validated {len(spec_files)} remote job spec(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
