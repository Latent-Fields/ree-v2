#!/usr/bin/env python3
"""Build provider-agnostic remote job specs for profiles routed to remote."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import get_profiles
from ree_v2.experiments.resource_policy import decide_execution_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="all", help="Profile name or 'all'")
    parser.add_argument("--machine", default="macbook_air_m2_2022")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "jobs" / "outgoing")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def build_job_spec(
    *,
    experiment_type: str,
    condition_name: str,
    seeds: tuple[int, ...],
    config_hash: str,
    offload_reasons: list[str],
    source_commit: str,
    contract_lock_hash: str,
) -> dict[str, Any]:
    job_id = f"{experiment_type}__{condition_name}"
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "ree_remote_job_spec/v1",
        "job_id": job_id,
        "created_utc": created,
        "experiment_type": experiment_type,
        "condition": condition_name,
        "seeds": list(seeds),
        "config_hash": config_hash,
        "execution_mode": "remote",
        "offload_reasons": offload_reasons,
        "source_commit": source_commit,
        "contract_lock_sha256": contract_lock_hash,
        "results_destination": f"evidence/experiments/{experiment_type}/runs",
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiles = get_profiles(args.profile)
    source_commit = git_head()
    contract_lock_hash = sha256_file(REPO_ROOT / "contracts" / "ree_assembly_contract_lock.v1.json")

    count = 0
    for profile in profiles:
        for condition in profile.conditions:
            mode, reasons, _batch_runtime, _seed_count = decide_execution_mode(profile, condition, args.machine)
            if mode != "remote":
                continue

            config_hash = hashlib.sha256(
                f"{profile.experiment_type}:{condition.name}:v1".encode("utf-8")
            ).hexdigest()[:12]
            payload = build_job_spec(
                experiment_type=profile.experiment_type,
                condition_name=condition.name,
                seeds=profile.default_seeds,
                config_hash=config_hash,
                offload_reasons=reasons,
                source_commit=source_commit,
                contract_lock_hash=contract_lock_hash,
            )

            out_path = args.out_dir / f"{payload['job_id']}.json"
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            count += 1
            print(f"wrote {out_path}")

    print(f"PASS: generated {count} remote job spec(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
