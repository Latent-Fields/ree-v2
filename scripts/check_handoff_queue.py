#!/usr/bin/env python3
"""Validate inbound handoff bundles and report runnable vs blocked items."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {"manifest.json", "metrics.json", "summary.md"}
CONTRACT_LOCK_PATH = REPO_ROOT / "contracts" / "ree_assembly_contract_lock.v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-dir", type=Path, default=REPO_ROOT / "jobs" / "incoming")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "evidence" / "experiments")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return FAIL when any discovered bundle is blocked",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_bundles(incoming_dir: Path) -> list[Path]:
    bundles: list[Path] = []
    if not incoming_dir.exists():
        return bundles
    for candidate in sorted(incoming_dir.rglob("*")):
        if not candidate.is_dir():
            continue
        names = {p.name for p in candidate.iterdir() if p.is_file()}
        if EXPECTED_FILES.issubset(names):
            bundles.append(candidate)
    return bundles


def parse_target(bundle_dir: Path) -> tuple[str, str] | None:
    parts = bundle_dir.name.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def read_contract_attestation(bundle_dir: Path) -> str | None:
    # Prefer explicit metadata if present.
    for meta_name in ("handoff_meta.json", "provenance.json", "bundle_meta.json", "job_spec.json"):
        path = bundle_dir / meta_name
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            for key in ("contract_lock_sha256", "contract_lock_hash"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value

    # Fallback to manifest-level field if an upstream exporter included it.
    manifest = load_json(bundle_dir / "manifest.json")
    if isinstance(manifest, dict):
        scenario = manifest.get("scenario", {})
        if isinstance(scenario, dict):
            value = scenario.get("contract_lock_sha256")
            if isinstance(value, str) and value:
                return value
    return None


def validate_bundle(bundle_dir: Path, expected_contract_hash: str, runs_root: Path) -> tuple[str, list[str], str, str]:
    reasons: list[str] = []
    target = parse_target(bundle_dir)
    experiment_type = "unknown"
    run_id = "unknown"

    if target is None:
        reasons.append("bundle_name_must_be_<experiment_type>__<run_id>")
    else:
        experiment_type, run_id = target

    manifest = load_json(bundle_dir / "manifest.json")
    if not isinstance(manifest, dict):
        reasons.append("manifest_not_an_object")
        return "BLOCKED", reasons, experiment_type, run_id

    if target is not None:
        if manifest.get("experiment_type") != experiment_type:
            reasons.append("manifest_experiment_type_mismatch")
        if manifest.get("run_id") != run_id:
            reasons.append("manifest_run_id_mismatch")

    attested_contract_hash = read_contract_attestation(bundle_dir)
    if not attested_contract_hash:
        reasons.append("missing_contract_lock_attestation")
    elif attested_contract_hash != expected_contract_hash:
        reasons.append("contract_lock_hash_mismatch")

    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        traces_dir = artifacts.get("traces_dir")
        if manifest.get("experiment_type") == "commit_dual_error_channels":
            if not isinstance(traces_dir, str):
                reasons.append("missing_artifacts.traces_dir_for_mech060")
            else:
                trace_file = bundle_dir / traces_dir / "channel_isolation.v1.json"
                if not trace_file.exists():
                    reasons.append("missing_mech060_channel_isolation_trace")

    if target is not None:
        destination = runs_root / experiment_type / "runs" / run_id
        if destination.exists():
            reasons.append("run_already_exists_in_runs_root")

    status = "RUNNABLE" if not reasons else "BLOCKED"
    return status, reasons, experiment_type, run_id


def main() -> int:
    args = parse_args()
    expected_contract_hash = sha256_file(CONTRACT_LOCK_PATH)
    bundles = discover_bundles(args.incoming_dir)
    if not bundles:
        print(f"PASS: no inbound handoff bundles in {args.incoming_dir}")
        return 0

    runnable = 0
    blocked = 0
    print("bundle\tstatus\texperiment_type\trun_id\treasons")
    for bundle in bundles:
        status, reasons, experiment_type, run_id = validate_bundle(
            bundle_dir=bundle,
            expected_contract_hash=expected_contract_hash,
            runs_root=args.runs_root,
        )
        reason_text = "none" if not reasons else ",".join(reasons)
        print(f"{bundle}\t{status}\t{experiment_type}\t{run_id}\t{reason_text}")
        if status == "RUNNABLE":
            runnable += 1
        else:
            blocked += 1

    summary = (
        f"PASS: checked inbound queue bundles={len(bundles)} runnable={runnable} blocked={blocked}"
        if not (args.strict and blocked > 0)
        else f"FAIL: checked inbound queue bundles={len(bundles)} runnable={runnable} blocked={blocked}"
    )
    print(summary)
    if args.strict and blocked > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
