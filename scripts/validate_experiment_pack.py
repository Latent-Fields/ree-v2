#!/usr/bin/env python3
"""Validate REE experiment packs against pinned schemas and contract locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import evaluate_failure_signatures, get_profile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_run_dirs(runs_root: Path):
    for experiment_dir in sorted(runs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        runs_dir = experiment_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                yield experiment_dir.name, run_dir.name, run_dir


def format_schema_errors(validator: Draft202012Validator, payload: Any, label: str) -> list[str]:
    errors = []
    for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        path = ".".join(str(item) for item in error.path) or "$"
        errors.append(f"{label}: {error.message} (at {path})")
    return errors


def validate_lock_file(lock_file: Path) -> list[str]:
    issues: list[str] = []
    if not lock_file.exists():
        return [f"Missing contract lock file: {lock_file}"]

    lock_data = load_json(lock_file)
    artifacts = lock_data.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        return [f"{lock_file}: lock file has no artifacts list"]

    for artifact in artifacts:
        local_path = REPO_ROOT / artifact["local_path"]
        expected_hash = artifact["sha256"]
        if not local_path.exists():
            issues.append(f"{lock_file}: missing locked file {local_path}")
            continue
        observed_hash = sha256_file(local_path)
        if observed_hash != expected_hash:
            issues.append(
                f"{lock_file}: hash mismatch for {artifact['local_path']} "
                f"expected={expected_hash} observed={observed_hash}"
            )

    return issues


def validate_run(
    experiment_type: str,
    run_id: str,
    run_dir: Path,
    manifest_validator: Draft202012Validator,
    metrics_validator: Draft202012Validator,
    adapter_validator: Draft202012Validator,
) -> list[str]:
    issues: list[str] = []

    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"

    if not manifest_path.exists():
        return [f"{run_dir}: missing manifest.json"]
    if not metrics_path.exists():
        issues.append(f"{run_dir}: missing metrics.json")
    if not summary_path.exists():
        issues.append(f"{run_dir}: missing summary.md")

    manifest = load_json(manifest_path)
    issues.extend(format_schema_errors(manifest_validator, manifest, str(manifest_path)))

    if manifest.get("experiment_type") != experiment_type:
        issues.append(
            f"{manifest_path}: experiment_type mismatch dir={experiment_type} payload={manifest.get('experiment_type')}"
        )
    if manifest.get("run_id") != run_id:
        issues.append(f"{manifest_path}: run_id mismatch dir={run_id} payload={manifest.get('run_id')}")

    linkage_fields = ("claim_ids_tested", "evidence_class", "evidence_direction")
    for key in linkage_fields:
        if key not in manifest:
            issues.append(f"{manifest_path}: missing required linkage field '{key}'")

    scenario = manifest.get("scenario", {})
    provenance_fields = ("jepa_source_mode", "jepa_source_commit", "jepa_patch_set_hash")
    for key in provenance_fields:
        if key not in scenario:
            issues.append(f"{manifest_path}: scenario missing JEPA provenance field '{key}'")

    status = str(manifest.get("status", ""))
    failure_signatures = manifest.get("failure_signatures", [])
    if status == "PASS" and failure_signatures:
        issues.append(f"{manifest_path}: status=PASS is invalid when failure_signatures is non-empty")
    if experiment_type == "commit_dual_error_channels" and status == "PASS":
        mech060_signatures = [sig for sig in failure_signatures if isinstance(sig, str) and sig.startswith("mech060:")]
        if mech060_signatures:
            issues.append(
                f"{manifest_path}: commit_dual_error_channels status=PASS cannot include mech060 signatures {sorted(mech060_signatures)}"
            )

    if metrics_path.exists():
        metrics = load_json(metrics_path)
        issues.extend(format_schema_errors(metrics_validator, metrics, str(metrics_path)))

        values = metrics.get("values", {})
        try:
            profile = get_profile(experiment_type)
        except KeyError as exc:
            issues.append(f"{run_dir}: {exc}")
        else:
            for metric_key in profile.required_metric_keys:
                if metric_key not in values:
                    issues.append(f"{metrics_path}: missing required metric '{metric_key}'")

            expected_signatures = set(evaluate_failure_signatures(experiment_type, values))
            emitted_signatures = set(manifest.get("failure_signatures", []))
            missing = expected_signatures - emitted_signatures
            if missing:
                issues.append(
                    f"{manifest_path}: missing triggered failure signatures {sorted(missing)}"
                )

    artifacts = manifest.get("artifacts", {})
    adapter_rel_path = artifacts.get("adapter_signals_path")
    if adapter_rel_path:
        adapter_path = run_dir / adapter_rel_path
        if not adapter_path.exists():
            issues.append(
                f"{manifest_path}: adapter_signals_path points to missing file ({adapter_rel_path}) "
                "signature=contract:jepa_adapter_signals_missing"
            )
        else:
            adapter_payload = load_json(adapter_path)
            adapter_errors = format_schema_errors(adapter_validator, adapter_payload, str(adapter_path))
            if adapter_errors:
                for error in adapter_errors:
                    issues.append(f"{error} signature=contract:jepa_adapter_signals_invalid")
            if adapter_payload.get("schema_version") != "jepa_adapter_signals/v1":
                issues.append(
                    f"{adapter_path}: schema_version must be jepa_adapter_signals/v1 "
                    "signature=contract:jepa_adapter_signals_version"
                )
            if adapter_payload.get("experiment_type") != experiment_type:
                issues.append(
                    f"{adapter_path}: experiment_type mismatch dir={experiment_type} payload={adapter_payload.get('experiment_type')}"
                )
            if adapter_payload.get("run_id") != run_id:
                issues.append(f"{adapter_path}: run_id mismatch dir={run_id} payload={adapter_payload.get('run_id')}")

    if experiment_type == "commit_dual_error_channels":
        traces_dir_rel = artifacts.get("traces_dir")
        if not traces_dir_rel:
            issues.append(f"{manifest_path}: commit_dual_error_channels runs must declare artifacts.traces_dir")
        else:
            trace_path = run_dir / traces_dir_rel / "channel_isolation.v1.json"
            if not trace_path.exists():
                issues.append(f"{manifest_path}: missing channel isolation trace artifact ({trace_path})")
            else:
                trace_payload = load_json(trace_path)
                required_trace_keys = {
                    "corr_pre_post",
                    "corr_pre_realized",
                    "corr_post_realized",
                    "coupling_mean",
                    "pre_noise_std",
                }
                missing_trace_keys = sorted(required_trace_keys - set(trace_payload.keys()))
                if missing_trace_keys:
                    issues.append(
                        f"{trace_path}: missing required key(s) {missing_trace_keys}"
                    )

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path, help="Root path: evidence/experiments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "manifest.schema.json")
    metrics_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "metrics.schema.json")
    adapter_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "jepa_adapter_signals.v1.json")

    manifest_validator = Draft202012Validator(manifest_schema, format_checker=FormatChecker())
    metrics_validator = Draft202012Validator(metrics_schema, format_checker=FormatChecker())
    adapter_validator = Draft202012Validator(adapter_schema, format_checker=FormatChecker())

    issues: list[str] = []
    run_count = 0
    for experiment_type, run_id, run_dir in iter_run_dirs(args.runs_root):
        run_count += 1
        issues.extend(
            validate_run(
                experiment_type,
                run_id,
                run_dir,
                manifest_validator,
                metrics_validator,
                adapter_validator,
            )
        )

    issues.extend(validate_lock_file(REPO_ROOT / "contracts" / "ree_assembly_contract_lock.v1.json"))

    if issues:
        print(f"FAIL: validation found {len(issues)} issue(s) across {run_count} run(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(f"PASS: validated {run_count} run(s), schemas, adapter files, and contract lock hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
