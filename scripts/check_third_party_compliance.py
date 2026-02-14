#!/usr/bin/env python3
"""Validate JEPA third-party provenance, notices, and attribution surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "third_party" / "jepa_sources.lock.v1.json"
NOTICES_PATH = REPO_ROOT / "third_party" / "THIRD_PARTY_NOTICES.md"
ATTRIBUTION_PATH = REPO_ROOT / "docs" / "third_party" / "JEPA_ATTRIBUTION_AND_CITATION.md"

REQUIRED_LOCK_FIELDS = [
    "source_mode",
    "upstream_repo_url",
    "upstream_commit",
    "upstream_license_id",
    "checkpoint_repo_id",
    "checkpoint_revision",
    "checkpoint_license_id",
    "checkpoint_filename",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "license_id",
    "patch_set",
    "compatibility_target",
    "last_verified_utc",
    "inference_mode",
    "optimizer_policy",
    "gradient_policy",
    "checkpoint_fallback_repo_id",
    "checkpoint_fallback_license_id",
    "checkpoint_fallback_filename",
    "checkpoint_fallback_sha256",
    "checkpoint_fallback_size_bytes",
]

ALLOWED_SOURCE_MODES = {"vendored_snapshot", "submodule_pin", "internal_minimal_impl"}
PLACEHOLDER_MARKERS = {"unknown", "placeholder", "reference-only", "todo", "tbd"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_hex_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    allowed = set("0123456789abcdef")
    return all(char in allowed for char in value.lower())


def parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def iter_manifest_paths() -> list[Path]:
    manifests: list[Path] = []
    runs_root = REPO_ROOT / "evidence" / "experiments"
    if not runs_root.exists():
        return manifests

    for manifest_path in sorted(runs_root.glob("*/runs/*/manifest.json")):
        manifests.append(manifest_path)
    return manifests


def main() -> int:
    _ = parse_args()

    issues: list[str] = []

    if not LOCK_PATH.exists():
        issues.append(f"missing lock file: {LOCK_PATH}")
    if not NOTICES_PATH.exists():
        issues.append(f"missing notices file: {NOTICES_PATH}")
    if not ATTRIBUTION_PATH.exists():
        issues.append(f"missing attribution file: {ATTRIBUTION_PATH}")

    if issues:
        print(f"FAIL: third-party compliance failed with {len(issues)} issue(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    lock = load_json(LOCK_PATH)
    notices_text = NOTICES_PATH.read_text(encoding="utf-8")
    attribution_text = ATTRIBUTION_PATH.read_text(encoding="utf-8")

    for field in REQUIRED_LOCK_FIELDS:
        if field not in lock:
            issues.append(f"lock missing required field '{field}'")

    source_mode = str(lock.get("source_mode", ""))
    if source_mode not in ALLOWED_SOURCE_MODES:
        issues.append(f"lock source_mode must be one of {sorted(ALLOWED_SOURCE_MODES)}; got '{source_mode}'")

    upstream_commit = str(lock.get("upstream_commit", ""))
    if not upstream_commit:
        issues.append("lock upstream_commit is empty")
    if has_placeholder(upstream_commit):
        issues.append(f"lock upstream_commit looks like placeholder: '{upstream_commit}'")
    if source_mode == "internal_minimal_impl" and not upstream_commit.startswith("internal_snapshot_sha256:"):
        issues.append("internal_minimal_impl requires upstream_commit to start with 'internal_snapshot_sha256:'")
    if source_mode == "submodule_pin" and len(upstream_commit) != 40:
        issues.append("submodule_pin requires a 40-char upstream_commit git SHA")

    checkpoint_repo_id = str(lock.get("checkpoint_repo_id", ""))
    checkpoint_revision = str(lock.get("checkpoint_revision", ""))
    checkpoint_license_id = str(lock.get("checkpoint_license_id", ""))
    checkpoint_filename = str(lock.get("checkpoint_filename", ""))
    checkpoint_sha256 = str(lock.get("checkpoint_sha256", ""))
    checkpoint_size_bytes = parse_positive_int(lock.get("checkpoint_size_bytes"))
    if not checkpoint_repo_id:
        issues.append("lock checkpoint_repo_id is empty")
    if not checkpoint_revision:
        issues.append("lock checkpoint_revision is empty")
    if has_placeholder(checkpoint_revision):
        issues.append(f"lock checkpoint_revision looks like placeholder: '{checkpoint_revision}'")
    if checkpoint_revision and len(checkpoint_revision) != 40:
        issues.append("checkpoint_revision should be a 40-char revision SHA")
    if not checkpoint_license_id:
        issues.append("lock checkpoint_license_id is empty")
    if not checkpoint_filename:
        issues.append("lock checkpoint_filename is empty")
    if not checkpoint_sha256:
        issues.append("lock checkpoint_sha256 is empty")
    elif not is_hex_sha256(checkpoint_sha256):
        issues.append("lock checkpoint_sha256 must be a 64-char lowercase hex digest")
    if checkpoint_size_bytes is None:
        issues.append("lock checkpoint_size_bytes must be a positive integer")

    fallback_repo_id = str(lock.get("checkpoint_fallback_repo_id", ""))
    fallback_license_id = str(lock.get("checkpoint_fallback_license_id", ""))
    fallback_filename = str(lock.get("checkpoint_fallback_filename", ""))
    fallback_sha256 = str(lock.get("checkpoint_fallback_sha256", ""))
    fallback_size_bytes = parse_positive_int(lock.get("checkpoint_fallback_size_bytes"))
    if not fallback_repo_id:
        issues.append("lock checkpoint_fallback_repo_id is empty")
    if not fallback_license_id:
        issues.append("lock checkpoint_fallback_license_id is empty")
    if not fallback_filename:
        issues.append("lock checkpoint_fallback_filename is empty")
    if not fallback_sha256:
        issues.append("lock checkpoint_fallback_sha256 is empty")
    elif not is_hex_sha256(fallback_sha256):
        issues.append("lock checkpoint_fallback_sha256 must be a 64-char lowercase hex digest")
    if fallback_size_bytes is None:
        issues.append("lock checkpoint_fallback_size_bytes must be a positive integer")

    if str(lock.get("compatibility_target", "")) != "IMPL-022":
        issues.append("lock compatibility_target must be IMPL-022")

    if not isinstance(lock.get("patch_set"), list):
        issues.append("lock patch_set must be a list")

    license_id = str(lock.get("license_id", ""))
    if license_id and license_id not in notices_text:
        issues.append("license_id from lock not found in THIRD_PARTY_NOTICES.md")
    if checkpoint_license_id and checkpoint_license_id not in notices_text:
        issues.append("checkpoint_license_id from lock not found in THIRD_PARTY_NOTICES.md")
    if str(lock.get("upstream_license_id", "")) and str(lock.get("upstream_license_id")) not in notices_text:
        issues.append("upstream_license_id from lock not found in THIRD_PARTY_NOTICES.md")

    if lock.get("inference_mode") != "inference_only_no_training":
        issues.append("lock inference_mode must be inference_only_no_training")
    if lock.get("optimizer_policy") != "forbidden":
        issues.append("lock optimizer_policy must be forbidden")
    if lock.get("gradient_policy") != "no_grad_only":
        issues.append("lock gradient_policy must be no_grad_only")

    upstream_repo_url = str(lock.get("upstream_repo_url", ""))
    if upstream_repo_url and upstream_repo_url not in notices_text and upstream_repo_url not in attribution_text:
        issues.append("upstream_repo_url not mentioned in notices or attribution docs")

    if upstream_commit and upstream_commit not in notices_text:
        issues.append("upstream_commit not mentioned in THIRD_PARTY_NOTICES.md")
    if checkpoint_repo_id and checkpoint_repo_id not in notices_text and checkpoint_repo_id not in attribution_text:
        issues.append("checkpoint_repo_id not mentioned in notices or attribution docs")
    if checkpoint_revision and checkpoint_revision not in notices_text and checkpoint_revision not in attribution_text:
        issues.append("checkpoint_revision not mentioned in notices or attribution docs")
    if checkpoint_filename and checkpoint_filename not in notices_text and checkpoint_filename not in attribution_text:
        issues.append("checkpoint_filename not mentioned in notices or attribution docs")

    citation_markers = [
        "arXiv:2301.08243",
        "arXiv:2506.09985",
        "arXiv:2412.10925",
        "I-JEPA",
        "V-JEPA 2",
    ]
    for marker in citation_markers:
        if marker not in attribution_text:
            issues.append(f"attribution doc missing citation marker '{marker}'")

    patch_hash = sha256_json(lock.get("patch_set", []))[:16]
    manifests = iter_manifest_paths()
    jepa_backed_runs = 0
    jepa_real_runs = 0
    jepa_fallback_runs = 0
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        scenario = manifest.get("scenario", {})
        backend = str(scenario.get("backend", ""))
        runner_version = str(manifest.get("runner", {}).get("version", ""))
        is_jepa_backed = backend == "jepa_inference" or "jepa_inference" in runner_version
        if not is_jepa_backed:
            continue

        jepa_backed_runs += 1
        fallback_used = bool(scenario.get("synthetic_frame_fallback", False))
        if fallback_used:
            jepa_fallback_runs += 1
        else:
            jepa_real_runs += 1

        if scenario.get("jepa_source_mode") != lock.get("source_mode"):
            issues.append(
                f"{manifest_path}: jepa_source_mode mismatch "
                f"manifest={scenario.get('jepa_source_mode')} lock={lock.get('source_mode')}"
            )
        if scenario.get("jepa_source_commit") != lock.get("upstream_commit"):
            issues.append(
                f"{manifest_path}: jepa_source_commit mismatch "
                f"manifest={scenario.get('jepa_source_commit')} lock={lock.get('upstream_commit')}"
            )
        if scenario.get("jepa_patch_set_hash") != patch_hash:
            issues.append(
                f"{manifest_path}: jepa_patch_set_hash mismatch "
                f"manifest={scenario.get('jepa_patch_set_hash')} expected={patch_hash}"
            )
        if scenario.get("jepa_checkpoint_repo_id") != checkpoint_repo_id:
            issues.append(
                f"{manifest_path}: jepa_checkpoint_repo_id mismatch "
                f"manifest={scenario.get('jepa_checkpoint_repo_id')} lock={checkpoint_repo_id}"
            )
        if scenario.get("jepa_checkpoint_revision") != checkpoint_revision:
            issues.append(
                f"{manifest_path}: jepa_checkpoint_revision mismatch "
                f"manifest={scenario.get('jepa_checkpoint_revision')} lock={checkpoint_revision}"
            )
        if scenario.get("jepa_checkpoint_license_id") != checkpoint_license_id:
            issues.append(
                f"{manifest_path}: jepa_checkpoint_license_id mismatch "
                f"manifest={scenario.get('jepa_checkpoint_license_id')} lock={checkpoint_license_id}"
            )
        scenario_filename = scenario.get("jepa_checkpoint_filename")
        if scenario_filename not in (None, "") and scenario_filename != checkpoint_filename:
            issues.append(
                f"{manifest_path}: jepa_checkpoint_filename mismatch "
                f"manifest={scenario_filename} lock={checkpoint_filename}"
            )
        if "jepa_patch_set_hash" not in scenario:
            issues.append(f"{manifest_path}: missing jepa_patch_set_hash on JEPA-backed run")
        if "jepa_source_commit" not in scenario:
            issues.append(f"{manifest_path}: missing jepa_source_commit on JEPA-backed run")
        if "jepa_source_mode" not in scenario:
            issues.append(f"{manifest_path}: missing jepa_source_mode on JEPA-backed run")
        if not fallback_used:
            required_real_fields = (
                "jepa_checkpoint_sha256",
                "jepa_checkpoint_size_bytes",
                "jepa_checkpoint_verified",
                "jepa_checkpoint_filename",
            )
            for field in required_real_fields:
                if field not in scenario:
                    issues.append(f"{manifest_path}: missing {field} on real JEPA run")

            if scenario.get("jepa_checkpoint_verified") is not True:
                issues.append(
                    f"{manifest_path}: jepa_checkpoint_verified must be true on real JEPA run"
                )
            if scenario.get("jepa_checkpoint_sha256") != checkpoint_sha256:
                issues.append(
                    f"{manifest_path}: jepa_checkpoint_sha256 mismatch "
                    f"manifest={scenario.get('jepa_checkpoint_sha256')} lock={checkpoint_sha256}"
                )
            scenario_size = parse_positive_int(scenario.get("jepa_checkpoint_size_bytes"))
            if checkpoint_size_bytes is None or scenario_size != checkpoint_size_bytes:
                issues.append(
                    f"{manifest_path}: jepa_checkpoint_size_bytes mismatch "
                    f"manifest={scenario.get('jepa_checkpoint_size_bytes')} lock={lock.get('checkpoint_size_bytes')}"
                )

    if issues:
        print(f"FAIL: third-party compliance failed with {len(issues)} issue(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(
        "PASS: third-party compliance verified "
        f"lock={LOCK_PATH} notices={NOTICES_PATH} attribution={ATTRIBUTION_PATH} "
        f"manifests={len(manifests)} jepa_backed_runs={jepa_backed_runs} "
        f"jepa_real_runs={jepa_real_runs} jepa_fallback_runs={jepa_fallback_runs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
