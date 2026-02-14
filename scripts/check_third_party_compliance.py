#!/usr/bin/env python3
"""Validate JEPA third-party provenance, notices, and attribution surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "third_party" / "jepa_sources.lock.v1.json"
NOTICES_PATH = REPO_ROOT / "third_party" / "THIRD_PARTY_NOTICES.md"
ATTRIBUTION_PATH = REPO_ROOT / "docs" / "third_party" / "JEPA_ATTRIBUTION_AND_CITATION.md"

REQUIRED_LOCK_FIELDS = [
    "source_mode",
    "upstream_commit",
    "license_id",
    "patch_set",
    "compatibility_target",
    "last_verified_utc",
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

    if str(lock.get("compatibility_target", "")) != "IMPL-022":
        issues.append("lock compatibility_target must be IMPL-022")

    if not isinstance(lock.get("patch_set"), list):
        issues.append("lock patch_set must be a list")

    license_id = str(lock.get("license_id", ""))
    if license_id and license_id not in notices_text:
        issues.append("license_id from lock not found in THIRD_PARTY_NOTICES.md")

    upstream_repo_url = str(lock.get("upstream_repo_url", ""))
    if upstream_repo_url and upstream_repo_url not in notices_text and upstream_repo_url not in attribution_text:
        issues.append("upstream_repo_url not mentioned in notices or attribution docs")

    if upstream_commit and upstream_commit not in notices_text:
        issues.append("upstream_commit not mentioned in THIRD_PARTY_NOTICES.md")

    citation_markers = [
        "arXiv:2301.08243",
        "arXiv:2506.09985",
        "arXiv:2412.10925",
        "I-JEPA",
        "V-JEPA2",
    ]
    for marker in citation_markers:
        if marker not in attribution_text:
            issues.append(f"attribution doc missing citation marker '{marker}'")

    patch_hash = sha256_json(lock.get("patch_set", []))[:16]
    manifests = iter_manifest_paths()
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        scenario = manifest.get("scenario", {})

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

    if issues:
        print(f"FAIL: third-party compliance failed with {len(issues)} issue(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(
        "PASS: third-party compliance verified "
        f"lock={LOCK_PATH} notices={NOTICES_PATH} attribution={ATTRIBUTION_PATH} manifests={len(manifests)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
