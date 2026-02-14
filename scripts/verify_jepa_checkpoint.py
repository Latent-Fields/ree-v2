#!/usr/bin/env python3
"""Verify local JEPA checkpoint file integrity against pinned lock metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = REPO_ROOT / "third_party" / "jepa_sources.lock.v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=Path,
        help="Path to local checkpoint file (for example model.safetensors)",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Path to jepa_sources lock file",
    )
    parser.add_argument(
        "--variant",
        choices=["primary", "fallback"],
        default="primary",
        help="Which checkpoint pin set to validate against in lock",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_fields(lock: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "fallback":
        return {
            "repo_id": lock.get("checkpoint_fallback_repo_id", ""),
            "revision": lock.get("checkpoint_revision", ""),
            "filename": lock.get("checkpoint_fallback_filename", ""),
            "sha256": lock.get("checkpoint_fallback_sha256", ""),
            "size_bytes": lock.get("checkpoint_fallback_size_bytes", 0),
            "license_id": lock.get("checkpoint_fallback_license_id", ""),
        }
    return {
        "repo_id": lock.get("checkpoint_repo_id", ""),
        "revision": lock.get("checkpoint_revision", ""),
        "filename": lock.get("checkpoint_filename", ""),
        "sha256": lock.get("checkpoint_sha256", ""),
        "size_bytes": lock.get("checkpoint_size_bytes", 0),
        "license_id": lock.get("checkpoint_license_id", ""),
    }


def main() -> int:
    args = parse_args()
    issues: list[str] = []

    if not args.lock.exists():
        print(f"FAIL: missing lock file: {args.lock}")
        return 1
    if not args.checkpoint_path.exists():
        print(f"FAIL: missing checkpoint file: {args.checkpoint_path}")
        return 1
    if not args.checkpoint_path.is_file():
        print(f"FAIL: checkpoint path is not a file: {args.checkpoint_path}")
        return 1

    lock = load_json(args.lock)
    pinned = lock_fields(lock, args.variant)

    required = ("repo_id", "revision", "filename", "sha256", "size_bytes", "license_id")
    for key in required:
        value = pinned.get(key)
        if value in ("", None, 0):
            issues.append(f"lock missing/empty '{key}' for variant '{args.variant}'")

    observed_name = args.checkpoint_path.name
    observed_size = args.checkpoint_path.stat().st_size
    observed_sha = sha256_file(args.checkpoint_path)

    if observed_name != str(pinned.get("filename", "")):
        issues.append(
            f"filename mismatch expected={pinned.get('filename')} observed={observed_name}"
        )
    if observed_sha.lower() != str(pinned.get("sha256", "")).lower():
        issues.append(
            f"sha256 mismatch expected={pinned.get('sha256')} observed={observed_sha}"
        )
    try:
        expected_size = int(pinned.get("size_bytes", 0))
    except Exception:
        expected_size = -1
        issues.append(f"invalid size_bytes in lock: {pinned.get('size_bytes')}")
    if expected_size != observed_size:
        issues.append(
            f"size mismatch expected={expected_size} observed={observed_size}"
        )

    if issues:
        print(
            f"FAIL: checkpoint integrity verification failed variant={args.variant} "
            f"repo_id={pinned.get('repo_id')} revision={pinned.get('revision')}"
        )
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(
        "PASS: checkpoint integrity verified "
        f"variant={args.variant} repo_id={pinned['repo_id']} revision={pinned['revision']} "
        f"license={pinned['license_id']} path={args.checkpoint_path} "
        f"sha256={observed_sha} size_bytes={observed_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
