#!/usr/bin/env python3
"""Pull remote results and place run packs under evidence/experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCOMING_DIR = REPO_ROOT / "jobs" / "incoming"

EXPECTED_FILES = {"manifest.json", "metrics.json", "summary.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-run-dir", type=Path, default=DEFAULT_INCOMING_DIR)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_result_bundles(job_run_dir: Path) -> list[Path]:
    bundles: list[Path] = []
    if not job_run_dir.exists():
        return bundles
    for candidate in sorted(job_run_dir.rglob("*")):
        if candidate.is_dir() and EXPECTED_FILES.issubset({path.name for path in candidate.iterdir() if path.is_file()}):
            bundles.append(candidate)
    return bundles


def parse_target(bundle_dir: Path) -> tuple[str, str] | None:
    # Expected naming: <experiment_type>__<run_id>
    parts = bundle_dir.name.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_optional_artifacts(bundle: Path, target_dir: Path, manifest: dict[str, object]) -> None:
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return

    adapter_path = artifacts.get("adapter_signals_path")
    if isinstance(adapter_path, str):
        src = bundle / adapter_path
        dst = target_dir / adapter_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for key in ("traces_dir", "media_dir"):
        rel = artifacts.get(key)
        if not isinstance(rel, str):
            continue
        src_dir = bundle / rel
        dst_dir = target_dir / rel
        if src_dir.is_dir():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def main() -> int:
    args = parse_args()
    args.runs_root.mkdir(parents=True, exist_ok=True)

    bundles = discover_result_bundles(args.job_run_dir)
    if not bundles:
        print(f"PASS: no completed remote result bundles in {args.job_run_dir}")
        return 0

    imported = 0
    skipped = 0
    for bundle in bundles:
        target = parse_target(bundle)
        if target is None:
            print(f"skip: unrecognized bundle naming format: {bundle.name}")
            skipped += 1
            continue

        experiment_type, run_id = target
        target_dir = args.runs_root / experiment_type / "runs" / run_id

        if args.dry_run:
            print(f"dry-run import OK: {bundle} -> {target_dir}")
            imported += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name in EXPECTED_FILES:
            shutil.copy2(bundle / file_name, target_dir / file_name)
        manifest_payload = _load_json(bundle / "manifest.json")
        _copy_optional_artifacts(bundle, target_dir, manifest_payload)
        imported += 1
        print(f"imported: {bundle} -> {target_dir}")

    print(f"PASS: processed {imported} bundle(s), skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
