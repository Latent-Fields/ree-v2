#!/usr/bin/env python3
"""Pull remote results and place run packs under evidence/experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXPECTED_FILES = {"manifest.json", "metrics.json", "summary.md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-run-dir", required=True, type=Path)
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
        adapter_signals = bundle / "jepa_adapter_signals.v1.json"
        if adapter_signals.exists():
            shutil.copy2(adapter_signals, target_dir / adapter_signals.name)
        imported += 1
        print(f"imported: {bundle} -> {target_dir}")

    print(f"PASS: processed {imported} bundle(s), skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
