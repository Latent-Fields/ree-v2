#!/usr/bin/env python3
"""Estimate profile runtime/memory and choose local vs remote execution mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import get_profiles
from ree_v2.experiments.resource_policy import decide_execution_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="all", help="Profile name or 'all'")
    parser.add_argument("--machine", required=True, help="Machine profile id")
    parser.add_argument("--thermal-throttling-detected", action="store_true")
    parser.add_argument("--oom-detected", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profiles = get_profiles(args.profile)

    print("experiment_type\tcondition\testimated_runtime_minutes\texecution_mode\toffload_reason")
    for profile in profiles:
        for condition in profile.conditions:
            mode, reasons, _batch_runtime, _seed_count = decide_execution_mode(
                profile,
                condition,
                args.machine,
                thermal_throttling_detected=args.thermal_throttling_detected,
                oom_detected=args.oom_detected,
            )
            reason = ",".join(reasons) if reasons else "none"
            print(
                f"{profile.experiment_type}\t{condition.name}\t{condition.resources.runtime_minutes:.1f}\t{mode}\t{reason}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
