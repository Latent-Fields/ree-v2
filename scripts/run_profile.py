#!/usr/bin/env python3
"""Run one profile/condition/seed via deterministic toy env runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import get_profile
from ree_v2.experiments.runner import execute_profile_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Experiment type/profile")
    parser.add_argument("--condition", default=None, help="Condition name (default: first profile condition)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "evidence" / "experiments")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-utc", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_profile(args.profile)

    condition_name = args.condition or profile.conditions[0].name
    if condition_name not in {item.name for item in profile.conditions}:
        known = ", ".join(item.name for item in profile.conditions)
        print(f"FAIL: unknown condition '{condition_name}' for profile '{profile.experiment_type}'. Known: {known}")
        return 1

    result = execute_profile_condition(
        experiment_type=profile.experiment_type,
        condition_name=condition_name,
        seed=args.seed,
        steps=args.steps,
        runs_root=args.runs_root,
        run_id=args.run_id,
        timestamp_utc=args.timestamp_utc,
        write=True,
    )

    print(
        f"PASS: run emitted profile={result.experiment_type} condition={result.condition_name} "
        f"seed={result.seed} run_id={result.run_id} status={result.status} path={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
