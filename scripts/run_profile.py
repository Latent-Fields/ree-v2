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
    parser.add_argument(
        "--backend",
        default="internal_minimal",
        choices=["internal_minimal", "jepa_inference"],
        help="Latent substrate backend mode",
    )
    parser.add_argument(
        "--jepa-checkpoint-path",
        type=Path,
        default=None,
        help="Optional local JEPA checkpoint path for jepa_inference backend",
    )
    parser.add_argument(
        "--force-synthetic-frames",
        action="store_true",
        help="Force deterministic synthetic-frame fallback for jepa_inference smoke mode",
    )
    parser.add_argument(
        "--require-real-jepa",
        action="store_true",
        help="Fail if jepa_inference cannot load a real checkpoint and falls back to synthetic mode",
    )
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
    if args.require_real_jepa and args.backend != "jepa_inference":
        print("FAIL: --require-real-jepa requires --backend jepa_inference")
        return 1
    if args.require_real_jepa and args.force_synthetic_frames:
        print("FAIL: --require-real-jepa is incompatible with --force-synthetic-frames")
        return 1

    try:
        result = execute_profile_condition(
            experiment_type=profile.experiment_type,
            condition_name=condition_name,
            seed=args.seed,
            backend=args.backend,
            steps=args.steps,
            runs_root=args.runs_root,
            run_id=args.run_id,
            timestamp_utc=args.timestamp_utc,
            jepa_checkpoint_path=args.jepa_checkpoint_path,
            force_synthetic_frames=args.force_synthetic_frames,
            require_real_jepa=args.require_real_jepa,
            write=True,
        )
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(
        f"PASS: run emitted profile={result.experiment_type} condition={result.condition_name} "
        f"seed={result.seed} backend={result.backend} run_id={result.run_id} status={result.status} path={result.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
