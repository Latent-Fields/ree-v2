#!/usr/bin/env python3
"""Run a batch of qualification runs through the toy environment runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import get_profiles
from ree_v2.experiments.runner import execute_profile_condition


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="all", help="Profile name or 'all'")
    parser.add_argument("--seeds", default="11", help="Comma-separated seeds")
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
    parser.add_argument(
        "--condition-mode",
        default="supports_only",
        choices=["supports_only", "all_conditions"],
        help="Which profile conditions to run",
    )
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "evidence" / "experiments")
    parser.add_argument("--timestamp-utc", default=None)
    return parser.parse_args()


def choose_conditions(profile, mode: str) -> list[str]:
    if mode == "all_conditions":
        return [item.name for item in profile.conditions]

    supports = [item.name for item in profile.conditions if item.evidence_direction == "supports"]
    if supports:
        return [supports[0]]
    return [profile.conditions[0].name]


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    profiles = get_profiles(args.profile)
    if args.require_real_jepa and args.backend != "jepa_inference":
        print("FAIL: --require-real-jepa requires --backend jepa_inference")
        return 1
    if args.require_real_jepa and args.force_synthetic_frames:
        print("FAIL: --require-real-jepa is incompatible with --force-synthetic-frames")
        return 1

    emitted = 0
    for profile in profiles:
        conditions = choose_conditions(profile, args.condition_mode)
        for condition_name in conditions:
            for seed in seeds:
                try:
                    result = execute_profile_condition(
                        experiment_type=profile.experiment_type,
                        condition_name=condition_name,
                        seed=seed,
                        backend=args.backend,
                        steps=args.steps,
                        runs_root=args.runs_root,
                        timestamp_utc=args.timestamp_utc,
                        jepa_checkpoint_path=args.jepa_checkpoint_path,
                        force_synthetic_frames=args.force_synthetic_frames,
                        require_real_jepa=args.require_real_jepa,
                        write=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"FAIL: profile={profile.experiment_type} condition={condition_name} "
                        f"seed={seed} backend={args.backend}: {exc}"
                    )
                    return 1
                emitted += 1
                print(
                    f"emitted profile={result.experiment_type} condition={result.condition_name} "
                    f"seed={result.seed} backend={result.backend} run_id={result.run_id} status={result.status}"
                )

    print(f"PASS: batch completed emitted_runs={emitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
