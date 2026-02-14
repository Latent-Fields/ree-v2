#!/usr/bin/env python3
"""Deterministic replay gate for qualification profile metric vectors."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.experiments.profiles import get_profiles, simulate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="all", help="Profile name or 'all'")
    parser.add_argument("--max-abs-delta", type=float, default=1e-6)
    return parser.parse_args()


def stream_presence(include_uncertainty: bool, include_action_token: bool) -> dict[str, bool]:
    return {
        "z_t": True,
        "z_hat": True,
        "pe_latent": True,
        "uncertainty_latent": include_uncertainty,
        "trace_context_mask_ids": True,
        "trace_action_token": include_action_token,
    }


def main() -> int:
    args = parse_args()
    profiles = get_profiles(args.profile)

    issues: list[str] = []
    replay_pairs = 0

    for profile in profiles:
        for condition in profile.conditions:
            seed = profile.default_seeds[0]
            first = simulate_metrics(profile.experiment_type, condition.name, seed)
            second = simulate_metrics(profile.experiment_type, condition.name, seed)
            replay_pairs += 1

            if set(first) != set(second):
                issues.append(
                    f"{profile.experiment_type}/{condition.name}: key drift detected "
                    f"lhs={sorted(first)} rhs={sorted(second)}"
                )
                continue

            max_abs_delta = 0.0
            for key in sorted(first):
                delta = abs(first[key] - second[key])
                max_abs_delta = max(max_abs_delta, delta)
                if delta > args.max_abs_delta + 1e-18:
                    issues.append(
                        f"{profile.experiment_type}/{condition.name}: metric drift {key} "
                        f"delta={delta:.10f} > {args.max_abs_delta}"
                    )

            first_stream = stream_presence(condition.include_uncertainty, condition.include_action_token)
            second_stream = stream_presence(condition.include_uncertainty, condition.include_action_token)
            if first_stream != second_stream:
                issues.append(f"{profile.experiment_type}/{condition.name}: stream presence drift detected")

            print(
                f"checked {profile.experiment_type}/{condition.name} seed={seed} "
                f"max_abs_delta={max_abs_delta:.12g}"
            )

    if issues:
        print(f"FAIL: deterministic replay gate failed with {len(issues)} issue(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print(
        f"PASS: deterministic replay gate passed for {replay_pairs} condition replay pair(s) "
        f"at tolerance {args.max_abs_delta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
