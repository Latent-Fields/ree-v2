"""Hook payload emitter for qualification and planned stubs."""

from __future__ import annotations

from typing import Any


def emit_v2_hooks(
    *,
    z_t: list[float],
    z_hat: list[list[float]],
    pe_latent: dict[str, Any],
    context_mask_ids: list[str],
    include_uncertainty: bool,
    uncertainty_latent: dict[str, float] | None,
    include_action_token: bool,
    action_token: str | None,
    commit_boundary: dict[str, Any],
    tri_loop_trace: dict[str, Any],
    control_axes: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hooks: dict[str, dict[str, Any]] = {
        "HK-001": {"z_t": z_t},
        "HK-002": {"z_hat": z_hat},
        "HK-003": {"pe_latent": {"mean": pe_latent["mean"], "p95": pe_latent["p95"]}},
        "HK-005": {"trace": {"context_mask_ids": context_mask_ids}},
        "HK-007": {"commit_boundary": commit_boundary},
        "HK-008": {"trace": tri_loop_trace},
        "HK-009": {"control_axes": control_axes},
    }

    if include_uncertainty and uncertainty_latent is not None:
        hooks["HK-004"] = {"uncertainty_latent": {"dispersion": uncertainty_latent["dispersion"]}}

    if include_action_token and action_token is not None:
        hooks["HK-006"] = {"trace": {"action_token": action_token}}

    return hooks


def emit_planned_stub_hooks() -> dict[str, dict[str, Any]]:
    return {
        "HK-101": {
            "hook_id": "HK-101",
            "planned_stub": True,
            "pre_commit_error": 0.0,
            "candidate_trajectory_id": "stub-candidate-001",
        },
        "HK-102": {
            "hook_id": "HK-102",
            "planned_stub": True,
            "post_commit_error": 0.0,
            "committed_trajectory_id": "stub-commit-001",
        },
        "HK-103": {
            "hook_id": "HK-103",
            "planned_stub": True,
            "commitment_trace_id": "stub-trace-001",
            "committed_trajectory_id": "stub-commit-001",
        },
        "HK-104": {
            "hook_id": "HK-104",
            "planned_stub": True,
            "candidate_trajectory_id": "stub-candidate-001",
            "candidate_source": "bootstrap_stub",
            "candidate_horizon": 3,
        },
    }
