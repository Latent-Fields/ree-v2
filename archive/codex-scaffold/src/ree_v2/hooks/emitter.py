"""Hook payload emitter for qualification bridge contracts."""

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


def emit_bridge_commit_hooks(
    *,
    pre_commit_error: float,
    post_commit_error: float,
    candidate_trajectory_id: str,
    committed_trajectory_id: str,
    commitment_trace_id: str,
    candidate_source: str,
    candidate_horizon: int,
) -> dict[str, dict[str, Any]]:
    return {
        "HK-101": {
            "pre_commit_error": pre_commit_error,
            "candidate_trajectory_id": candidate_trajectory_id,
        },
        "HK-102": {
            "post_commit_error": post_commit_error,
            "committed_trajectory_id": committed_trajectory_id,
        },
        "HK-103": {
            "commitment_trace_id": commitment_trace_id,
            "committed_trajectory_id": committed_trajectory_id,
        },
        "HK-104": {
            "candidate_trajectory_id": candidate_trajectory_id,
            "candidate_source": candidate_source,
            "candidate_horizon": candidate_horizon,
        },
    }


def emit_planned_stub_hooks() -> dict[str, dict[str, Any]]:
    """Backward-compatible shim for callers expecting planned stub hooks."""

    return emit_bridge_commit_hooks(
        pre_commit_error=0.0,
        post_commit_error=0.0,
        candidate_trajectory_id="stub-candidate-001",
        committed_trajectory_id="stub-commit-001",
        commitment_trace_id="stub-trace-001",
        candidate_source="bootstrap_stub",
        candidate_horizon=3,
    )
