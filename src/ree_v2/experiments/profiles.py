"""Profile definitions for REE-v2 qualification bootstrap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ResourceEstimate:
    runtime_minutes: float
    memory_gb: float


@dataclass(frozen=True)
class ProfileCondition:
    name: str
    base_metrics: dict[str, float]
    evidence_direction: str
    include_uncertainty: bool
    include_action_token: bool
    resources: ResourceEstimate


@dataclass(frozen=True)
class FailureRule:
    signature_id: str
    metric_key: str
    op: str
    threshold: float


@dataclass(frozen=True)
class ClaimProfile:
    experiment_type: str
    claim_id: str
    evidence_class: str
    required_metric_keys: tuple[str, ...]
    default_seeds: tuple[int, ...]
    conditions: tuple[ProfileCondition, ...]
    failure_rules: tuple[FailureRule, ...]


PROFILE_CATALOG: dict[str, ClaimProfile] = {
    "trajectory_integrity": ClaimProfile(
        experiment_type="trajectory_integrity",
        claim_id="MECH-056",
        evidence_class="simulation",
        required_metric_keys=(
            "ledger_edit_detected_count",
            "explanation_policy_divergence_rate",
            "domination_lock_in_events",
            "commitment_reversal_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="trajectory_first_enabled",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=32.0, memory_gb=5.4),
                base_metrics={
                    "ledger_edit_detected_count": 0.0,
                    "explanation_policy_divergence_rate": 0.022,
                    "domination_lock_in_events": 0.0,
                    "commitment_reversal_rate": 0.041,
                    "latent_prediction_error_mean": 0.044,
                    "latent_prediction_error_p95": 0.082,
                    "latent_residual_coverage_rate": 0.983,
                    "precision_input_completeness_rate": 0.978,
                    "latent_uncertainty_calibration_error": 0.108,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="trajectory_first_ablated",
                evidence_direction="weakens",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=41.0, memory_gb=6.2),
                base_metrics={
                    "ledger_edit_detected_count": 2.0,
                    "explanation_policy_divergence_rate": 0.081,
                    "domination_lock_in_events": 1.0,
                    "commitment_reversal_rate": 0.123,
                    "latent_prediction_error_mean": 0.091,
                    "latent_prediction_error_p95": 0.154,
                    "latent_residual_coverage_rate": 0.901,
                    "precision_input_completeness_rate": 0.903,
                    "latent_uncertainty_calibration_error": 0.198,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="valence_on_mapping_adaptive",
                evidence_direction="weakens",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=37.0, memory_gb=5.9),
                base_metrics={
                    "ledger_edit_detected_count": 1.0,
                    "explanation_policy_divergence_rate": 0.067,
                    "domination_lock_in_events": 1.0,
                    "commitment_reversal_rate": 0.093,
                    "latent_prediction_error_mean": 0.072,
                    "latent_prediction_error_p95": 0.136,
                    "latent_residual_coverage_rate": 0.964,
                    "precision_input_completeness_rate": 0.957,
                    "latent_uncertainty_calibration_error": 0.149,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="valence_on_mapping_frozen",
                evidence_direction="mixed",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=38.0, memory_gb=6.0),
                base_metrics={
                    "ledger_edit_detected_count": 1.0,
                    "explanation_policy_divergence_rate": 0.055,
                    "domination_lock_in_events": 0.0,
                    "commitment_reversal_rate": 0.074,
                    "latent_prediction_error_mean": 0.066,
                    "latent_prediction_error_p95": 0.124,
                    "latent_residual_coverage_rate": 0.969,
                    "precision_input_completeness_rate": 0.962,
                    "latent_uncertainty_calibration_error": 0.138,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="valence_off_or_neutral_mapping_adaptive",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=35.0, memory_gb=5.6),
                base_metrics={
                    "ledger_edit_detected_count": 0.0,
                    "explanation_policy_divergence_rate": 0.031,
                    "domination_lock_in_events": 0.0,
                    "commitment_reversal_rate": 0.047,
                    "latent_prediction_error_mean": 0.049,
                    "latent_prediction_error_p95": 0.094,
                    "latent_residual_coverage_rate": 0.981,
                    "precision_input_completeness_rate": 0.974,
                    "latent_uncertainty_calibration_error": 0.112,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="valence_off_or_neutral_mapping_frozen",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=34.0, memory_gb=5.5),
                base_metrics={
                    "ledger_edit_detected_count": 0.0,
                    "explanation_policy_divergence_rate": 0.024,
                    "domination_lock_in_events": 0.0,
                    "commitment_reversal_rate": 0.036,
                    "latent_prediction_error_mean": 0.042,
                    "latent_prediction_error_p95": 0.083,
                    "latent_residual_coverage_rate": 0.986,
                    "precision_input_completeness_rate": 0.978,
                    "latent_uncertainty_calibration_error": 0.104,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("ledger_editing", "ledger_edit_detected_count", ">", 0.0),
            FailureRule("explanation_policy_divergence", "explanation_policy_divergence_rate", ">", 0.05),
            FailureRule("domination_lock_in", "domination_lock_in_events", ">", 0.0),
            FailureRule("mech056:latent_geometry_distortion", "latent_procrustes_drift", ">", 0.16),
            FailureRule(
                "mech056:valence_mapping_competition",
                "hippocampal_retrieval_valence_skew",
                ">",
                0.22,
            ),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
    "jepa_anchor_ablation": ClaimProfile(
        experiment_type="jepa_anchor_ablation",
        claim_id="MECH-058",
        evidence_class="ablation",
        required_metric_keys=(
            "latent_prediction_error_mean",
            "latent_prediction_error_p95",
            "latent_rollout_consistency_rate",
            "e1_e2_timescale_separation_ratio",
            "representation_drift_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="ema_anchor_on",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=68.0, memory_gb=8.7),
                base_metrics={
                    "latent_prediction_error_mean": 0.128,
                    "latent_prediction_error_p95": 0.206,
                    "latent_rollout_consistency_rate": 0.921,
                    "e1_e2_timescale_separation_ratio": 1.84,
                    "representation_drift_rate": 0.039,
                    "latent_residual_coverage_rate": 0.984,
                    "precision_input_completeness_rate": 0.978,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="ema_anchor_off",
                evidence_direction="weakens",
                include_uncertainty=False,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=64.0, memory_gb=8.2),
                base_metrics={
                    "latent_prediction_error_mean": 0.209,
                    "latent_prediction_error_p95": 0.341,
                    "latent_rollout_consistency_rate": 0.646,
                    "e1_e2_timescale_separation_ratio": 1.21,
                    "representation_drift_rate": 0.138,
                    "latent_residual_coverage_rate": 0.931,
                    "precision_input_completeness_rate": 0.938,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech058:ema_drift_under_shift", "representation_drift_rate", ">", 0.12),
            FailureRule("mech058:latent_cluster_collapse", "latent_rollout_consistency_rate", "<", 0.70),
            FailureRule("mech058:anchor_separation_collapse", "e1_e2_timescale_separation_ratio", "<", 1.50),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
    "jepa_uncertainty_channels": ClaimProfile(
        experiment_type="jepa_uncertainty_channels",
        claim_id="MECH-059",
        evidence_class="simulation",
        required_metric_keys=(
            "latent_prediction_error_mean",
            "latent_uncertainty_calibration_error",
            "precision_input_completeness_rate",
            "uncertainty_coverage_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="deterministic_plus_dispersion",
                evidence_direction="mixed",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=74.0, memory_gb=9.1),
                base_metrics={
                    "latent_prediction_error_mean": 0.164,
                    "latent_prediction_error_p95": 0.242,
                    "latent_uncertainty_calibration_error": 0.156,
                    "precision_input_completeness_rate": 0.944,
                    "uncertainty_coverage_rate": 0.882,
                    "latent_residual_coverage_rate": 0.976,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="explicit_uncertainty_head",
                evidence_direction="supports",
                include_uncertainty=True,
                include_action_token=False,
                resources=ResourceEstimate(runtime_minutes=88.0, memory_gb=10.6),
                base_metrics={
                    "latent_prediction_error_mean": 0.147,
                    "latent_prediction_error_p95": 0.224,
                    "latent_uncertainty_calibration_error": 0.118,
                    "precision_input_completeness_rate": 0.971,
                    "uncertainty_coverage_rate": 0.927,
                    "latent_residual_coverage_rate": 0.982,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech059:calibration_slope_break", "latent_uncertainty_calibration_error", ">", 0.20),
            FailureRule(
                "mech059:uncertainty_metric_gaming_detected",
                "precision_minus_coverage",
                ">",
                0.20,
            ),
            FailureRule("mech059:abstention_reliability_collapse", "uncertainty_coverage_rate", "<", 0.70),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
    "commit_dual_error_channels": ClaimProfile(
        experiment_type="commit_dual_error_channels",
        claim_id="MECH-060",
        evidence_class="ablation",
        required_metric_keys=(
            "pre_commit_error_signal_to_noise",
            "post_commit_error_attribution_gain",
            "cross_channel_leakage_rate",
            "commitment_reversal_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="single_error_stream",
                evidence_direction="weakens",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=56.0, memory_gb=7.9),
                base_metrics={
                    "pre_commit_error_signal_to_noise": 0.92,
                    "post_commit_error_attribution_gain": 0.191,
                    "cross_channel_leakage_rate": 0.248,
                    "commitment_reversal_rate": 0.121,
                    "latent_prediction_error_mean": 0.179,
                    "latent_prediction_error_p95": 0.262,
                    "latent_residual_coverage_rate": 0.951,
                    "precision_input_completeness_rate": 0.931,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="pre_post_split_streams",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=63.0, memory_gb=8.4),
                base_metrics={
                    "pre_commit_error_signal_to_noise": 2.18,
                    "post_commit_error_attribution_gain": 0.431,
                    "cross_channel_leakage_rate": 0.086,
                    "commitment_reversal_rate": 0.053,
                    "latent_prediction_error_mean": 0.142,
                    "latent_prediction_error_p95": 0.221,
                    "latent_residual_coverage_rate": 0.976,
                    "precision_input_completeness_rate": 0.966,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("mech060:precommit_channel_contamination", "pre_commit_error_signal_to_noise", "<", 1.20),
            FailureRule("mech060:postcommit_channel_contamination", "cross_channel_leakage_rate", ">", 0.20),
            FailureRule("mech060:attribution_reliability_break", "post_commit_error_attribution_gain", "<", 0.25),
            FailureRule("mech060:commitment_reversal_spike", "commitment_reversal_rate", ">", 0.10),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
    "tri_loop_arbitration_policy": ClaimProfile(
        experiment_type="tri_loop_arbitration_policy",
        claim_id="Q-016",
        evidence_class="simulation",
        required_metric_keys=(
            "tri_loop_gate_conflict_rate",
            "tri_loop_policy_alignment_rate",
            "tri_loop_arbitration_override_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="veto_lattice",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=52.0, memory_gb=7.2),
                base_metrics={
                    "tri_loop_gate_conflict_rate": 0.06,
                    "tri_loop_policy_alignment_rate": 0.92,
                    "tri_loop_arbitration_override_rate": 0.08,
                    "latent_prediction_error_mean": 0.13,
                    "latent_prediction_error_p95": 0.21,
                    "latent_residual_coverage_rate": 0.98,
                    "precision_input_completeness_rate": 0.97,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="weighted_merge",
                evidence_direction="mixed",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=54.0, memory_gb=7.4),
                base_metrics={
                    "tri_loop_gate_conflict_rate": 0.12,
                    "tri_loop_policy_alignment_rate": 0.86,
                    "tri_loop_arbitration_override_rate": 0.14,
                    "latent_prediction_error_mean": 0.16,
                    "latent_prediction_error_p95": 0.24,
                    "latent_residual_coverage_rate": 0.98,
                    "precision_input_completeness_rate": 0.97,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="mode_conditioned_precedence",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=56.0, memory_gb=7.6),
                base_metrics={
                    "tri_loop_gate_conflict_rate": 0.08,
                    "tri_loop_policy_alignment_rate": 0.90,
                    "tri_loop_arbitration_override_rate": 0.10,
                    "latent_prediction_error_mean": 0.145,
                    "latent_prediction_error_p95": 0.23,
                    "latent_residual_coverage_rate": 0.98,
                    "precision_input_completeness_rate": 0.97,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("q016:tri_loop_conflict_spike", "tri_loop_gate_conflict_rate", ">", 0.15),
            FailureRule("q016:tri_loop_alignment_break", "tri_loop_policy_alignment_rate", "<", 0.85),
            FailureRule("q016:tri_loop_override_spike", "tri_loop_arbitration_override_rate", ">", 0.15),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
    "control_axis_ablation": ClaimProfile(
        experiment_type="control_axis_ablation",
        claim_id="Q-017",
        evidence_class="ablation",
        required_metric_keys=(
            "control_axis_stability_index",
            "control_axis_readout_entropy",
            "control_axis_policy_loss_rate",
        ),
        default_seeds=(11, 29, 47),
        conditions=(
            ProfileCondition(
                name="full_axis",
                evidence_direction="supports",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=58.0, memory_gb=7.5),
                base_metrics={
                    "control_axis_stability_index": 0.91,
                    "control_axis_readout_entropy": 1.10,
                    "control_axis_policy_loss_rate": 0.05,
                    "latent_prediction_error_mean": 0.14,
                    "latent_prediction_error_p95": 0.23,
                    "latent_residual_coverage_rate": 0.98,
                    "precision_input_completeness_rate": 0.95,
                    "fatal_error_count": 0.0,
                },
            ),
            ProfileCondition(
                name="reduced_axis",
                evidence_direction="weakens",
                include_uncertainty=False,
                include_action_token=True,
                resources=ResourceEstimate(runtime_minutes=58.0, memory_gb=7.5),
                base_metrics={
                    "control_axis_stability_index": 0.73,
                    "control_axis_readout_entropy": 0.62,
                    "control_axis_policy_loss_rate": 0.17,
                    "latent_prediction_error_mean": 0.19,
                    "latent_prediction_error_p95": 0.28,
                    "latent_residual_coverage_rate": 0.98,
                    "precision_input_completeness_rate": 0.88,
                    "fatal_error_count": 0.0,
                },
            ),
        ),
        failure_rules=(
            FailureRule("q017:control_axis_stability_drop", "control_axis_stability_index", "<", 0.80),
            FailureRule("q017:control_axis_policy_loss_spike", "control_axis_policy_loss_rate", ">", 0.12),
            FailureRule("q017:control_axis_entropy_collapse", "control_axis_readout_entropy", "<", 0.75),
            FailureRule("bridge:commit_boundary_join_coverage_gap", "commit_boundary_join_coverage_rate", "<", 0.99),
            FailureRule("bridge:tri_loop_trace_coverage_gap", "tri_loop_trace_coverage_rate", "<", 0.99),
        ),
    ),
}

CLAIM_PROBE_ALIAS_TO_BASE: dict[str, str] = {
    "claim_probe_mech_056": "trajectory_integrity",
    "claim_probe_mech_058": "jepa_anchor_ablation",
    "claim_probe_mech_059": "jepa_uncertainty_channels",
    "claim_probe_mech_060": "commit_dual_error_channels",
    "claim_probe_mech_062": "tri_loop_arbitration_policy",
    "claim_probe_q_017": "control_axis_ablation",
}


def _add_claim_probe_aliases(catalog: dict[str, ClaimProfile]) -> None:
    """Register claim_probe_* aliases that reuse validated core profiles.

    This keeps proposal-facing claim probe experiment types executable while
    preserving the same metric contracts and failure signatures as the base
    profiles.
    """

    alias_specs = [
        ("claim_probe_mech_056", "MECH-056"),
        ("claim_probe_mech_058", "MECH-058"),
        ("claim_probe_mech_059", "MECH-059"),
        ("claim_probe_mech_060", "MECH-060"),
        ("claim_probe_mech_062", "MECH-062"),
        ("claim_probe_q_017", "Q-017"),
    ]

    for alias_name, claim_id in alias_specs:
        base_name = CLAIM_PROBE_ALIAS_TO_BASE[alias_name]
        base = catalog[base_name]
        catalog[alias_name] = ClaimProfile(
            experiment_type=alias_name,
            claim_id=claim_id,
            evidence_class=base.evidence_class,
            required_metric_keys=base.required_metric_keys,
            default_seeds=base.default_seeds,
            conditions=base.conditions,
            failure_rules=base.failure_rules,
        )


_add_claim_probe_aliases(PROFILE_CATALOG)


def resolve_execution_experiment_type(experiment_type: str) -> str:
    """Resolve proposal-facing aliases to their executable base profile."""

    if experiment_type in CLAIM_PROBE_ALIAS_TO_BASE:
        return CLAIM_PROBE_ALIAS_TO_BASE[experiment_type]
    if experiment_type.startswith("claim_probe_"):
        claim_id = _claim_id_from_probe_experiment_type(experiment_type)
        if claim_id:
            return _infer_base_profile_from_claim_id(claim_id)
    return experiment_type


def get_profiles(profile: str = "all") -> list[ClaimProfile]:
    if profile == "all":
        return [PROFILE_CATALOG[name] for name in sorted(PROFILE_CATALOG)]
    return [get_profile(profile)]


def get_profile(experiment_type: str) -> ClaimProfile:
    if experiment_type.startswith("claim_probe_") and experiment_type not in PROFILE_CATALOG:
        claim_id = _claim_id_from_probe_experiment_type(experiment_type)
        if claim_id:
            base_name = _infer_base_profile_from_claim_id(claim_id)
            base = PROFILE_CATALOG[base_name]
            PROFILE_CATALOG[experiment_type] = ClaimProfile(
                experiment_type=experiment_type,
                claim_id=claim_id,
                evidence_class=base.evidence_class,
                required_metric_keys=base.required_metric_keys,
                default_seeds=base.default_seeds,
                conditions=base.conditions,
                failure_rules=base.failure_rules,
            )
            CLAIM_PROBE_ALIAS_TO_BASE[experiment_type] = base_name
    if experiment_type not in PROFILE_CATALOG:
        known = ", ".join(sorted(PROFILE_CATALOG))
        raise KeyError(f"Unknown profile '{experiment_type}'. Known profiles: {known}")
    return PROFILE_CATALOG[experiment_type]


def _claim_id_from_probe_experiment_type(experiment_type: str) -> str | None:
    if not experiment_type.startswith("claim_probe_"):
        return None
    token = experiment_type[len("claim_probe_") :].strip()
    parts = [p for p in token.split("_") if p]
    if len(parts) < 2:
        return None
    prefix = parts[0].upper()
    number = parts[1]
    if not number.isdigit():
        return None
    return f"{prefix}-{number.zfill(3)}"


def _infer_base_profile_from_claim_id(claim_id: str) -> str:
    explicit: dict[str, str] = {
        "MECH-056": "trajectory_integrity",
        "MECH-058": "jepa_anchor_ablation",
        "MECH-059": "jepa_uncertainty_channels",
        "MECH-060": "commit_dual_error_channels",
        "MECH-062": "tri_loop_arbitration_policy",
        "Q-016": "tri_loop_arbitration_policy",
        "Q-017": "control_axis_ablation",
    }
    if claim_id in explicit:
        return explicit[claim_id]
    if claim_id in {"MECH-053", "MECH-054", "MECH-063", "ARC-005"}:
        return "control_axis_ablation"
    if claim_id in {"MECH-061", "IMPL-022", "ARC-003"}:
        return "commit_dual_error_channels"
    return "trajectory_integrity"


def _seeded_jitter(
    experiment_type: str,
    condition_name: str,
    seed: int,
    metric_key: str,
    scale: float,
) -> float:
    material = f"{experiment_type}|{condition_name}|{seed}|{metric_key}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    centered = (raw * 2.0) - 1.0
    return centered * scale


def _apply_operation(lhs: float, op: str, rhs: float) -> bool:
    operations: dict[str, Callable[[float, float], bool]] = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }
    return operations[op](lhs, rhs)


def _resolve_metric_key(metrics: dict[str, float], metric_key: str) -> float:
    if metric_key == "precision_minus_coverage":
        return metrics.get("precision_input_completeness_rate", 0.0) - metrics.get("uncertainty_coverage_rate", 0.0)
    if metric_key in {"commit_boundary_join_coverage_rate", "tri_loop_trace_coverage_rate"}:
        # Historical runs predate bridge metrics; treat missing as non-failing for backward compatibility.
        return metrics.get(metric_key, 1.0)
    return metrics.get(metric_key, 0.0)


def simulate_metrics(experiment_type: str, condition_name: str, seed: int) -> dict[str, float]:
    # Local import avoids circular dependency with the runner module importing profile metadata.
    from ree_v2.experiments.runner import execute_profile_condition

    result = execute_profile_condition(
        experiment_type=experiment_type,
        condition_name=condition_name,
        seed=seed,
        write=False,
    )
    return dict(result.metrics_values)


def evaluate_failure_signatures(experiment_type: str, metrics: dict[str, float]) -> list[str]:
    profile = get_profile(experiment_type)
    signatures: list[str] = []
    for rule in profile.failure_rules:
        lhs = _resolve_metric_key(metrics, rule.metric_key)
        if _apply_operation(lhs, rule.op, rule.threshold):
            signatures.append(rule.signature_id)
    return signatures
