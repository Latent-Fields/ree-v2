"""Seeded toy environment drivers for qualification profiles."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToyRollout:
    experiment_type: str
    condition_name: str
    seed: int
    steps: int
    signals: dict[str, list[float]]
    events: dict[str, list[int]]
    context_values: list[float]
    actions: list[float]


def _stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return int(digest[:16], 16)


def _rng_for(experiment_type: str, condition_name: str, seed: int) -> random.Random:
    return random.Random(_stable_seed(experiment_type, condition_name, seed))


def _init_rollout(experiment_type: str, condition_name: str, seed: int, steps: int) -> ToyRollout:
    return ToyRollout(
        experiment_type=experiment_type,
        condition_name=condition_name,
        seed=seed,
        steps=steps,
        signals={},
        events={},
        context_values=[],
        actions=[],
    )


def _trajectory_integrity_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "trajectory_first_enabled": {
            "obs_noise": 0.035,
            "error_sigma": 0.038,
            "ledger_prob": 0.004,
            "domination_prob": 0.003,
            "reversal_prob": 0.042,
            "divergence_base": 0.022,
            "uncertainty_scale": 1.05,
            "policy_error_prob": 0.030,
            "procrustes_base": 0.083,
            "knn_base": 0.915,
            "trustworthiness_base": 0.924,
            "continuity_base": 0.917,
            "retrieval_skew_base": 0.064,
            "path_entropy_base": 0.452,
            "recall_error_base": 0.061,
            "valence_mode": "off_or_neutral",
            "mapping_mode": "adaptive",
        },
        "trajectory_first_ablated": {
            "obs_noise": 0.055,
            "error_sigma": 0.073,
            "ledger_prob": 0.060,
            "domination_prob": 0.035,
            "reversal_prob": 0.118,
            "divergence_base": 0.081,
            "uncertainty_scale": 0.88,
            "policy_error_prob": 0.115,
            "procrustes_base": 0.123,
            "knn_base": 0.846,
            "trustworthiness_base": 0.862,
            "continuity_base": 0.853,
            "retrieval_skew_base": 0.274,
            "path_entropy_base": 0.684,
            "recall_error_base": 0.143,
            "valence_mode": "on",
            "mapping_mode": "adaptive",
        },
        "valence_on_mapping_adaptive": {
            "obs_noise": 0.051,
            "error_sigma": 0.061,
            "ledger_prob": 0.041,
            "domination_prob": 0.024,
            "reversal_prob": 0.093,
            "divergence_base": 0.067,
            "uncertainty_scale": 0.95,
            "policy_error_prob": 0.102,
            "procrustes_base": 0.121,
            "knn_base": 0.861,
            "trustworthiness_base": 0.877,
            "continuity_base": 0.866,
            "retrieval_skew_base": 0.286,
            "path_entropy_base": 0.672,
            "recall_error_base": 0.141,
            "valence_mode": "on",
            "mapping_mode": "adaptive",
        },
        "valence_on_mapping_frozen": {
            "obs_noise": 0.046,
            "error_sigma": 0.055,
            "ledger_prob": 0.030,
            "domination_prob": 0.018,
            "reversal_prob": 0.073,
            "divergence_base": 0.056,
            "uncertainty_scale": 0.98,
            "policy_error_prob": 0.079,
            "procrustes_base": 0.162,
            "knn_base": 0.812,
            "trustworthiness_base": 0.824,
            "continuity_base": 0.815,
            "retrieval_skew_base": 0.114,
            "path_entropy_base": 0.538,
            "recall_error_base": 0.102,
            "valence_mode": "on",
            "mapping_mode": "frozen",
        },
        "valence_off_or_neutral_mapping_adaptive": {
            "obs_noise": 0.037,
            "error_sigma": 0.041,
            "ledger_prob": 0.011,
            "domination_prob": 0.006,
            "reversal_prob": 0.046,
            "divergence_base": 0.028,
            "uncertainty_scale": 1.03,
            "policy_error_prob": 0.038,
            "procrustes_base": 0.092,
            "knn_base": 0.901,
            "trustworthiness_base": 0.914,
            "continuity_base": 0.903,
            "retrieval_skew_base": 0.072,
            "path_entropy_base": 0.472,
            "recall_error_base": 0.073,
            "valence_mode": "off_or_neutral",
            "mapping_mode": "adaptive",
        },
        "valence_off_or_neutral_mapping_frozen": {
            "obs_noise": 0.033,
            "error_sigma": 0.036,
            "ledger_prob": 0.008,
            "domination_prob": 0.004,
            "reversal_prob": 0.036,
            "divergence_base": 0.024,
            "uncertainty_scale": 1.06,
            "policy_error_prob": 0.029,
            "procrustes_base": 0.081,
            "knn_base": 0.923,
            "trustworthiness_base": 0.933,
            "continuity_base": 0.924,
            "retrieval_skew_base": 0.054,
            "path_entropy_base": 0.444,
            "recall_error_base": 0.064,
            "valence_mode": "off_or_neutral",
            "mapping_mode": "frozen",
        },
    }[condition_name]

    rng = _rng_for("trajectory_integrity", condition_name, seed)
    rollout = _init_rollout("trajectory_integrity", condition_name, seed, steps)

    latent_errors: list[float] = []
    uncertainties: list[float] = []
    divergence_series: list[float] = []
    procrustes_drift_series: list[float] = []
    knn_overlap_series: list[float] = []
    trustworthiness_series: list[float] = []
    continuity_series: list[float] = []
    retrieval_skew_series: list[float] = []
    path_entropy_series: list[float] = []
    recall_error_series: list[float] = []

    events = {
        "ledger_edit": [],
        "domination_lock_in": [],
        "commitment_reversal": [],
        "policy_error": [],
        "conflict_signature": [],
        "residual_present": [],
        "precision_complete": [],
    }

    state = rng.uniform(-0.25, 0.25)
    for step in range(steps):
        state += 0.08 * math.sin((step + 1) / 9.0) + rng.gauss(0.0, cfg["obs_noise"])
        target = state + 0.05 * math.sin((step + 1) / 5.0)
        prediction = target + rng.gauss(0.0, cfg["error_sigma"])
        error = abs(prediction - target)
        uncertainty = max(0.0, (error * cfg["uncertainty_scale"]) + rng.gauss(0.0, 0.01))

        divergence = max(0.0, min(1.0, cfg["divergence_base"] + abs(rng.gauss(0.0, 0.011))))
        procrustes_drift = max(0.0, cfg["procrustes_base"] + abs(rng.gauss(0.0, 0.014)))
        knn_overlap = max(0.0, min(1.0, cfg["knn_base"] - abs(rng.gauss(0.0, 0.012))))
        trustworthiness = max(
            0.0,
            min(1.0, cfg["trustworthiness_base"] - abs(rng.gauss(0.0, 0.013))),
        )
        continuity = max(0.0, min(1.0, cfg["continuity_base"] - abs(rng.gauss(0.0, 0.013))))

        valence_adjust = 0.02 if cfg["valence_mode"] == "on" else -0.005
        mapping_adjust = 0.035 if cfg["mapping_mode"] == "adaptive" else -0.012
        retrieval_skew = max(
            0.0,
            cfg["retrieval_skew_base"] + valence_adjust + mapping_adjust + rng.gauss(0.0, 0.016),
        )
        path_entropy = max(
            0.0,
            min(
                1.0,
                cfg["path_entropy_base"] + (0.028 if cfg["mapping_mode"] == "adaptive" else -0.01) + rng.gauss(0.0, 0.019),
            ),
        )
        recall_error = max(
            0.0,
            cfg["recall_error_base"] + (0.018 if cfg["valence_mode"] == "on" else -0.004) + abs(rng.gauss(0.0, 0.011)),
        )

        latent_errors.append(error)
        uncertainties.append(uncertainty)
        divergence_series.append(divergence)
        procrustes_drift_series.append(procrustes_drift)
        knn_overlap_series.append(knn_overlap)
        trustworthiness_series.append(trustworthiness)
        continuity_series.append(continuity)
        retrieval_skew_series.append(retrieval_skew)
        path_entropy_series.append(path_entropy)
        recall_error_series.append(recall_error)

        ledger_flag = 1 if rng.random() < cfg["ledger_prob"] else 0
        domination_flag = 1 if rng.random() < cfg["domination_prob"] else 0
        reversal_flag = 1 if rng.random() < cfg["reversal_prob"] else 0
        policy_error_flag = 1 if rng.random() < cfg["policy_error_prob"] else 0
        conflict_flag = 1 if (ledger_flag or domination_flag or policy_error_flag or divergence > 0.08) else 0

        events["ledger_edit"].append(ledger_flag)
        events["domination_lock_in"].append(domination_flag)
        events["commitment_reversal"].append(reversal_flag)
        events["policy_error"].append(policy_error_flag)
        events["conflict_signature"].append(conflict_flag)
        events["residual_present"].append(1)
        events["precision_complete"].append(1)

        rollout.context_values.append(state)
        rollout.actions.append(1.0 if policy_error_flag == 0 else -1.0)

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "uncertainty": uncertainties,
            "divergence": divergence_series,
            "procrustes_drift": procrustes_drift_series,
            "knn_overlap": knn_overlap_series,
            "trustworthiness": trustworthiness_series,
            "continuity": continuity_series,
            "retrieval_skew": retrieval_skew_series,
            "path_entropy": path_entropy_series,
            "recall_error": recall_error_series,
        }
    )
    rollout.events.update(events)
    return rollout


def _jepa_anchor_ablation_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "ema_anchor_on": {
            "anchor_decay": 0.93,
            "predict_gain": 0.34,
            "predict_noise": 0.026,
            "shift_scale": 0.08,
            "consistency_threshold": 0.18,
            "drift_threshold": 0.35,
        },
        "ema_anchor_off": {
            "anchor_decay": 0.70,
            "predict_gain": 0.12,
            "predict_noise": 0.052,
            "shift_scale": 0.24,
            "consistency_threshold": 0.28,
            "drift_threshold": 0.12,
        },
    }[condition_name]

    rng = _rng_for("jepa_anchor_ablation", condition_name, seed)
    rollout = _init_rollout("jepa_anchor_ablation", condition_name, seed, steps)

    latent_errors: list[float] = []
    fast_delta: list[float] = []
    slow_delta: list[float] = []
    drift_values: list[float] = []

    events = {
        "rollout_consistent": [],
        "drift_event": [],
        "residual_present": [],
        "precision_complete": [],
    }

    latent = rng.uniform(-0.2, 0.2)
    anchor = latent
    prev_latent = latent
    prev_anchor = anchor

    for step in range(steps):
        shift = cfg["shift_scale"] if step >= (steps // 2) else 0.0
        latent += (0.045 * math.sin((step + 1) / 7.0)) + (shift / max(steps, 1)) + rng.gauss(0.0, 0.03)
        anchor = (cfg["anchor_decay"] * anchor) + ((1.0 - cfg["anchor_decay"]) * latent)

        fast = abs(latent - prev_latent)
        slow = abs(anchor - prev_anchor)

        pred_next = latent + (cfg["predict_gain"] * (latent - anchor)) + rng.gauss(0.0, cfg["predict_noise"])
        target_next = latent + (0.045 * math.sin((step + 2) / 7.0)) + (shift / max(steps, 1))
        error = abs(pred_next - target_next)
        drift = abs(latent - anchor)

        latent_errors.append(error)
        fast_delta.append(fast)
        slow_delta.append(slow)
        drift_values.append(drift)

        events["rollout_consistent"].append(1 if error < cfg["consistency_threshold"] else 0)
        events["drift_event"].append(1 if drift > cfg["drift_threshold"] else 0)
        events["residual_present"].append(1)
        events["precision_complete"].append(1)

        rollout.context_values.append(latent)
        rollout.actions.append(0.0)

        prev_latent = latent
        prev_anchor = anchor

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "fast_delta": fast_delta,
            "slow_delta": slow_delta,
            "drift_value": drift_values,
        }
    )
    rollout.events.update(events)
    return rollout


def _jepa_uncertainty_channels_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "deterministic_plus_dispersion": {
            "error_mu": 0.165,
            "error_sigma": 0.045,
            "spike_prob": 0.11,
            "spike_range": (0.06, 0.18),
            "uncertainty_scale": 0.72,
            "uncertainty_noise": 0.05,
            "dropout_prob": 0.07,
        },
        "explicit_uncertainty_head": {
            "error_mu": 0.148,
            "error_sigma": 0.038,
            "spike_prob": 0.08,
            "spike_range": (0.04, 0.12),
            "uncertainty_scale": 0.98,
            "uncertainty_noise": 0.025,
            "dropout_prob": 0.02,
        },
    }[condition_name]

    rng = _rng_for("jepa_uncertainty_channels", condition_name, seed)
    rollout = _init_rollout("jepa_uncertainty_channels", condition_name, seed, steps)

    latent_errors: list[float] = []
    uncertainty_series: list[float] = []

    events = {
        "uncertainty_available": [],
        "residual_present": [],
        "precision_complete": [],
    }

    state = rng.uniform(-0.15, 0.15)
    for step in range(steps):
        state += 0.03 * math.sin((step + 1) / 6.0) + rng.gauss(0.0, 0.018)
        error = max(0.0, rng.gauss(cfg["error_mu"], cfg["error_sigma"]))
        if rng.random() < cfg["spike_prob"]:
            error += rng.uniform(*cfg["spike_range"])

        uncertainty = max(0.0, (error * cfg["uncertainty_scale"]) + rng.gauss(0.0, cfg["uncertainty_noise"]))
        available = 0 if rng.random() < cfg["dropout_prob"] else 1

        latent_errors.append(error)
        uncertainty_series.append(uncertainty if available else float("nan"))

        events["uncertainty_available"].append(available)
        events["residual_present"].append(1)
        events["precision_complete"].append(available)

        rollout.context_values.append(state)
        rollout.actions.append(0.0)

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "uncertainty": uncertainty_series,
        }
    )
    rollout.events.update(events)
    return rollout


def _commit_dual_error_channels_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "single_error_stream": {
            "pre_noise": 0.16,
            "post_noise": 0.14,
            "channel_coupling": 0.92,
            "post_gain": 0.08,
            "reversal_cutoff": 0.22,
        },
        "pre_post_split_streams": {
            "pre_noise": 0.04,
            "post_noise": 0.03,
            "channel_coupling": 0.04,
            "post_gain": 0.62,
            "reversal_cutoff": 0.34,
        },
    }[condition_name]

    rng = _rng_for("commit_dual_error_channels", condition_name, seed)
    rollout = _init_rollout("commit_dual_error_channels", condition_name, seed, steps)

    pre_signal: list[float] = []
    pre_noise: list[float] = []
    post_signal: list[float] = []
    realized_signal: list[float] = []
    latent_errors: list[float] = []
    coupling_series: list[float] = []

    events = {
        "commitment_reversal": [],
        "residual_present": [],
        "precision_complete": [],
    }

    latent = rng.uniform(-0.2, 0.2)
    for step in range(steps):
        latent += 0.02 * math.sin((step + 1) / 8.0) + rng.gauss(0.0, 0.04)
        driver = latent + rng.gauss(0.0, 0.30)

        pre_n = rng.gauss(0.0, cfg["pre_noise"])
        pre = driver + pre_n

        realized = driver + rng.gauss(0.0, 0.22)

        post_n = rng.gauss(0.0, cfg["post_noise"])
        post_target = realized + (cfg["post_gain"] * (realized - pre))
        post = (cfg["channel_coupling"] * pre) + ((1.0 - cfg["channel_coupling"]) * post_target) + post_n

        pre_signal.append(pre)
        pre_noise.append(pre_n)
        post_signal.append(post)
        realized_signal.append(realized)
        coupling_series.append(cfg["channel_coupling"])

        latent_errors.append(abs(realized - pre))
        reversal = 1 if abs(post - realized) > cfg["reversal_cutoff"] else 0
        events["commitment_reversal"].append(reversal)
        events["residual_present"].append(1)
        events["precision_complete"].append(1)

        rollout.context_values.append(latent)
        rollout.actions.append(1.0 if post >= pre else -1.0)

    rollout.signals.update(
        {
            "pre_signal": pre_signal,
            "pre_noise": pre_noise,
            "post_signal": post_signal,
            "realized_signal": realized_signal,
            "latent_error": latent_errors,
            "channel_coupling": coupling_series,
        }
    )
    rollout.events.update(events)
    return rollout


def _tri_loop_arbitration_policy_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "veto_lattice": {
            "conflict_prob": 0.06,
            "alignment_base": 0.93,
            "override_prob": 0.08,
            "error_mu": 0.13,
            "error_sigma": 0.035,
        },
        "weighted_merge": {
            "conflict_prob": 0.12,
            "alignment_base": 0.86,
            "override_prob": 0.14,
            "error_mu": 0.16,
            "error_sigma": 0.042,
        },
        "mode_conditioned_precedence": {
            "conflict_prob": 0.08,
            "alignment_base": 0.90,
            "override_prob": 0.10,
            "error_mu": 0.145,
            "error_sigma": 0.038,
        },
    }[condition_name]

    rng = _rng_for("tri_loop_arbitration_policy", condition_name, seed)
    rollout = _init_rollout("tri_loop_arbitration_policy", condition_name, seed, steps)

    latent_errors: list[float] = []
    policy_alignment: list[float] = []
    gate_conflict: list[float] = []
    arbitration_override: list[float] = []

    events = {
        "gate_conflict": [],
        "policy_alignment": [],
        "arbitration_override": [],
        "residual_present": [],
        "precision_complete": [],
    }

    state = rng.uniform(-0.1, 0.1)
    for step in range(steps):
        state += 0.025 * math.sin((step + 1) / 6.0) + rng.gauss(0.0, 0.02)
        conflict = 1 if rng.random() < cfg["conflict_prob"] else 0
        override = 1 if rng.random() < cfg["override_prob"] else 0
        alignment_value = max(0.0, min(1.0, cfg["alignment_base"] - (0.12 * conflict) - (0.06 * override) + rng.gauss(0.0, 0.015)))
        error = max(0.0, rng.gauss(cfg["error_mu"] + (0.045 * conflict), cfg["error_sigma"]))

        latent_errors.append(error)
        policy_alignment.append(alignment_value)
        gate_conflict.append(float(conflict))
        arbitration_override.append(float(override))

        events["gate_conflict"].append(conflict)
        events["policy_alignment"].append(1 if alignment_value >= 0.85 else 0)
        events["arbitration_override"].append(override)
        events["residual_present"].append(1)
        events["precision_complete"].append(1)

        rollout.context_values.append(state)
        rollout.actions.append(1.0 if alignment_value >= 0.85 else -1.0)

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "policy_alignment": policy_alignment,
            "gate_conflict": gate_conflict,
            "arbitration_override": arbitration_override,
        }
    )
    rollout.events.update(events)
    return rollout


def _control_axis_ablation_rollout(condition_name: str, seed: int, steps: int) -> ToyRollout:
    cfg = {
        "full_axis": {
            "tonic_scale": 0.15,
            "phasic_scale": 0.07,
            "stability_base": 0.91,
            "policy_loss_prob": 0.05,
            "weights": [0.34, 0.33, 0.33],
            "error_mu": 0.14,
        },
        "reduced_axis": {
            "tonic_scale": 0.09,
            "phasic_scale": 0.11,
            "stability_base": 0.73,
            "policy_loss_prob": 0.17,
            "weights": [0.78, 0.18, 0.04],
            "error_mu": 0.19,
        },
    }[condition_name]

    rng = _rng_for("control_axis_ablation", condition_name, seed)
    rollout = _init_rollout("control_axis_ablation", condition_name, seed, steps)

    latent_errors: list[float] = []
    axis_stability: list[float] = []
    policy_loss: list[float] = []
    control_axis_weights: list[list[float]] = []

    events = {
        "policy_loss": [],
        "residual_present": [],
        "precision_complete": [],
    }

    tonic_state = rng.uniform(-0.05, 0.05)
    for step in range(steps):
        tonic_state += cfg["tonic_scale"] * math.sin((step + 1) / 11.0) + rng.gauss(0.0, 0.015)
        phasic = abs(cfg["phasic_scale"] * math.cos((step + 1) / 4.0) + rng.gauss(0.0, 0.01))
        stability = max(0.0, min(1.0, cfg["stability_base"] - (0.35 * phasic) + rng.gauss(0.0, 0.02)))
        loss = 1 if rng.random() < cfg["policy_loss_prob"] else 0
        error = max(0.0, rng.gauss(cfg["error_mu"] + (0.055 * loss), 0.04))

        latent_errors.append(error)
        axis_stability.append(stability)
        policy_loss.append(float(loss))
        control_axis_weights.append(list(cfg["weights"]))

        events["policy_loss"].append(loss)
        events["residual_present"].append(1)
        events["precision_complete"].append(1 if stability >= 0.60 else 0)

        rollout.context_values.append(tonic_state)
        rollout.actions.append(1.0 if stability >= 0.75 else -1.0)

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "axis_stability": axis_stability,
            "policy_loss": policy_loss,
            "control_axis_weights": control_axis_weights,
        }
    )
    rollout.events.update(events)
    return rollout


def run_toy_rollout(experiment_type: str, condition_name: str, seed: int, steps: int = 120) -> ToyRollout:
    if steps <= 0:
        raise ValueError("steps must be > 0")

    if experiment_type == "trajectory_integrity":
        return _trajectory_integrity_rollout(condition_name, seed, steps)
    if experiment_type == "jepa_anchor_ablation":
        return _jepa_anchor_ablation_rollout(condition_name, seed, steps)
    if experiment_type == "jepa_uncertainty_channels":
        return _jepa_uncertainty_channels_rollout(condition_name, seed, steps)
    if experiment_type == "commit_dual_error_channels":
        return _commit_dual_error_channels_rollout(condition_name, seed, steps)
    if experiment_type == "tri_loop_arbitration_policy":
        return _tri_loop_arbitration_policy_rollout(condition_name, seed, steps)
    if experiment_type == "control_axis_ablation":
        return _control_axis_ablation_rollout(condition_name, seed, steps)

    raise KeyError(f"Unsupported experiment_type: {experiment_type}")
