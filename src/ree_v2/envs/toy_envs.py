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
        },
        "trajectory_first_ablated": {
            "obs_noise": 0.055,
            "error_sigma": 0.073,
            "ledger_prob": 0.060,
            "domination_prob": 0.035,
            "reversal_prob": 0.118,
            "divergence_base": 0.081,
            "uncertainty_scale": 0.88,
        },
    }[condition_name]

    rng = _rng_for("trajectory_integrity", condition_name, seed)
    rollout = _init_rollout("trajectory_integrity", condition_name, seed, steps)

    latent_errors: list[float] = []
    uncertainties: list[float] = []
    divergence_series: list[float] = []

    events = {
        "ledger_edit": [],
        "domination_lock_in": [],
        "commitment_reversal": [],
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

        latent_errors.append(error)
        uncertainties.append(uncertainty)
        divergence_series.append(divergence)

        events["ledger_edit"].append(1 if rng.random() < cfg["ledger_prob"] else 0)
        events["domination_lock_in"].append(1 if rng.random() < cfg["domination_prob"] else 0)
        events["commitment_reversal"].append(1 if rng.random() < cfg["reversal_prob"] else 0)
        events["residual_present"].append(1)
        events["precision_complete"].append(1)

        rollout.context_values.append(state)
        rollout.actions.append(0.0)

    rollout.signals.update(
        {
            "latent_error": latent_errors,
            "uncertainty": uncertainties,
            "divergence": divergence_series,
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
            "pre_noise": 0.12,
            "post_noise": 0.11,
            "channel_coupling": 0.88,
            "post_gain": 0.12,
            "reversal_cutoff": 0.26,
        },
        "pre_post_split_streams": {
            "pre_noise": 0.06,
            "post_noise": 0.05,
            "channel_coupling": 0.12,
            "post_gain": 0.48,
            "reversal_cutoff": 0.44,
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
        post = (cfg["channel_coupling"] * pre) + ((1.0 - cfg["channel_coupling"]) * (realized + cfg["post_gain"])) + post_n

        pre_signal.append(pre)
        pre_noise.append(pre_n)
        post_signal.append(post)
        realized_signal.append(realized)
        coupling_series.append(cfg["channel_coupling"])

        latent_errors.append(abs(realized - pre))
        reversal = 1 if abs(pre - post) > cfg["reversal_cutoff"] else 0
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

    raise KeyError(f"Unsupported experiment_type: {experiment_type}")
