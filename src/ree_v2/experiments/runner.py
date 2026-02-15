"""Executable toy qualification runner producing contract-compliant run packs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ree_v2.envs import run_toy_rollout
from ree_v2.experiments.profiles import evaluate_failure_signatures, get_profile
from ree_v2.hooks.emitter import emit_planned_stub_hooks, emit_v2_hooks
from ree_v2.latent_substrate.encoder import LatentEncoder
from ree_v2.latent_substrate.jepa_inference_backend import JEPAInferenceBackend
from ree_v2.latent_substrate.predictor import FastPredictor
from ree_v2.latent_substrate.target_anchor import EmaTargetAnchor
from ree_v2.sensor_adapter.adapter import SensorAdapter
from ree_v2.signal_export.adapter_signals import build_adapter_signals
from ree_v2.signal_export.metrics_export import build_metrics_payload

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_NAME = "ree-v2-qualification-harness"
RUNNER_VERSION = "toy_env_runner.v1"
_CHECKPOINT_DIGEST_CACHE: dict[str, tuple[str, int]] = {}


@dataclass(frozen=True)
class RunExecutionResult:
    experiment_type: str
    condition_name: str
    seed: int
    backend: str
    run_id: str
    run_dir: Path
    status: str
    metrics_values: dict[str, float]
    manifest: dict[str, Any]
    adapter_signals: dict[str, Any]
    hook_payloads: dict[str, Any]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    var = sum((value - mu) ** 2 for value in values) / len(values)
    return math.sqrt(var)


def _pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mean_x = _mean(x)
    mean_y = _mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return 0.0
    return num / (den_x * den_y)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _parse_timestamp_utc(timestamp_utc: str | None) -> dt.datetime:
    if timestamp_utc is None:
        return _utc_now()
    parsed = dt.datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).replace(microsecond=0)


def _git_value(args: list[str], fallback: str = "unknown") -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return fallback


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_digest(path: Path) -> tuple[str, int]:
    cache_key = str(path.resolve())
    cached = _CHECKPOINT_DIGEST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    digest = _sha256_file(path)
    size = path.stat().st_size
    _CHECKPOINT_DIGEST_CACHE[cache_key] = (digest, size)
    return digest, size


def _non_nan(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def _compute_metrics(experiment_type: str, rollout: Any) -> dict[str, float]:
    errors = rollout.signals.get("latent_error", [])
    metrics: dict[str, float] = {
        "latent_prediction_error_mean": round(_mean(errors), 12),
        "latent_prediction_error_p95": round(_quantile(errors, 0.95), 12),
        "latent_residual_coverage_rate": round(_mean([float(v) for v in rollout.events.get("residual_present", [1])]), 12),
        "precision_input_completeness_rate": round(
            _mean([float(v) for v in rollout.events.get("precision_complete", [1])]),
            12,
        ),
        "fatal_error_count": 0.0,
    }

    if experiment_type == "trajectory_integrity":
        uncertainties = rollout.signals.get("uncertainty", [])
        calibration = _mean([abs(u - e) for e, u in zip(errors, uncertainties)])
        metrics.update(
            {
                "ledger_edit_detected_count": float(sum(rollout.events.get("ledger_edit", []))),
                "explanation_policy_divergence_rate": round(_mean(rollout.signals.get("divergence", [])), 12),
                "domination_lock_in_events": float(sum(rollout.events.get("domination_lock_in", []))),
                "commitment_reversal_rate": round(
                    _mean([float(v) for v in rollout.events.get("commitment_reversal", [])]),
                    12,
                ),
                "latent_uncertainty_calibration_error": round(calibration, 12),
            }
        )

    elif experiment_type == "jepa_anchor_ablation":
        fast_delta = rollout.signals.get("fast_delta", [])
        slow_delta = rollout.signals.get("slow_delta", [])
        ratio = _mean(fast_delta) / max(_mean(slow_delta), 1e-9)
        ratio = min(3.0, ratio)
        metrics.update(
            {
                "latent_rollout_consistency_rate": round(
                    _mean([float(v) for v in rollout.events.get("rollout_consistent", [])]),
                    12,
                ),
                "e1_e2_timescale_separation_ratio": round(ratio, 12),
                "representation_drift_rate": round(
                    _mean([float(v) for v in rollout.events.get("drift_event", [])]),
                    12,
                ),
            }
        )

    elif experiment_type == "jepa_uncertainty_channels":
        uncertainty_values = rollout.signals.get("uncertainty", [])
        pairs = [(e, u) for e, u in zip(errors, uncertainty_values) if not math.isnan(u)]
        available_unc = [u for _e, u in pairs]
        available_err = [e for e, _u in pairs]

        max_u = max(available_unc) if available_unc else 1.0
        max_e = max(available_err) if available_err else 1.0
        calibration = _mean([abs((u / max_u) - (e / max_e)) for e, u in pairs]) if pairs else 0.0

        high_error_threshold = _quantile(errors, 0.70)
        high_uncertainty_threshold = _quantile(available_unc, 0.65) if available_unc else float("inf")
        high_error_indices = [idx for idx, value in enumerate(errors) if value >= high_error_threshold]
        covered = 0
        for idx in high_error_indices:
            unc = uncertainty_values[idx]
            if not math.isnan(unc) and unc >= high_uncertainty_threshold:
                covered += 1
        coverage = covered / max(len(high_error_indices), 1)

        metrics.update(
            {
                "latent_uncertainty_calibration_error": round(calibration, 12),
                "uncertainty_coverage_rate": round(coverage, 12),
            }
        )

    elif experiment_type == "commit_dual_error_channels":
        pre_signal = rollout.signals.get("pre_signal", [])
        pre_noise = rollout.signals.get("pre_noise", [])
        post_signal = rollout.signals.get("post_signal", [])
        realized_signal = rollout.signals.get("realized_signal", [])
        coupling_series = rollout.signals.get("channel_coupling", [1.0])

        snr = _mean([abs(value) for value in pre_signal]) / max(_stddev(pre_noise), 1e-9)
        coupling_mean = _mean(coupling_series)
        gain = max(
            0.0,
            abs(_pearson_corr(post_signal, realized_signal))
            - abs(_pearson_corr(pre_signal, realized_signal))
            + ((1.0 - coupling_mean) * 0.35),
        )
        leakage = abs(_pearson_corr(pre_signal, post_signal)) * coupling_mean

        metrics.update(
            {
                "pre_commit_error_signal_to_noise": round(snr, 12),
                "post_commit_error_attribution_gain": round(gain, 12),
                "cross_channel_leakage_rate": round(min(1.0, leakage), 12),
                "commitment_reversal_rate": round(
                    _mean([float(v) for v in rollout.events.get("commitment_reversal", [])]),
                    12,
                ),
            }
        )

    else:
        raise KeyError(f"Unsupported experiment_type for metrics: {experiment_type}")

    return metrics


def _build_internal_backend_outputs(
    *,
    include_uncertainty: bool,
    include_action_token: bool,
    context_values: list[float],
    actions: list[float],
) -> dict[str, Any]:
    adapter = SensorAdapter(context_window=4)
    encoder = LatentEncoder(latent_dim=8)
    anchor = EmaTargetAnchor(decay=0.98)
    predictor = FastPredictor(horizon=3)

    context_window = [round(value, 6) for value in context_values[-4:]]
    action_payload: dict[str, float] | None = None
    if include_action_token and actions:
        action_payload = {"value": round(actions[-1], 6)}

    ingress = adapter.adapt(
        obs_t={"value": round(context_values[-1] if context_values else 0.0, 6)},
        ctx_window=context_window,
        a_t=action_payload,
        mode_tags=["qualification", "toy_env"],
    )

    z_t = encoder.encode(ingress["obs_t"], ingress["ctx_window"])
    anchor_state = anchor.update(z_t)
    prediction = predictor.predict(z_t, ingress["trace"], include_uncertainty=include_uncertainty)

    return {
        "z_t": anchor_state,
        "z_hat": prediction["z_hat"],
        "pe_latent": prediction["pe_latent"],
        "uncertainty_latent": prediction.get("uncertainty_latent"),
        "context_mask_ids": ingress["trace"]["context_mask_ids"],
        "action_token": ingress["trace"].get("action_token"),
        "backend_metadata": {
            "backend": "internal_minimal",
            "model_source": "ree_v2_internal_minimal",
            "synthetic_frame_fallback": False,
            "fallback_reason": "",
        },
    }


def _build_jepa_inference_outputs(
    *,
    include_uncertainty: bool,
    include_action_token: bool,
    context_values: list[float],
    actions: list[float],
    lock: dict[str, Any],
    jepa_checkpoint_path: Path | None,
    force_synthetic_frames: bool,
) -> dict[str, Any]:
    adapter = SensorAdapter(context_window=4)
    context_window = [round(value, 6) for value in context_values[-4:]]
    action_payload: dict[str, float] | None = None
    if include_action_token and actions:
        action_payload = {"value": round(actions[-1], 6)}

    ingress = adapter.adapt(
        obs_t={"value": round(context_values[-1] if context_values else 0.0, 6)},
        ctx_window=context_window,
        a_t=action_payload,
        mode_tags=["qualification", "toy_env", "jepa_inference"],
    )

    backend = JEPAInferenceBackend(
        lock_payload=lock,
        checkpoint_path=jepa_checkpoint_path,
        latent_dim=16,
        horizon=3,
        device="cpu",
        force_synthetic_fallback=force_synthetic_frames,
    )
    result = backend.infer(
        obs_t=ingress["obs_t"],
        ctx_window=ingress["ctx_window"],
        a_t=ingress["a_t"],
        include_uncertainty=include_uncertainty,
        context_mask_ids=ingress["trace"]["context_mask_ids"],
    )

    return {
        "z_t": result["z_t"],
        "z_hat": result["z_hat"],
        "pe_latent": result["pe_latent"],
        "uncertainty_latent": result.get("uncertainty_latent"),
        "context_mask_ids": ingress["trace"]["context_mask_ids"],
        "action_token": ingress["trace"].get("action_token"),
        "backend_metadata": result.get("backend_metadata", {}),
    }


def _build_hook_payloads_from_backend_outputs(
    *,
    include_uncertainty: bool,
    include_action_token: bool,
    backend_outputs: dict[str, Any],
) -> dict[str, Any]:
    v2_hooks = emit_v2_hooks(
        z_t=backend_outputs["z_t"],
        z_hat=backend_outputs["z_hat"],
        pe_latent=backend_outputs["pe_latent"],
        context_mask_ids=backend_outputs["context_mask_ids"],
        include_uncertainty=include_uncertainty,
        uncertainty_latent=backend_outputs.get("uncertainty_latent"),
        include_action_token=include_action_token,
        action_token=backend_outputs.get("action_token"),
    )

    return {
        "required": v2_hooks,
        "planned_stubs": emit_planned_stub_hooks(),
    }


def _jepa_lock() -> dict[str, Any]:
    return _load_json(REPO_ROOT / "third_party" / "jepa_sources.lock.v1.json")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _checkpoint_verification(
    *,
    backend: str,
    backend_metadata: dict[str, Any],
    lock: dict[str, Any],
    jepa_checkpoint_path: Path | None,
) -> dict[str, Any]:
    expected_filename = str(lock.get("checkpoint_filename", ""))
    expected_sha = str(lock.get("checkpoint_sha256", ""))
    expected_size = _positive_int(lock.get("checkpoint_size_bytes"))
    fallback_used = bool(backend_metadata.get("synthetic_frame_fallback", False))

    payload: dict[str, Any] = {
        "jepa_checkpoint_filename": expected_filename,
        "jepa_checkpoint_sha256": expected_sha,
        "jepa_checkpoint_size_bytes": expected_size if expected_size is not None else 0,
        "jepa_checkpoint_verified": False,
        "jepa_checkpoint_verification_reason": "not_jepa_inference_backend",
    }
    if backend != "jepa_inference":
        return payload

    if jepa_checkpoint_path is None or not jepa_checkpoint_path.exists():
        payload["jepa_checkpoint_verification_reason"] = str(
            backend_metadata.get("fallback_reason", "checkpoint_missing")
        )
        return payload

    observed_sha, observed_size = _checkpoint_digest(jepa_checkpoint_path)
    observed_filename = jepa_checkpoint_path.name
    payload["jepa_checkpoint_filename"] = observed_filename
    payload["jepa_checkpoint_sha256"] = observed_sha
    payload["jepa_checkpoint_size_bytes"] = observed_size

    reasons: list[str] = []
    if not expected_filename or not expected_sha or expected_size is None:
        reasons.append("lock_integrity_fields_incomplete")
    else:
        if observed_filename != expected_filename:
            reasons.append("filename_mismatch")
        if observed_sha.lower() != expected_sha.lower():
            reasons.append("sha256_mismatch")
        if observed_size != expected_size:
            reasons.append("size_mismatch")
    if fallback_used:
        reasons.append(str(backend_metadata.get("fallback_reason", "backend_fallback_used")))

    if reasons:
        payload["jepa_checkpoint_verified"] = False
        payload["jepa_checkpoint_verification_reason"] = ",".join(reasons)
    else:
        payload["jepa_checkpoint_verified"] = True
        payload["jepa_checkpoint_verification_reason"] = "verified_against_lock"
    return payload


def _fail_hard_status(metrics: dict[str, float], failure_signatures: list[str]) -> str:
    if metrics.get("fatal_error_count", 0.0) > 0:
        return "FAIL"
    if failure_signatures:
        return "FAIL"
    return "PASS"


def _effective_evidence_direction(expected_direction: str, status: str) -> str:
    if status == "FAIL" and expected_direction == "supports":
        return "weakens"
    return expected_direction


def _channel_isolation_trace(experiment_type: str, rollout: Any) -> dict[str, float] | None:
    if experiment_type != "commit_dual_error_channels":
        return None

    pre_signal = rollout.signals.get("pre_signal", [])
    post_signal = rollout.signals.get("post_signal", [])
    realized_signal = rollout.signals.get("realized_signal", [])
    pre_noise = rollout.signals.get("pre_noise", [])
    coupling_series = rollout.signals.get("channel_coupling", [0.0])

    return {
        "corr_pre_post": round(_pearson_corr(pre_signal, post_signal), 12),
        "corr_pre_realized": round(_pearson_corr(pre_signal, realized_signal), 12),
        "corr_post_realized": round(_pearson_corr(post_signal, realized_signal), 12),
        "coupling_mean": round(_mean(coupling_series), 12),
        "pre_noise_std": round(_stddev(pre_noise), 12),
    }


def _summary_text(
    *,
    experiment_type: str,
    condition_name: str,
    seed: int,
    status: str,
    backend: str,
    metrics: dict[str, float],
    failure_signatures: list[str],
    channel_trace: dict[str, float] | None,
    hook_payloads: dict[str, Any],
    backend_metadata: dict[str, Any],
    checkpoint_verification: dict[str, Any],
) -> str:
    metric_lines = []
    for key in sorted(metrics):
        if key == "fatal_error_count":
            continue
        metric_lines.append(f"- {key}: `{metrics[key]}`")

    hook_ids = sorted(hook_payloads["required"].keys()) + sorted(hook_payloads["planned_stubs"].keys())
    lines = [
        f"# {experiment_type} run summary",
        "",
        f"- condition: `{condition_name}`",
        f"- seed: `{seed}`",
        f"- status: `{status}`",
        f"- backend: `{backend}`",
        f"- failure_signatures: `{', '.join(failure_signatures) if failure_signatures else 'none'}`",
        f"- emitted_hooks: `{', '.join(hook_ids)}`",
        f"- model_source: `{backend_metadata.get('model_source', 'n/a')}`",
        f"- synthetic_frame_fallback: `{backend_metadata.get('synthetic_frame_fallback', False)}`",
        f"- checkpoint_verified: `{checkpoint_verification.get('jepa_checkpoint_verified', False)}`",
        f"- checkpoint_verify_reason: `{checkpoint_verification.get('jepa_checkpoint_verification_reason', 'n/a')}`",
        "",
        "## Metrics",
        *metric_lines,
    ]

    if channel_trace is not None:
        lines.extend(
            [
                "",
                "## Channel Provenance/Isolation",
                f"- corr_pre_post: `{channel_trace['corr_pre_post']}`",
                f"- corr_pre_realized: `{channel_trace['corr_pre_realized']}`",
                f"- corr_post_realized: `{channel_trace['corr_post_realized']}`",
                f"- coupling_mean: `{channel_trace['coupling_mean']}`",
                f"- pre_noise_std: `{channel_trace['pre_noise_std']}`",
            ]
        )

    return "\n".join(lines)


def execute_profile_condition(
    *,
    experiment_type: str,
    condition_name: str,
    seed: int,
    backend: str = "internal_minimal",
    steps: int = 120,
    runs_root: Path | None = None,
    run_id: str | None = None,
    timestamp_utc: str | None = None,
    jepa_checkpoint_path: Path | None = None,
    force_synthetic_frames: bool = False,
    require_real_jepa: bool = False,
    write: bool = True,
) -> RunExecutionResult:
    profile = get_profile(experiment_type)
    condition = next(item for item in profile.conditions if item.name == condition_name)

    ts = _parse_timestamp_utc(timestamp_utc)
    timestamp_str = ts.isoformat().replace("+00:00", "Z")
    run_stamp = ts.strftime("%Y-%m-%dT%H%M%SZ")

    if run_id is None:
        run_id = f"{run_stamp}_{experiment_type.replace('_', '-')}_seed{seed}_{condition_name}_toyenv_{backend}"

    rollout = run_toy_rollout(experiment_type, condition_name, seed, steps=steps)
    metrics_values = _compute_metrics(experiment_type, rollout)

    lock = _jepa_lock()
    patch_hash = _stable_hash(lock.get("patch_set", []))[:16]

    if backend == "internal_minimal":
        backend_outputs = _build_internal_backend_outputs(
            include_uncertainty=condition.include_uncertainty,
            include_action_token=condition.include_action_token,
            context_values=rollout.context_values,
            actions=rollout.actions,
        )
    elif backend == "jepa_inference":
        backend_outputs = _build_jepa_inference_outputs(
            include_uncertainty=condition.include_uncertainty,
            include_action_token=condition.include_action_token,
            context_values=rollout.context_values,
            actions=rollout.actions,
            lock=lock,
            jepa_checkpoint_path=jepa_checkpoint_path,
            force_synthetic_frames=force_synthetic_frames,
        )
    else:
        raise KeyError(f"Unsupported backend '{backend}'. Expected internal_minimal|jepa_inference")

    hook_payloads = _build_hook_payloads_from_backend_outputs(
        include_uncertainty=condition.include_uncertainty,
        include_action_token=condition.include_action_token,
        backend_outputs=backend_outputs,
    )

    failure_signatures = evaluate_failure_signatures(experiment_type, metrics_values)
    status = _fail_hard_status(metrics_values, failure_signatures)
    evidence_direction = _effective_evidence_direction(condition.evidence_direction, status)
    channel_trace = _channel_isolation_trace(experiment_type, rollout)
    backend_metadata = backend_outputs.get("backend_metadata", {})
    checkpoint_verification = _checkpoint_verification(
        backend=backend,
        backend_metadata=backend_metadata,
        lock=lock,
        jepa_checkpoint_path=jepa_checkpoint_path,
    )
    if (
        backend == "jepa_inference"
        and require_real_jepa
        and bool(backend_metadata.get("synthetic_frame_fallback", False))
    ):
        reason = str(backend_metadata.get("fallback_reason", "unknown"))
        source = str(backend_metadata.get("model_source", "unknown"))
        raise RuntimeError(
            "Real JEPA checkpoint required but backend used synthetic fallback "
            f"(reason={reason}, model_source={source}). "
            "Provide --jepa-checkpoint-path to a valid local checkpoint and retry."
        )
    if (
        backend == "jepa_inference"
        and require_real_jepa
        and not bool(checkpoint_verification.get("jepa_checkpoint_verified", False))
    ):
        reason = str(checkpoint_verification.get("jepa_checkpoint_verification_reason", "unknown"))
        raise RuntimeError(
            "Real JEPA checkpoint required but lock verification failed "
            f"(reason={reason})."
        )

    config_hash = _stable_hash(
        {
            "experiment_type": experiment_type,
            "condition": condition_name,
            "seed": seed,
            "steps": steps,
            "runner_version": RUNNER_VERSION,
            "driver": "deterministic_toy_env_v1",
            "backend": backend,
            "jepa_checkpoint_revision": lock.get("checkpoint_revision", ""),
            "synthetic_frame_fallback": backend_metadata.get("synthetic_frame_fallback", False),
            "jepa_checkpoint_verified": checkpoint_verification.get("jepa_checkpoint_verified", False),
            "require_real_jepa": require_real_jepa,
        }
    )[:12]

    source_commit = _git_value(["rev-parse", "HEAD"])
    source_branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], fallback="unknown")

    manifest: dict[str, Any] = {
        "schema_version": "experiment_pack/v1",
        "experiment_type": experiment_type,
        "run_id": run_id,
        "status": status,
        "timestamp_utc": timestamp_str,
        "source_repo": {
            "name": "ree-v2",
            "commit": source_commit,
            "branch": source_branch,
        },
        "runner": {
            "name": RUNNER_NAME,
            "version": f"{RUNNER_VERSION}+{backend}",
        },
        "scenario": {
            "name": experiment_type,
            "condition": condition_name,
            "seed": seed,
            "seed_cohort": list(profile.default_seeds),
            "config_hash": config_hash,
            "trajectory_steps": steps,
            "execution_driver": "deterministic_toy_env_v1",
            "backend": backend,
            "jepa_source_mode": lock["source_mode"],
            "jepa_source_commit": lock["upstream_commit"],
            "jepa_patch_set_hash": patch_hash,
            "jepa_checkpoint_repo_id": lock.get("checkpoint_repo_id", ""),
            "jepa_checkpoint_revision": lock.get("checkpoint_revision", ""),
            "jepa_checkpoint_license_id": lock.get("checkpoint_license_id", ""),
            "jepa_model_source": backend_metadata.get("model_source", ""),
            "synthetic_frame_fallback": backend_metadata.get("synthetic_frame_fallback", False),
            "synthetic_frame_fallback_reason": backend_metadata.get("fallback_reason", ""),
            "jepa_checkpoint_filename": checkpoint_verification.get("jepa_checkpoint_filename", ""),
            "jepa_checkpoint_sha256": checkpoint_verification.get("jepa_checkpoint_sha256", ""),
            "jepa_checkpoint_size_bytes": checkpoint_verification.get("jepa_checkpoint_size_bytes", 0),
            "jepa_checkpoint_verified": checkpoint_verification.get("jepa_checkpoint_verified", False),
            "jepa_checkpoint_verification_reason": checkpoint_verification.get(
                "jepa_checkpoint_verification_reason",
                "",
            ),
            "require_real_jepa": require_real_jepa,
        },
        "stop_criteria_version": "stop_criteria/v1",
        "claim_ids_tested": [profile.claim_id],
        "evidence_class": profile.evidence_class,
        "evidence_direction": evidence_direction,
        "failure_signatures": failure_signatures,
        "artifacts": {
            "metrics_path": "metrics.json",
            "summary_path": "summary.md",
            "adapter_signals_path": "jepa_adapter_signals.v1.json",
        },
    }
    if channel_trace is not None:
        manifest["artifacts"]["traces_dir"] = "traces"

    metrics_payload = build_metrics_payload(metrics_values)

    estimator = "dispersion"
    if condition_name == "explicit_uncertainty_head":
        estimator = "head"
    if not condition.include_uncertainty:
        estimator = "none"

    adapter_name = "ree_v2_toy_jepa_adapter"
    if backend == "jepa_inference":
        adapter_name = "ree_v2_jepa_inference_adapter"

    adapter_signals = build_adapter_signals(
        experiment_type=experiment_type,
        run_id=run_id,
        include_uncertainty=condition.include_uncertainty,
        include_action_token=condition.include_action_token,
        metrics_values=metrics_values,
        adapter_name=adapter_name,
        adapter_version=f"{RUNNER_VERSION}+{backend}",
        uncertainty_estimator=estimator,
    )

    summary = _summary_text(
        experiment_type=experiment_type,
        condition_name=condition_name,
        seed=seed,
        status=status,
        backend=backend,
        metrics=metrics_values,
        failure_signatures=failure_signatures,
        channel_trace=channel_trace,
        hook_payloads=hook_payloads,
        backend_metadata=backend_metadata,
        checkpoint_verification=checkpoint_verification,
    )

    base_root = runs_root or (REPO_ROOT / "evidence" / "experiments")
    run_dir = base_root / experiment_type / "runs" / run_id

    if write:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "metrics.json", metrics_payload)
        _write_json(run_dir / "jepa_adapter_signals.v1.json", adapter_signals)
        if channel_trace is not None:
            traces_dir = run_dir / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            _write_json(traces_dir / "channel_isolation.v1.json", channel_trace)
        (run_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")

    return RunExecutionResult(
        experiment_type=experiment_type,
        condition_name=condition_name,
        seed=seed,
        backend=backend,
        run_id=run_id,
        run_dir=run_dir,
        status=status,
        metrics_values=metrics_values,
        manifest=manifest,
        adapter_signals=adapter_signals,
        hook_payloads=hook_payloads,
    )
