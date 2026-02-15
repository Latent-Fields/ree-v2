"""JEPA adapter signals exporter for experiment pack ingestion."""

from __future__ import annotations

from typing import Any


def build_adapter_signals(
    *,
    experiment_type: str,
    run_id: str,
    include_uncertainty: bool,
    include_action_token: bool,
    metrics_values: dict[str, float],
    adapter_name: str = "ree_v2_jepa_adapter",
    adapter_version: str = "bootstrap.v1",
    uncertainty_estimator: str = "dispersion",
) -> dict[str, Any]:
    signal_metrics: dict[str, float] = {
        "latent_prediction_error_mean": float(metrics_values.get("latent_prediction_error_mean", 0.0)),
        "latent_prediction_error_p95": float(metrics_values.get("latent_prediction_error_p95", 0.0)),
        "latent_residual_coverage_rate": float(metrics_values.get("latent_residual_coverage_rate", 0.97)),
        "precision_input_completeness_rate": float(metrics_values.get("precision_input_completeness_rate", 0.97)),
    }

    if include_uncertainty:
        signal_metrics["latent_uncertainty_calibration_error"] = float(
            metrics_values.get("latent_uncertainty_calibration_error", 0.12)
        )
    else:
        uncertainty_estimator = "none"

    if "commit_boundary_join_coverage_rate" in metrics_values:
        signal_metrics["commit_boundary_join_coverage_rate"] = float(
            metrics_values["commit_boundary_join_coverage_rate"]
        )
    if "tri_loop_trace_coverage_rate" in metrics_values:
        signal_metrics["tri_loop_trace_coverage_rate"] = float(metrics_values["tri_loop_trace_coverage_rate"])

    return {
        "schema_version": "jepa_adapter_signals/v1",
        "experiment_type": experiment_type,
        "run_id": run_id,
        "adapter": {
            "name": adapter_name,
            "version": adapter_version,
        },
        "stream_presence": {
            "z_t": True,
            "z_hat": True,
            "pe_latent": True,
            "uncertainty_latent": include_uncertainty,
            "trace_context_mask_ids": True,
            "trace_action_token": include_action_token,
            "trace_commit_boundary_token": True,
            "trace_tri_loop_gate": True,
            "trace_control_axis_telemetry": True,
        },
        "pe_latent_fields": ["mean", "p95", "by_mask"],
        "uncertainty_estimator": uncertainty_estimator,
        "signal_metrics": signal_metrics,
    }
