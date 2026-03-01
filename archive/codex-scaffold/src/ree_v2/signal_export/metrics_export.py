"""Experiment pack metrics exporter."""

from __future__ import annotations


def build_metrics_payload(values: dict[str, float]) -> dict[str, object]:
    numeric_values: dict[str, float] = {}
    for key, value in values.items():
        numeric_values[key] = float(value)
    return {
        "schema_version": "experiment_pack_metrics/v1",
        "values": numeric_values,
    }
