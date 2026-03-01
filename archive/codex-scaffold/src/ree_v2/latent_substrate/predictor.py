"""Fast latent predictor separated from slow anchor state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class FastPredictor:
    horizon: int = 3

    def predict(
        self,
        z_t: list[float],
        trace: dict[str, Any],
        include_uncertainty: bool,
    ) -> dict[str, Any]:
        z_hat: list[list[float]] = []
        residuals: list[float] = []

        for step in range(1, self.horizon + 1):
            step_scale = 0.03 * step
            forecast = [value + step_scale for value in z_t]
            z_hat.append(forecast)
            residuals.extend(abs(f - v) for f, v in zip(forecast, z_t))

        pe_latent = {
            "mean": sum(residuals) / len(residuals),
            "p95": _quantile(residuals, 0.95),
            "by_mask": {
                mask_id: round((index + 1) * 0.01, 6)
                for index, mask_id in enumerate(trace.get("context_mask_ids", []))
            },
        }

        out: dict[str, Any] = {
            "z_hat": z_hat,
            "pe_latent": pe_latent,
        }

        if include_uncertainty:
            out["uncertainty_latent"] = {
                "dispersion": round(pe_latent["mean"] * 1.2, 6),
                "calibration_error": round(pe_latent["mean"] * 0.1, 6),
            }

        return out
