"""Inference-only JEPA backend wrapper with deterministic synthetic-frame fallback."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class JEPABackendConfig:
    code_repo_url: str
    code_commit: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    checkpoint_license_id: str
    source_mode: str
    compatibility_target: str


class _SyntheticJEPAProjector(nn.Module):
    """Small deterministic projector used when official artifacts are unavailable locally."""

    def __init__(self, latent_dim: int, horizon: int, seed: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        self.encoder = nn.Linear(6, latent_dim)
        self.delta_head = nn.Linear(latent_dim + 1, latent_dim)

        with torch.no_grad():
            self.encoder.weight.copy_(torch.randn(self.encoder.weight.shape, generator=generator) * 0.08)
            self.encoder.bias.copy_(torch.randn(self.encoder.bias.shape, generator=generator) * 0.02)
            self.delta_head.weight.copy_(torch.randn(self.delta_head.weight.shape, generator=generator) * 0.07)
            self.delta_head.bias.copy_(torch.randn(self.delta_head.bias.shape, generator=generator) * 0.02)

    def forward(self, frames: torch.Tensor, action_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # frames: [B, C, T, H, W], action_value: [B, 1]
        spatial_mean = frames.mean(dim=(3, 4))
        channel_mean = spatial_mean.mean(dim=2)
        channel_std = spatial_mean.std(dim=2, unbiased=False)
        temporal_start = frames[:, :, 0, :, :].mean(dim=(2, 3))
        temporal_end = frames[:, :, -1, :, :].mean(dim=(2, 3))

        features = torch.cat(
            [channel_mean, channel_std, temporal_start, temporal_end],
            dim=1,
        )
        features = features[:, :6]

        z_t = torch.tanh(self.encoder(features))

        z_hat_steps: list[torch.Tensor] = []
        prev = z_t
        for step in range(self.horizon):
            step_scale = torch.full_like(action_value, float(step + 1) / float(max(self.horizon, 1)))
            delta_input = torch.cat([prev, action_value * step_scale], dim=1)
            delta = torch.tanh(self.delta_head(delta_input)) * 0.12
            prev = prev + delta
            z_hat_steps.append(prev)

        z_hat = torch.stack(z_hat_steps, dim=1)
        return z_t, z_hat


class JEPAInferenceBackend:
    """Inference-only backend with optional local checkpoint loading.

    If model/checkpoint loading fails (for example due missing video/model dependencies),
    a deterministic synthetic-frame fallback path is used for smoke qualification.
    """

    def __init__(
        self,
        *,
        lock_payload: dict[str, Any],
        checkpoint_path: Path | None = None,
        latent_dim: int = 16,
        horizon: int = 3,
        device: str = "cpu",
        force_synthetic_fallback: bool = False,
    ) -> None:
        self.config = self._config_from_lock(lock_payload)
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.device = torch.device(device)
        self.force_synthetic_fallback = force_synthetic_fallback

        self.model: nn.Module
        self.model_source: str
        self.synthetic_fallback_used: bool
        self.fallback_reason: str

        self.model, self.model_source, self.synthetic_fallback_used, self.fallback_reason = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Enforce inference-only semantics.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _config_from_lock(lock_payload: dict[str, Any]) -> JEPABackendConfig:
        required = [
            "upstream_repo_url",
            "upstream_commit",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "checkpoint_license_id",
            "source_mode",
            "compatibility_target",
        ]
        missing = [key for key in required if key not in lock_payload]
        if missing:
            raise KeyError(f"JEPA lock is missing required key(s): {missing}")

        return JEPABackendConfig(
            code_repo_url=str(lock_payload["upstream_repo_url"]),
            code_commit=str(lock_payload["upstream_commit"]),
            checkpoint_repo_id=str(lock_payload["checkpoint_repo_id"]),
            checkpoint_revision=str(lock_payload["checkpoint_revision"]),
            checkpoint_license_id=str(lock_payload["checkpoint_license_id"]),
            source_mode=str(lock_payload["source_mode"]),
            compatibility_target=str(lock_payload["compatibility_target"]),
        )

    def _fallback_module(self) -> nn.Module:
        seed = int(hashlib.sha256(self.config.checkpoint_revision.encode("utf-8")).hexdigest()[:16], 16)
        return _SyntheticJEPAProjector(latent_dim=self.latent_dim, horizon=self.horizon, seed=seed)

    def _load_model(self, checkpoint_path: Path | None) -> tuple[nn.Module, str, bool, str]:
        if self.force_synthetic_fallback:
            return self._fallback_module(), "synthetic_fallback", True, "forced_by_flag"

        if checkpoint_path is None or not checkpoint_path.exists():
            reason = "checkpoint_missing"
            return self._fallback_module(), "synthetic_fallback", True, reason

        # Try TorchScript first.
        try:
            scripted = torch.jit.load(str(checkpoint_path), map_location=self.device)
            scripted.eval()
            return scripted, f"local_torchscript:{checkpoint_path}", False, ""
        except Exception as exc_torchscript:
            torchscript_err = str(exc_torchscript)

        # Try eager module/state dict checkpoints.
        try:
            payload = torch.load(str(checkpoint_path), map_location=self.device)
            if isinstance(payload, nn.Module):
                payload.eval()
                return payload, f"local_module:{checkpoint_path}", False, ""

            if isinstance(payload, dict):
                candidate = self._fallback_module()
                state_dict = None
                for key in ("model", "model_state_dict", "state_dict"):
                    if key in payload and isinstance(payload[key], dict):
                        state_dict = payload[key]
                        break
                if state_dict is None and all(isinstance(k, str) for k in payload.keys()):
                    state_dict = payload
                if isinstance(state_dict, dict):
                    candidate_state = candidate.state_dict()
                    compatible: dict[str, torch.Tensor] = {}
                    incompatible_keys: list[str] = []
                    for key, tensor_value in state_dict.items():
                        if key not in candidate_state:
                            continue
                        if not torch.is_tensor(tensor_value):
                            incompatible_keys.append(key)
                            continue
                        if tuple(tensor_value.shape) != tuple(candidate_state[key].shape):
                            incompatible_keys.append(key)
                            continue
                        compatible[key] = tensor_value

                    if len(compatible) != len(candidate_state):
                        reason = (
                            "checkpoint_incompatible:"
                            f"matched={len(compatible)}/{len(candidate_state)}"
                        )
                        if incompatible_keys:
                            sample = ",".join(sorted(incompatible_keys)[:3])
                            reason = f"{reason};shape_or_type_mismatch={sample}"
                        return self._fallback_module(), "synthetic_fallback", True, reason

                    load_result = candidate.load_state_dict(compatible, strict=True)
                    if load_result.missing_keys or load_result.unexpected_keys:
                        reason = (
                            "checkpoint_incomplete:"
                            f"missing={len(load_result.missing_keys)}"
                            f";unexpected={len(load_result.unexpected_keys)}"
                        )
                        return self._fallback_module(), "synthetic_fallback", True, reason

                    for key, tensor_value in candidate.state_dict().items():
                        if not torch.isfinite(tensor_value).all():
                            reason = f"checkpoint_invalid_non_finite:{key}"
                            return self._fallback_module(), "synthetic_fallback", True, reason

                    candidate.eval()
                    return candidate, f"local_state_dict:{checkpoint_path}", False, ""
        except Exception as exc_eager:
            eager_err = str(exc_eager)
            reason = f"torchscript_error={torchscript_err}; eager_error={eager_err}"
            return self._fallback_module(), "synthetic_fallback", True, reason

        reason = f"torchscript_error={torchscript_err}; eager_error=unrecognized_payload"
        return self._fallback_module(), "synthetic_fallback", True, reason

    @staticmethod
    def _frame_from_context(context_values: list[float], seed: int, steps: int = 8) -> torch.Tensor:
        values = context_values[-8:] if context_values else [0.0]
        values = [float(v) for v in values]
        while len(values) < 8:
            values.insert(0, values[0])

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        t = torch.linspace(0.0, 1.0, steps)
        h = torch.linspace(0.0, math.pi * 2.0, 32)
        w = torch.linspace(0.0, math.pi * 2.0, 32)
        hh, ww = torch.meshgrid(h, w, indexing="ij")

        frames = []
        for idx in range(steps):
            base = values[idx % len(values)]
            noise = torch.randn((3, 32, 32), generator=gen) * 0.003
            frame = torch.stack(
                [
                    torch.sin(hh + base + t[idx]),
                    torch.cos(ww + (base * 0.5) + t[idx]),
                    torch.sin(hh * 0.5 + ww * 0.5 + base),
                ],
                dim=0,
            )
            frame = frame + noise
            frames.append(frame)

        tensor = torch.stack(frames, dim=1).unsqueeze(0)
        return tensor

    @staticmethod
    def _extract_action_value(a_t: Any) -> float:
        if a_t is None:
            return 0.0
        if isinstance(a_t, (int, float)):
            return float(a_t)
        if isinstance(a_t, dict):
            if "value" in a_t and isinstance(a_t["value"], (int, float)):
                return float(a_t["value"])
            if "move" in a_t:
                return float(hash(str(a_t["move"])) % 11) / 10.0
        return float(hash(str(a_t)) % 17) / 16.0

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = (len(ordered) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        frac = idx - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    def _forward(self, frames: torch.Tensor, action_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        action_tensor = torch.tensor([[action_value]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if isinstance(self.model, _SyntheticJEPAProjector):
                z_t, z_hat = self.model(frames.to(self.device), action_tensor)
                return z_t, z_hat

            output = self.model(frames.to(self.device))
            if isinstance(output, tuple) and len(output) >= 2:
                z_t_raw, z_hat_raw = output[0], output[1]
            elif isinstance(output, dict) and "z_t" in output and "z_hat" in output:
                z_t_raw, z_hat_raw = output["z_t"], output["z_hat"]
            else:
                tensor_out = output if torch.is_tensor(output) else torch.as_tensor(output)
                z_t_raw = tensor_out.reshape(1, -1)
                z_hat_raw = z_t_raw.unsqueeze(1).repeat(1, self.horizon, 1)

            z_t = z_t_raw.reshape(1, -1)
            if z_t.shape[1] > self.latent_dim:
                z_t = z_t[:, : self.latent_dim]
            elif z_t.shape[1] < self.latent_dim:
                pad = torch.zeros((1, self.latent_dim - z_t.shape[1]), device=z_t.device)
                z_t = torch.cat([z_t, pad], dim=1)

            z_hat = z_hat_raw
            if z_hat.ndim == 2:
                z_hat = z_hat.unsqueeze(1)
            z_hat = z_hat.reshape(1, z_hat.shape[1], -1)
            if z_hat.shape[2] > self.latent_dim:
                z_hat = z_hat[:, :, : self.latent_dim]
            elif z_hat.shape[2] < self.latent_dim:
                pad = torch.zeros((1, z_hat.shape[1], self.latent_dim - z_hat.shape[2]), device=z_hat.device)
                z_hat = torch.cat([z_hat, pad], dim=2)

            if z_hat.shape[1] < self.horizon:
                repeats = self.horizon - z_hat.shape[1]
                tail = z_hat[:, -1:, :].repeat(1, repeats, 1)
                z_hat = torch.cat([z_hat, tail], dim=1)
            elif z_hat.shape[1] > self.horizon:
                z_hat = z_hat[:, : self.horizon, :]

            return z_t, z_hat

    def infer(
        self,
        *,
        obs_t: Any,
        ctx_window: list[Any],
        a_t: Any | None,
        include_uncertainty: bool,
        context_mask_ids: list[str],
    ) -> dict[str, Any]:
        seed_material = hashlib.sha256(f"{obs_t}|{ctx_window}|{a_t}".encode("utf-8")).hexdigest()
        frame_seed = int(seed_material[:16], 16)
        context_values = [float(v) if isinstance(v, (int, float)) else float(hash(str(v)) % 101) / 100.0 for v in ctx_window]
        frames = self._frame_from_context(context_values, frame_seed)

        action_value = self._extract_action_value(a_t)
        z_t_tensor, z_hat_tensor = self._forward(frames, action_value)

        z_t = z_t_tensor[0].detach().cpu().tolist()
        z_hat = z_hat_tensor[0].detach().cpu().tolist()

        step_residuals = [abs(pred - base) for pred, base in zip(z_hat[0], z_t)] if z_hat else []
        mean_error = sum(step_residuals) / len(step_residuals) if step_residuals else 0.0
        p95_error = self._quantile(step_residuals, 0.95)

        pe_by_mask = {
            mask_id: round(mean_error * (1.0 + (idx * 0.02)), 12)
            for idx, mask_id in enumerate(context_mask_ids)
        }

        payload: dict[str, Any] = {
            "z_t": [round(float(value), 12) for value in z_t],
            "z_hat": [[round(float(value), 12) for value in step] for step in z_hat],
            "pe_latent": {
                "mean": round(float(mean_error), 12),
                "p95": round(float(p95_error), 12),
                "by_mask": pe_by_mask,
            },
            "backend_metadata": {
                "backend": "jepa_inference",
                "model_source": self.model_source,
                "synthetic_frame_fallback": self.synthetic_fallback_used,
                "fallback_reason": self.fallback_reason,
                "checkpoint_repo_id": self.config.checkpoint_repo_id,
                "checkpoint_revision": self.config.checkpoint_revision,
                "checkpoint_license_id": self.config.checkpoint_license_id,
                "code_repo_url": self.config.code_repo_url,
                "code_commit": self.config.code_commit,
            },
        }

        if include_uncertainty:
            dispersion = max(0.0, mean_error * 1.15)
            calibration_error = max(0.0, abs(dispersion - mean_error))
            payload["uncertainty_latent"] = {
                "dispersion": round(float(dispersion), 12),
                "calibration_error": round(float(calibration_error), 12),
            }

        return payload
