"""Maps raw observations into substrate-friendly ingress payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class SensorAdapter:
    """Minimal adapter honoring the IMPL-022 ingress contract."""

    context_window: int = 4

    def adapt(
        self,
        obs_t: Any,
        ctx_window: list[Any],
        a_t: Any | None = None,
        mode_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        if len(ctx_window) > self.context_window:
            ctx_window = ctx_window[-self.context_window :]

        context_mask_ids = [self._stable_token(f"ctx:{index}:{value}") for index, value in enumerate(ctx_window)]
        action_token = self._stable_token(f"action:{a_t}") if a_t is not None else None

        return {
            "obs_t": obs_t,
            "ctx_window": ctx_window,
            "a_t": a_t,
            "mode_tags": mode_tags or [],
            "trace": {
                "context_mask_ids": context_mask_ids,
                "action_token": action_token,
            },
        }

    @staticmethod
    def _stable_token(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
