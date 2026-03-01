"""Deterministic latent encoder used for qualification smoke runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class LatentEncoder:
    latent_dim: int = 8

    def encode(self, obs_t: Any, ctx_window: list[Any]) -> list[float]:
        seed_material = f"{obs_t}|{ctx_window}".encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        values: list[float] = []
        for i in range(self.latent_dim):
            byte_val = digest[i % len(digest)]
            values.append((byte_val / 255.0) * 2.0 - 1.0)
        return values
