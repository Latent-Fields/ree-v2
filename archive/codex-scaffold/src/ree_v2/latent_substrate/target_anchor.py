"""Slow EMA target anchor for MECH-058 timescale separation checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmaTargetAnchor:
    decay: float = 0.98
    _state: list[float] | None = field(default=None, init=False, repr=False)

    def update(self, latent: list[float]) -> list[float]:
        if self._state is None:
            self._state = list(latent)
            return list(self._state)

        self._state = [
            self.decay * previous + (1.0 - self.decay) * current
            for previous, current in zip(self._state, latent)
        ]
        return list(self._state)
