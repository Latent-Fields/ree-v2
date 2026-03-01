"""Utilities for reading hook registry contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HookEntry:
    hook_id: str
    hook_name: str
    tier: str
    status: str
    key_fields: tuple[str, ...]
    availability: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_registry_path() -> Path:
    return _repo_root() / "contracts" / "hook_registry.v1.json"


def load_hook_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or default_registry_path()
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_hook_entries(path: Path | None = None) -> list[HookEntry]:
    data = load_hook_registry(path)
    entries: list[HookEntry] = []
    for raw in data.get("hooks", []):
        entries.append(
            HookEntry(
                hook_id=raw["hook_id"],
                hook_name=raw["hook_name"],
                tier=raw["tier"],
                status=raw["status"],
                key_fields=tuple(raw.get("key_fields", [])),
                availability=raw["availability"],
            )
        )
    return entries
