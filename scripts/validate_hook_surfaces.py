#!/usr/bin/env python3
"""Validate required v2 hooks and planned stub hook payload surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ree_v2.hooks.emitter import emit_planned_stub_hooks, emit_v2_hooks
from ree_v2.latent_substrate.encoder import LatentEncoder
from ree_v2.latent_substrate.predictor import FastPredictor
from ree_v2.sensor_adapter.adapter import SensorAdapter

REQUIRED_IDS = [f"HK-00{i}" for i in range(1, 10)]
STUB_IDS = [f"HK-10{i}" for i in range(1, 5)]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_key_path(payload: dict[str, Any], key_path: str) -> bool:
    cursor: Any = payload
    for token in key_path.split("."):
        if not isinstance(cursor, dict) or token not in cursor:
            return False
        cursor = cursor[token]
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path, help="Path to contracts/hook_registry.v1.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    hooks = {item["hook_id"]: item for item in registry.get("hooks", [])}
    bridge_schema = load_registry(REPO_ROOT / "contracts" / "schemas" / "v1" / "bridge_hooks.v1.json")
    bridge_validator = Draft202012Validator(bridge_schema)

    issues: list[str] = []
    for hook_id in REQUIRED_IDS + STUB_IDS:
        if hook_id not in hooks:
            issues.append(f"missing hook_id in registry: {hook_id}")

    adapter = SensorAdapter(context_window=4)
    ingress = adapter.adapt(obs_t={"x": 1.0}, ctx_window=[1, 2, 3, 4], a_t={"move": "left"})

    encoder = LatentEncoder(latent_dim=8)
    z_t = encoder.encode(ingress["obs_t"], ingress["ctx_window"])

    predictor = FastPredictor(horizon=3)
    prediction = predictor.predict(
        z_t,
        ingress["trace"],
        include_uncertainty=True,
    )

    emitted_required = emit_v2_hooks(
        z_t=z_t,
        z_hat=prediction["z_hat"],
        pe_latent=prediction["pe_latent"],
        context_mask_ids=ingress["trace"]["context_mask_ids"],
        include_uncertainty=True,
        uncertainty_latent=prediction.get("uncertainty_latent"),
        include_action_token=True,
        action_token=ingress["trace"]["action_token"],
        commit_boundary={
            "commit_id": "cb-validate-surface",
            "trajectory_id": "trajectory-validate-surface",
            "issued_at_step": 3,
            "ttl_steps": 8,
            "mode_snapshot": "validation_mode",
        },
        tri_loop_trace={
            "gate_motor": "action_conditioned",
            "gate_cognitive_set": "validation_profile",
            "gate_motivational": "error_reduction",
            "gate_arbitration_policy": "dual_stream_pre_post",
            "gate_conflict_flag": False,
        },
        control_axes={
            "tonic": 0.1,
            "phasic": 0.05,
            "readout_weights": [0.5, 0.3, 0.2],
        },
    )
    emitted_stubs = emit_planned_stub_hooks()
    bridge_payloads = {
        hook_id: emitted_required.get(hook_id)
        for hook_id in ("HK-007", "HK-008", "HK-009")
    }
    for error in sorted(bridge_validator.iter_errors(bridge_payloads), key=lambda item: list(item.path)):
        path = ".".join(str(item) for item in error.path) or "$"
        issues.append(f"bridge schema validation failed at {path}: {error.message}")

    for hook_id in REQUIRED_IDS:
        hook = hooks.get(hook_id)
        payload = emitted_required.get(hook_id)
        if payload is None:
            issues.append(f"{hook_id}: required hook did not emit payload")
            continue

        for key_path in hook.get("key_fields", []):
            if not has_key_path(payload, key_path):
                issues.append(f"{hook_id}: payload missing required key field '{key_path}'")

    for hook_id in STUB_IDS:
        hook = hooks.get(hook_id)
        payload = emitted_stubs.get(hook_id)
        if payload is None:
            issues.append(f"{hook_id}: planned stub payload missing")
            continue
        if payload.get("planned_stub") is not True:
            issues.append(f"{hook_id}: planned stub payload missing planned_stub=true")
        for key_path in hook.get("key_fields", []):
            if not has_key_path(payload, key_path):
                issues.append(f"{hook_id}: planned stub missing key field '{key_path}'")

    if issues:
        print(f"FAIL: hook surface validation found {len(issues)} issue(s)")
        for issue in issues:
            print(f" - {issue}")
        return 1

    print("PASS: hook surface contract verified for HK-001..HK-009 and HK-101..HK-104")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
