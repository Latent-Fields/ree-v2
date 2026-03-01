"""Execution placement policy shared by estimators and remote-job tooling."""

from __future__ import annotations

from dataclasses import dataclass

from .profiles import ClaimProfile, ProfileCondition


@dataclass(frozen=True)
class MachineProfile:
    machine_id: str
    safe_memory_gb: float
    runtime_trigger_minutes: float
    batch_trigger_minutes: float


MACHINE_PROFILES: dict[str, MachineProfile] = {
    "macbook_air_m2_2022": MachineProfile(
        machine_id="macbook_air_m2_2022",
        safe_memory_gb=8.0,
        runtime_trigger_minutes=360.0,
        batch_trigger_minutes=360.0,
    )
}


def decide_execution_mode(
    profile: ClaimProfile,
    condition: ProfileCondition,
    machine_id: str,
    *,
    thermal_throttling_detected: bool = False,
    oom_detected: bool = False,
) -> tuple[str, list[str], float, int]:
    if machine_id not in MACHINE_PROFILES:
        known = ", ".join(sorted(MACHINE_PROFILES))
        raise KeyError(f"Unknown machine '{machine_id}'. Known machines: {known}")

    machine = MACHINE_PROFILES[machine_id]
    per_run = condition.resources.runtime_minutes
    seeds_per_condition = len(profile.default_seeds)
    batch_runtime = per_run * seeds_per_condition

    reasons: list[str] = []
    if per_run > machine.runtime_trigger_minutes:
        reasons.append(f"per_run>{machine.runtime_trigger_minutes:.0f}m")
    if batch_runtime > machine.batch_trigger_minutes:
        reasons.append(f"batch>{machine.batch_trigger_minutes:.0f}m")
    if seeds_per_condition > 2:
        reasons.append("seeds_per_condition>2")
    if condition.resources.memory_gb > machine.safe_memory_gb:
        reasons.append(f"memory>{machine.safe_memory_gb:.1f}GB")
    if thermal_throttling_detected:
        reasons.append("thermal_throttling_detected")
    if oom_detected:
        reasons.append("oom_detected")

    mode = "remote" if reasons else "local"
    return mode, reasons, batch_runtime, seeds_per_condition
