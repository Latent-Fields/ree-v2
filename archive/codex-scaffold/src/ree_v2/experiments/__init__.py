"""Qualification profile catalog and deterministic simulation helpers."""

from .profiles import (
    ClaimProfile,
    ProfileCondition,
    ResourceEstimate,
    evaluate_failure_signatures,
    get_profile,
    get_profiles,
    simulate_metrics,
)
from .resource_policy import MACHINE_PROFILES, MachineProfile, decide_execution_mode
from .runner import RunExecutionResult, execute_profile_condition

__all__ = [
    "ClaimProfile",
    "MachineProfile",
    "MACHINE_PROFILES",
    "ProfileCondition",
    "ResourceEstimate",
    "decide_execution_mode",
    "evaluate_failure_signatures",
    "execute_profile_condition",
    "get_profile",
    "get_profiles",
    "RunExecutionResult",
    "simulate_metrics",
]
