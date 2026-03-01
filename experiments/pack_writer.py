"""
Experiment Pack v1 writer utilities.

This module emits run artifacts compatible with the REE Assembly
Experiment Pack interface contract:

<output_root>/<experiment_type>/runs/<run_id>/
  manifest.json
  metrics.json
  summary.md

V2 note: JEPA adapter signals removed. This module no longer writes
jepa_adapter_signals.v1.json. The EmittedPack and write_pack() signatures
are stripped of adapter_signals to reflect the native REE E1/E2 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Optional, Union


MANIFEST_SCHEMA_VERSION = "experiment_pack/v1"
METRICS_SCHEMA_VERSION = "experiment_pack_metrics/v1"
STOP_CRITERIA_VERSION = "stop_criteria/v1"
OUTPUT_ROOT_ENV = "REE_EXPERIMENT_OUTPUT_ROOT"
EVIDENCE_DIRECTIONS = {"supports", "weakens", "mixed", "unknown"}
REQUIRED_ENVIRONMENT_FIELDS = (
    "env_id",
    "env_version",
    "dynamics_hash",
    "reward_hash",
    "observation_hash",
    "config_hash",
    "tier",
)
DEFAULT_PRODUCER_CAPABILITIES = {
    "trajectory_integrity_channelized_bias": True,
    "mech056_dispatch_metric_set": True,
    "mech056_summary_escalation_trace": True,
}
DEFAULT_ENVIRONMENT = {
    "env_id": "ree.unknown",
    "env_version": "unknown",
    "dynamics_hash": "unknown",
    "reward_hash": "unknown",
    "observation_hash": "unknown",
    "config_hash": "unknown",
    "tier": "unknown",
}

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_EXPERIMENT_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class EmittedPack:
    """Paths for an emitted Experiment Pack."""

    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    summary_path: Path


def normalize_timestamp_utc(timestamp_utc: Optional[str] = None) -> str:
    """Return an RFC3339 UTC timestamp string."""
    if timestamp_utc is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    normalized = timestamp_utc
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def deterministic_run_id(experiment_type: str, seed: int, timestamp_utc: str) -> str:
    """Build deterministic run id from timestamp, experiment type, and seed."""
    normalized = normalize_timestamp_utc(timestamp_utc)
    dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    ts = dt.strftime("%Y-%m-%dT%H%M%SZ")
    exp_slug = _EXPERIMENT_SLUG_RE.sub("-", experiment_type.lower()).strip("-")
    return f"{ts}_{exp_slug}_seed{seed}"


def stable_config_hash(config_payload: Mapping[str, Any]) -> str:
    """Stable short hash for a scenario/config payload."""
    payload = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def resolve_output_root(cli_output_root: Optional[str]) -> Path:
    """Resolve output root from CLI arg, env var, or default."""
    if cli_output_root:
        return Path(cli_output_root)
    from_env = os.getenv(OUTPUT_ROOT_ENV)
    if from_env:
        return Path(from_env)
    return Path("runs")


def discover_source_repo(repo_root: Path) -> dict[str, str]:
    """Best-effort source repository metadata."""
    source_repo = {
        "name": repo_root.name,
        "commit": _git_value(["rev-parse", "HEAD"], repo_root) or "unknown",
    }
    branch = _git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if branch and branch != "HEAD":
        source_repo["branch"] = branch
    return source_repo


def _git_value(args: list[str], cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return value or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ExperimentPackWriter:
    """Reusable writer for Experiment Pack v1 artifacts."""

    def __init__(
        self,
        output_root: Path,
        repo_root: Path,
        runner_name: str,
        runner_version: str,
    ):
        self.output_root = output_root
        self.repo_root = repo_root
        self.runner_name = runner_name
        self.runner_version = runner_version

    def write_pack(
        self,
        experiment_type: str,
        run_id: str,
        timestamp_utc: str,
        status: str,
        metrics_values: Mapping[str, Any],
        summary_markdown: str,
        scenario: Optional[Mapping[str, Any]] = None,
        failure_signatures: Optional[list[str]] = None,
        claim_ids_tested: Optional[list[str]] = None,
        evidence_class: Optional[str] = None,
        evidence_direction: Optional[str] = None,
        producer_capabilities: Optional[Mapping[str, bool]] = None,
        environment: Optional[Mapping[str, Any]] = None,
        traces_dir: Optional[str] = None,
        media_dir: Optional[str] = None,
    ) -> EmittedPack:
        status = status.upper()
        if status not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid status '{status}' (expected PASS or FAIL)")

        normalized_ts = normalize_timestamp_utc(timestamp_utc)
        clean_metrics = _clean_numeric_metrics(metrics_values)
        clean_claim_ids = _clean_claim_ids(claim_ids_tested or [])
        clean_evidence_class = _clean_evidence_class(evidence_class)
        clean_evidence_direction = _clean_evidence_direction(evidence_direction)
        clean_producer_capabilities = _clean_producer_capabilities(producer_capabilities)
        clean_environment = _clean_environment(environment)

        run_dir = self.output_root / experiment_type / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        metrics_path = run_dir / "metrics.json"
        summary_path = run_dir / "summary.md"
        manifest_path = run_dir / "manifest.json"

        metrics_doc = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "values": clean_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8")
        summary_path.write_text(summary_markdown.rstrip() + "\n", encoding="utf-8")

        artifacts: dict[str, str] = {
            "metrics_path": "metrics.json",
            "summary_path": "summary.md",
        }
        if traces_dir:
            (run_dir / traces_dir).mkdir(parents=True, exist_ok=True)
            artifacts["traces_dir"] = traces_dir
        if media_dir:
            (run_dir / media_dir).mkdir(parents=True, exist_ok=True)
            artifacts["media_dir"] = media_dir

        manifest_doc: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_type": experiment_type,
            "run_id": run_id,
            "status": status,
            "timestamp_utc": normalized_ts,
            "source_repo": discover_source_repo(self.repo_root),
            "runner": {
                "name": self.runner_name,
                "version": self.runner_version,
            },
            "artifacts": artifacts,
            "stop_criteria_version": STOP_CRITERIA_VERSION,
            "claim_ids_tested": clean_claim_ids,
            "evidence_class": clean_evidence_class,
            "evidence_direction": clean_evidence_direction,
            "producer_capabilities": clean_producer_capabilities,
            "environment": clean_environment,
        }

        if scenario:
            manifest_doc["scenario"] = dict(scenario)

        signatures = _dedupe_strings(failure_signatures or [])
        if signatures:
            manifest_doc["failure_signatures"] = signatures
        else:
            manifest_doc["failure_signatures"] = []

        manifest_path.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")

        return EmittedPack(
            run_dir=run_dir,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
        )


def _clean_numeric_metrics(metrics_values: Mapping[str, Any]) -> dict[str, Union[float, int]]:
    clean: dict[str, Union[float, int]] = {}
    for key, value in metrics_values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metrics keys must be non-empty strings")
        if not _SNAKE_CASE_RE.match(key):
            raise ValueError(f"metric key must be snake_case: {key}")

        numeric_value = _coerce_numeric(value)
        clean[key] = numeric_value

    return clean


def _clean_claim_ids(claim_ids: list[str]) -> list[str]:
    cleaned: list[str] = []
    for claim_id in claim_ids:
        value = str(claim_id).strip()
        if value:
            cleaned.append(value)
    return _dedupe_strings(cleaned)


def _clean_evidence_class(evidence_class: Optional[str]) -> str:
    if evidence_class is None:
        return "simulation"
    cleaned = evidence_class.strip()
    if not cleaned:
        return "simulation"
    return cleaned


def _clean_evidence_direction(evidence_direction: Optional[str]) -> str:
    if evidence_direction is None:
        return "unknown"
    cleaned = evidence_direction.strip().lower()
    if cleaned not in EVIDENCE_DIRECTIONS:
        expected = ", ".join(sorted(EVIDENCE_DIRECTIONS))
        raise ValueError(
            f"invalid evidence_direction '{evidence_direction}' (expected one of: {expected})"
        )
    return cleaned


def _clean_producer_capabilities(
    producer_capabilities: Optional[Mapping[str, bool]],
) -> dict[str, bool]:
    clean = dict(DEFAULT_PRODUCER_CAPABILITIES)
    if producer_capabilities is None:
        return clean

    for key, value in producer_capabilities.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("producer capability keys must be non-empty strings")
        if not isinstance(value, bool):
            raise TypeError(f"producer capability '{key}' must be a boolean")
        clean[key] = value
    return clean


def _clean_environment(environment: Optional[Mapping[str, Any]]) -> dict[str, str]:
    clean = dict(DEFAULT_ENVIRONMENT)
    if environment is not None:
        for key, value in environment.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("environment keys must be non-empty strings")
            clean[key] = str(value).strip() if value is not None else "unknown"

    for key in REQUIRED_ENVIRONMENT_FIELDS:
        value = clean.get(key, "").strip()
        clean[key] = value if value else "unknown"
    return clean


def _coerce_numeric(value: Any) -> Union[float, int]:
    if isinstance(value, bool):
        raise TypeError("metrics values must be numeric; booleans are not allowed")
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item"):
        item_value = value.item()
        if isinstance(item_value, bool):
            raise TypeError("metrics values must be numeric; booleans are not allowed")
        if isinstance(item_value, (int, float)):
            return item_value
    raise TypeError(f"metrics values must be int/float, got {type(value)!r}")


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped
