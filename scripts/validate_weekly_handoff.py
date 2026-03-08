#!/usr/bin/env python3
"""Validate weekly handoff markdown against required v1 template contract."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = [
    "Metadata",
    "Contract Sync",
    "CI Gates",
    "Run-Pack Inventory",
    "Claim Summary",
    "Open Blockers",
    "Local Compute Options Watch",
]

CI_COLUMNS = ["gate", "status", "evidence"]
CI_GATES = {
    "schema_validation": {"PASS", "FAIL"},
    "seed_determinism": {"PASS", "FAIL"},
    "hook_surface_coverage": {"PASS", "FAIL", "N/A"},
    "remote_export_import": {"PASS", "FAIL", "N/A"},
}

RUN_COLUMNS = [
    "experiment_type",
    "run_id",
    "seed",
    "condition_or_scenario",
    "status",
    "evidence_direction",
    "claim_ids_tested",
    "failure_signatures",
    "execution_mode",
    "compute_backend",
    "runtime_minutes",
    "pack_path",
]

CLAIM_COLUMNS = [
    "claim_id",
    "runs_added",
    "supports",
    "weakens",
    "mixed",
    "unknown",
    "recurring_failure_signatures",
]

METADATA_KEYS = ["week_of_utc", "producer_repo", "producer_commit", "generated_utc"]
CONTRACT_KEYS = [
    "ree_assembly_repo",
    "ree_assembly_commit",
    "contract_lock_path",
    "contract_lock_hash",
    "schema_version_set",
]
LOCAL_WATCH_KEYS = [
    "local_options_last_updated_utc",
    "rolling_3mo_cloud_spend_eur",
    "local_blocked_sessions_this_week",
    "recommended_local_action",
    "rationale",
]

ALLOWED_DIRECTIONS = {"supports", "weakens", "mixed", "unknown"}
ALLOWED_EXECUTION = {"local", "remote"}
ALLOWED_LOCAL_ACTIONS = {"hold_cloud_only", "upgrade_low", "upgrade_mid", "upgrade_high", "N/A"}


class ValidationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to weekly handoff markdown")
    return parser.parse_args()


def extract_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip("\n")
    return sections


def parse_bullets(section_body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in section_body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        payload = stripped[2:]
        if ":" not in payload:
            continue
        key, value = payload.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_first_table(section_body: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line.rstrip() for line in section_body.splitlines()]
    table_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_start = idx
            break

    if table_start is None:
        raise ValidationError("No markdown table found in section")

    if table_start + 1 >= len(lines):
        raise ValidationError("Table missing separator row")

    header_line = lines[table_start].strip()
    separator_line = lines[table_start + 1].strip()

    if not (separator_line.startswith("|") and "---" in separator_line):
        raise ValidationError("Table separator row malformed")

    headers = [cell.strip() for cell in header_line.strip("|").split("|")]

    rows: list[dict[str, str]] = []
    idx = table_start + 2
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            raise ValidationError("Table row column count mismatch")
        rows.append(dict(zip(headers, cells)))
        idx += 1

    return headers, rows


def require_section(sections: dict[str, str], section_name: str) -> str:
    if section_name not in sections:
        raise ValidationError(f"Missing required section: {section_name}")
    return sections[section_name]


def validate_required_bullets(section_name: str, section_body: str, keys: list[str]) -> None:
    fields = parse_bullets(section_body)
    missing = [key for key in keys if key not in fields]
    if missing:
        raise ValidationError(f"Section '{section_name}' missing required field(s): {missing}")


def validate_ci_gates(section_body: str) -> None:
    headers, rows = parse_first_table(section_body)
    if headers != CI_COLUMNS:
        raise ValidationError(f"CI Gates columns mismatch. Expected {CI_COLUMNS}, found {headers}")

    row_map = {row.get("gate", ""): row for row in rows}
    for gate, allowed_statuses in CI_GATES.items():
        if gate not in row_map:
            raise ValidationError(f"CI Gates missing row for '{gate}'")
        status = row_map[gate].get("status", "")
        if status not in allowed_statuses:
            raise ValidationError(f"CI gate '{gate}' has invalid status '{status}'")


def validate_run_inventory(section_body: str) -> None:
    headers, rows = parse_first_table(section_body)
    if headers != RUN_COLUMNS:
        raise ValidationError(f"Run-Pack Inventory columns mismatch. Expected {RUN_COLUMNS}, found {headers}")

    if not rows:
        raise ValidationError("Run-Pack Inventory must contain at least one run row")

    for row in rows:
        direction = row.get("evidence_direction", "")
        if direction not in ALLOWED_DIRECTIONS:
            raise ValidationError(f"Invalid evidence_direction '{direction}' in Run-Pack Inventory")
        mode = row.get("execution_mode", "")
        if mode not in ALLOWED_EXECUTION:
            raise ValidationError(f"Invalid execution_mode '{mode}' in Run-Pack Inventory")


def validate_claim_summary(section_body: str) -> None:
    headers, rows = parse_first_table(section_body)
    if headers != CLAIM_COLUMNS:
        raise ValidationError(f"Claim Summary columns mismatch. Expected {CLAIM_COLUMNS}, found {headers}")

    if not rows:
        raise ValidationError("Claim Summary must contain at least one claim row")


def validate_open_blockers(section_body: str) -> None:
    bullets = [line.strip()[2:] for line in section_body.splitlines() if line.strip().startswith("- ")]
    if not bullets:
        raise ValidationError("Open Blockers must include at least one bullet")


def validate_local_watch(section_body: str) -> None:
    fields = parse_bullets(section_body)
    missing = [key for key in LOCAL_WATCH_KEYS if key not in fields]
    if missing:
        raise ValidationError(f"Local Compute Options Watch missing field(s): {missing}")

    action_raw = fields["recommended_local_action"].strip().strip("`")
    if action_raw not in ALLOWED_LOCAL_ACTIONS:
        raise ValidationError(
            "Local Compute Options Watch recommended_local_action must be one of "
            f"{sorted(ALLOWED_LOCAL_ACTIONS)}; got '{action_raw}'"
        )


def validate(text: str) -> None:
    sections = extract_sections(text)

    for section_name in REQUIRED_SECTIONS:
        require_section(sections, section_name)

    validate_required_bullets("Metadata", sections["Metadata"], METADATA_KEYS)
    validate_required_bullets("Contract Sync", sections["Contract Sync"], CONTRACT_KEYS)
    local_watch_fields = parse_bullets(sections["Local Compute Options Watch"])
    validate_ci_gates(sections["CI Gates"])
    validate_run_inventory(sections["Run-Pack Inventory"])
    validate_claim_summary(sections["Claim Summary"])
    validate_open_blockers(sections["Open Blockers"])
    validate_local_watch(sections["Local Compute Options Watch"])


def main() -> int:
    args = parse_args()
    try:
        text = args.input.read_text(encoding="utf-8")
        validate(text)
    except FileNotFoundError:
        print(f"FAIL: input not found: {args.input}")
        return 1
    except ValidationError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: weekly handoff is valid: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
