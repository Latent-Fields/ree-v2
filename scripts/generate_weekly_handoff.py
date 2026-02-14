#!/usr/bin/env python3
"""Generate weekly handoff markdown from repo state and contract template."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ree_v2.experiments.profiles import get_profile
from ree_v2.experiments.resource_policy import decide_execution_mode

TEMPLATE_PATH_DEFAULT = Path(
    "/Users/dgolden/Documents/GitHub/REE_assembly/evidence/planning/WEEKLY_HANDOFF_TEMPLATE.md"
)

REQUIRED_TEMPLATE_MARKERS = [
    "## Required Fields and Sections",
    "## CI Gates",
    "## Run-Pack Inventory",
    "## Claim Summary",
    "## Open Blockers",
    "## Local Compute Options Watch",
]

GATE_DEFS = [
    (
        "schema_validation",
        ["python3", "scripts/validate_experiment_pack.py", "--runs-root", "evidence/experiments"],
    ),
    (
        "seed_determinism",
        [
            "python3",
            "scripts/check_seed_determinism.py",
            "--profile",
            "all",
            "--max-abs-delta",
            "1e-6",
        ],
    ),
    (
        "hook_surface_coverage",
        [
            "python3",
            "scripts/validate_hook_surfaces.py",
            "--registry",
            "contracts/hook_registry.v1.json",
        ],
    ),
]

REMOTE_ESTIMATE_COMMAND = [
    "python3",
    "scripts/estimate_run_resources.py",
    "--profile",
    "all",
    "--machine",
    "macbook_air_m2_2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output markdown path, e.g. evidence/planning/weekly_handoff/latest.md",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE_PATH_DEFAULT,
        help="Path to weekly handoff template contract",
    )
    parser.add_argument(
        "--week-of-utc",
        type=str,
        default=None,
        help="Week start date (YYYY-MM-DD). Defaults to Monday of current UTC week.",
    )
    parser.add_argument(
        "--generated-utc",
        type=str,
        default=None,
        help="Override generated timestamp (RFC3339 UTC).",
    )
    return parser.parse_args()


def rfc3339_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def monday_of_current_utc_week(today: dt.date) -> dt.date:
    return today - dt.timedelta(days=today.weekday())


def load_template_contract(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in REQUIRED_TEMPLATE_MARKERS if marker not in text]
    if missing:
        raise ValueError(f"Template missing required marker(s): {missing}")
    return text


def git_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_gate(command: list[str]) -> tuple[str, str]:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    status = "PASS" if result.returncode == 0 else "FAIL"
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    first_line = output.splitlines()[0] if output else "no output"
    return status, f"`{' '.join(command)}` :: {first_line}"


def collect_ci_gates() -> list[tuple[str, str, str]]:
    gates: list[tuple[str, str, str]] = []

    for gate_name, command in GATE_DEFS:
        status, evidence = run_gate(command)
        gates.append((gate_name, status, evidence))

    remote_status = "PASS"
    remote_evidence_parts: list[str] = []

    status, evidence = run_gate(REMOTE_ESTIMATE_COMMAND)
    remote_evidence_parts.append(evidence)
    if status != "PASS":
        remote_status = "FAIL"

    with tempfile.TemporaryDirectory(prefix="ree_v2_handoff_jobs_") as temp_root:
        temp_root_path = Path(temp_root)
        temp_outgoing = temp_root_path / "outgoing"
        temp_completed = temp_root_path / "completed"
        temp_outgoing.mkdir(parents=True, exist_ok=True)
        temp_completed.mkdir(parents=True, exist_ok=True)

        remote_commands = [
            [
                "python3",
                "scripts/build_remote_job_spec.py",
                "--profile",
                "all",
                "--out-dir",
                str(temp_outgoing),
            ],
            [
                "python3",
                "scripts/submit_remote_job.py",
                "--job-spec-dir",
                str(temp_outgoing),
                "--dry-run",
            ],
            [
                "python3",
                "scripts/pull_remote_results.py",
                "--job-run-dir",
                str(temp_completed),
                "--runs-root",
                "evidence/experiments",
                "--dry-run",
            ],
        ]

        for command in remote_commands:
            status, evidence = run_gate(command)
            remote_evidence_parts.append(evidence)
            if status != "PASS":
                remote_status = "FAIL"

    gates.append(("remote_export_import", remote_status, " ; ".join(remote_evidence_parts)))
    return gates


def condition_lookup(experiment_type: str, condition_name: str) -> tuple[str, str, str]:
    """Return execution_mode, compute_backend, runtime_minutes."""
    try:
        profile = get_profile(experiment_type)
        condition = next(item for item in profile.conditions if item.name == condition_name)
        mode, _reasons, _batch_runtime, _seed_count = decide_execution_mode(
            profile,
            condition,
            "macbook_air_m2_2022",
        )
        backend = "local_cpu" if mode == "local" else "cloud_gpu_a10g"
        return mode, backend, f"{condition.resources.runtime_minutes:.1f}"
    except Exception:
        return "remote", "cloud_gpu_a10g", "N/A"


def collect_run_inventory() -> list[dict[str, str]]:
    runs_root = REPO_ROOT / "evidence" / "experiments"
    rows: list[dict[str, str]] = []

    for experiment_dir in sorted(runs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        runs_dir = experiment_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = load_json(manifest_path)

            scenario = manifest.get("scenario", {})
            condition = str(scenario.get("condition") or scenario.get("name") or "N/A")
            mode, backend, runtime = condition_lookup(manifest.get("experiment_type", ""), condition)

            claim_ids = manifest.get("claim_ids_tested", [])
            failures = manifest.get("failure_signatures", [])
            row = {
                "experiment_type": str(manifest.get("experiment_type", "")),
                "run_id": str(manifest.get("run_id", run_dir.name)),
                "seed": str(scenario.get("seed", "N/A")),
                "condition_or_scenario": condition,
                "status": str(manifest.get("status", "")),
                "evidence_direction": str(manifest.get("evidence_direction", "unknown")),
                "claim_ids_tested": ",".join(str(item) for item in claim_ids) if claim_ids else "N/A",
                "failure_signatures": ",".join(str(item) for item in failures) if failures else "none",
                "execution_mode": mode,
                "compute_backend": backend,
                "runtime_minutes": runtime,
                "pack_path": str(run_dir.relative_to(REPO_ROOT)),
            }
            rows.append(row)

    return rows


def build_claim_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    per_claim: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs_added": 0,
            "supports": 0,
            "weakens": 0,
            "mixed": 0,
            "unknown": 0,
            "signatures": Counter(),
        }
    )

    for row in rows:
        claim_ids = [item.strip() for item in row["claim_ids_tested"].split(",") if item.strip() and item.strip() != "N/A"]
        signatures = [item.strip() for item in row["failure_signatures"].split(",") if item.strip() and item.strip() != "none"]
        direction = row["evidence_direction"] if row["evidence_direction"] in {"supports", "weakens", "mixed", "unknown"} else "unknown"

        for claim_id in claim_ids:
            bucket = per_claim[claim_id]
            bucket["runs_added"] += 1
            bucket[direction] += 1
            bucket["signatures"].update(signatures)

    summary_rows: list[dict[str, str]] = []
    for claim_id in sorted(per_claim):
        bucket = per_claim[claim_id]
        recurring = sorted(sig for sig, count in bucket["signatures"].items() if count > 1)
        summary_rows.append(
            {
                "claim_id": claim_id,
                "runs_added": str(bucket["runs_added"]),
                "supports": str(bucket["supports"]),
                "weakens": str(bucket["weakens"]),
                "mixed": str(bucket["mixed"]),
                "unknown": str(bucket["unknown"]),
                "recurring_failure_signatures": ",".join(recurring) if recurring else "none",
            }
        )

    return summary_rows


def local_options_watch() -> dict[str, str]:
    path = REPO_ROOT / "docs" / "ops" / "local_compute_options.md"
    if path.exists():
        updated = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        last_updated = updated.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    else:
        last_updated = "N/A"

    rolling_spend = 0.0
    blocked_sessions = 0

    if rolling_spend >= 250.0:
        recommendation = "upgrade_high"
        rationale = "Spend threshold suggests high tier only if workload is sustained >10h/week; keep cloud-first otherwise."
    elif rolling_spend >= 100.0 or blocked_sessions > 2:
        recommendation = "upgrade_mid"
        rationale = "Mid-tier trigger crossed by spend or repeated blocked sessions."
    elif rolling_spend < 80.0 and blocked_sessions > 0:
        recommendation = "upgrade_low"
        rationale = "Low-tier trigger met: rising local friction with low cloud spend."
    else:
        recommendation = "hold_cloud_only"
        rationale = "No spend/blocking pressure above hobby thresholds; keep cloud-first policy."

    return {
        "local_options_last_updated_utc": last_updated,
        "rolling_3mo_cloud_spend_eur": f"{rolling_spend:.0f}",
        "local_blocked_sessions_this_week": str(blocked_sessions),
        "recommended_local_action": recommendation,
        "rationale": rationale,
    }


def markdown_table(columns: list[str], rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    if not body_lines:
        body_lines.append("| " + " | ".join("" for _ in columns) + " |")
    return "\n".join([header, sep] + body_lines)


def render_handoff(
    *,
    week_of_utc: str,
    generated_utc: str,
    template_path: Path,
    template_hash: str,
    producer_commit: str,
    ree_assembly_commit: str,
    contract_lock_hash: str,
    schema_version_set: str,
    ci_gates: list[tuple[str, str, str]],
    run_rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    blockers: list[str],
    local_watch: dict[str, str],
) -> str:
    ci_columns = ["gate", "status", "evidence"]
    ci_rows = [{"gate": g, "status": s, "evidence": e} for g, s, e in ci_gates]

    run_columns = [
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

    claim_columns = [
        "claim_id",
        "runs_added",
        "supports",
        "weakens",
        "mixed",
        "unknown",
        "recurring_failure_signatures",
    ]

    lines = [
        f"# Weekly Handoff - ree-v2 - {week_of_utc}",
        "",
        "## Metadata",
        f"- week_of_utc: `{week_of_utc}`",
        "- producer_repo: `ree-v2`",
        f"- producer_commit: `{producer_commit}`",
        f"- generated_utc: `{generated_utc}`",
        "",
        "## Contract Sync",
        "- ree_assembly_repo: `REE_assembly`",
        f"- ree_assembly_commit: `{ree_assembly_commit}`",
        "- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`",
        f"- contract_lock_hash: `{contract_lock_hash}`",
        f"- schema_version_set: `{schema_version_set}`",
        f"- template_path: `{template_path}`",
        f"- template_sha256: `{template_hash}`",
        "",
        "## CI Gates",
        markdown_table(ci_columns, ci_rows),
        "",
        "## Run-Pack Inventory",
        markdown_table(run_columns, run_rows),
        "",
        "## Claim Summary",
        markdown_table(claim_columns, claim_rows),
        "",
        "## Open Blockers",
    ]

    for blocker in blockers:
        lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Local Compute Options Watch",
            f"- local_options_last_updated_utc: `{local_watch['local_options_last_updated_utc']}`",
            f"- rolling_3mo_cloud_spend_eur: `{local_watch['rolling_3mo_cloud_spend_eur']}`",
            f"- local_blocked_sessions_this_week: `{local_watch['local_blocked_sessions_this_week']}`",
            f"- recommended_local_action: `{local_watch['recommended_local_action']}`",
            f"- rationale: {local_watch['rationale']}",
            "",
        ]
    )

    return "\n".join(lines)


def schema_versions() -> str:
    manifest_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "manifest.schema.json")
    metrics_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "metrics.schema.json")
    adapter_schema = load_json(REPO_ROOT / "contracts" / "schemas" / "v1" / "jepa_adapter_signals.v1.json")
    hook_registry = load_json(REPO_ROOT / "contracts" / "hook_registry.v1.json")

    values = {
        str(manifest_schema["properties"]["schema_version"]["const"]),
        str(metrics_schema["properties"]["schema_version"]["const"]),
        str(adapter_schema["properties"]["schema_version"]["const"]),
        str(hook_registry.get("schema_version", "hook_registry/v1")),
    }
    return ", ".join(sorted(values))


def main() -> int:
    args = parse_args()

    template_text = load_template_contract(args.template)
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()

    today = dt.datetime.now(dt.timezone.utc).date()
    if args.week_of_utc:
        week_of = dt.date.fromisoformat(args.week_of_utc)
    else:
        week_of = monday_of_current_utc_week(today)

    generated_utc = args.generated_utc or rfc3339_utc_now()

    producer_commit = git_commit(REPO_ROOT)
    ree_assembly_commit = git_commit(Path("/Users/dgolden/Documents/GitHub/REE_assembly"))

    contract_lock_path = REPO_ROOT / "contracts" / "ree_assembly_contract_lock.v1.json"
    contract_lock_hash = sha256_file(contract_lock_path)

    ci_gates = collect_ci_gates()
    run_rows = collect_run_inventory()
    claim_rows = build_claim_summary(run_rows)

    failed = [gate for gate, status, _evidence in ci_gates if status == "FAIL"]
    if failed:
        blockers = [
            "CI gate failures require remediation before governance handoff: " + ", ".join(failed),
            "Re-run weekly handoff generation after fixes to update gate evidence.",
        ]
    else:
        blockers = ["none"]

    local_watch = local_options_watch()

    markdown = render_handoff(
        week_of_utc=week_of.isoformat(),
        generated_utc=generated_utc,
        template_path=args.template,
        template_hash=template_hash,
        producer_commit=producer_commit,
        ree_assembly_commit=ree_assembly_commit,
        contract_lock_hash=contract_lock_hash,
        schema_version_set=schema_versions(),
        ci_gates=ci_gates,
        run_rows=run_rows,
        claim_rows=claim_rows,
        blockers=blockers,
        local_watch=local_watch,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(f"Wrote weekly handoff: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
