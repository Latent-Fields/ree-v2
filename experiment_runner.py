#!/usr/bin/env python3
"""
REE Experiment Runner

Runs pending experiments from experiment_queue.json sequentially,
writing live progress to runner_status.json for the claims explorer
dashboard.

Usage:
    # Run in foreground (Ctrl+C to stop):
    python experiment_runner.py

    # Run detached (survives closing this terminal):
    nohup python experiment_runner.py > runner.log 2>&1 &
    echo $! > runner.pid      # save PID so you can kill it later

    # Stop a detached run:
    kill $(cat runner.pid)

    # Specify where to write the status file:
    python experiment_runner.py --status-file /path/to/runner_status.json

The runner writes status to runner_status.json (default: auto-detected
REE_assembly path, or ./runner_status.json as fallback).

It skips items whose script doesn't exist yet (status: needs_script) and
moves on to the next runnable item.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
QUEUE_FILE = REPO_ROOT / "experiment_queue.json"
PID_FILE = REPO_ROOT / "runner.pid"

# Experiment outputs go here (each experiment writes its own JSON):
EVIDENCE_DIR = REPO_ROOT / "evidence" / "experiments"

# Auto-detect REE_assembly: look one level up from ree-v1-minimal
_REE_ASSEMBLY_CANDIDATES = [
    REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status.json",
    Path.home() / "Documents" / "GitHub" / "REE_assembly" / "evidence" / "experiments" / "runner_status.json",
]

# Refresh interval: how often (seconds) to re-write status during a run
STATUS_WRITE_INTERVAL = 5

# Regex patterns for parsing experiment stdout
RE_SEED_CONDITION = re.compile(r'Seed\s+(\d+)\s+Condition\s+(\w+)')
RE_EP_PROGRESS = re.compile(r'ep\s+(\d+)/(\d+)')
RE_RUN_DONE = re.compile(r'harm\s+[\d.]+\s*→\s*[\d.]+')
RE_SAVED_TO = re.compile(r'Results saved to:\s+(.+)')


def find_default_status_path() -> Path:
    for candidate in _REE_ASSEMBLY_CANDIDATES:
        if candidate.parent.exists():
            return candidate
    return REPO_ROOT / "runner_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


_write_status_lock = threading.Lock()


def write_status(status: dict, path: Path) -> None:
    """Atomic write: write to .tmp then rename to avoid partial reads.

    Lock ensures the heartbeat thread and main thread don't race on the
    same .tmp file (which would cause ENOENT on rename).
    """
    with _write_status_lock:
        tmp = path.with_suffix(".tmp")
        status["last_updated"] = now_utc()
        tmp.write_text(json.dumps(status, indent=2))
        tmp.rename(path)


def load_queue() -> dict:
    with open(QUEUE_FILE) as f:
        return json.load(f)


def estimate_minutes(item: dict, calibration: dict) -> float:
    seeds = item.get("seeds", 3)
    conditions = item.get("conditions", 2)
    episodes = item.get("episodes_per_run", 200)
    ms_per = calibration.get("ms_per_episode_condition", 750)
    return (seeds * conditions * episodes * ms_per) / 60_000


def build_initial_status(queue_data: dict) -> dict:
    """Build the status document skeleton from the queue."""
    calibration = queue_data.get("calibration", {})
    queue_items = []
    for item in queue_data.get("items", []):
        queue_items.append({
            "queue_id": item["queue_id"],
            "backlog_id": item.get("backlog_id", ""),
            "claim_id": item.get("claim_id", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "estimated_minutes": round(estimate_minutes(item, calibration), 1),
            "status": item.get("status", "pending"),
            "status_reason": item.get("status_reason", ""),
            "ree_version": "v2",
        })
    return {
        "schema_version": "v1",
        "runner_pid": os.getpid(),
        "runner_started_at": now_utc(),
        "last_updated": now_utc(),
        "idle": False,
        "current": None,
        "queue": queue_items,
        "completed": [],
    }


def run_experiment(item: dict, status: dict, status_path: Path, calibration: dict) -> dict:
    """
    Run one experiment as a subprocess, tracking progress line by line.

    Returns a result dict with keys: result, result_summary, completed_at,
    output_file (if found in stdout), error (if failed).
    """
    script = REPO_ROOT / item["script"]
    # -u: force unbuffered stdout so progress lines reach the runner in real time
    # (without -u Python block-buffers when writing to a pipe, so no lines arrive
    # until the experiment finishes and the buffer flushes on process exit)
    args = [sys.executable, "-u", str(script)] + item.get("args", [])

    seeds = item.get("seeds", 3)
    conditions = item.get("conditions", 2)
    total_runs = seeds * conditions
    episodes_per_run = item.get("episodes_per_run", 200)

    # Track progress
    runs_done = 0
    current_run_label = "starting..."
    episodes_in_run = 0
    recent_lines: list[str] = []

    started_at = time.monotonic()
    started_at_utc = now_utc()

    def overall_pct() -> float:
        run_frac = (runs_done + episodes_in_run / max(episodes_per_run, 1)) / max(total_runs, 1)
        return round(run_frac * 100, 1)

    def seconds_remaining() -> float:
        elapsed = time.monotonic() - started_at
        pct = overall_pct()
        if pct <= 0:
            total_estimated = estimate_minutes(item, calibration) * 60
            return total_estimated
        total_estimated = elapsed / (pct / 100)
        return max(0, total_estimated - elapsed)

    def update_status_current():
        status["current"] = {
            "queue_id": item["queue_id"],
            "backlog_id": item.get("backlog_id", ""),
            "claim_id": item.get("claim_id", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "script": item["script"],
            "started_at": started_at_utc,
            "progress": {
                "run_label": current_run_label,
                "runs_done": runs_done,
                "runs_total": total_runs,
                "episodes_done": episodes_in_run,
                "episodes_total": episodes_per_run,
                "overall_pct": overall_pct(),
            },
            "seconds_elapsed": round(time.monotonic() - started_at),
            "seconds_remaining": round(seconds_remaining()),
            "recent_lines": recent_lines[-5:],
            "ree_version": "v2",
        }
        # Update queue item status to running
        for qi in status["queue"]:
            if qi["queue_id"] == item["queue_id"]:
                qi["status"] = "running"
        write_status(status, status_path)

    print(f"[runner] Starting: {item['title']} ({item['queue_id']})", flush=True)
    print(f"[runner] Command: {' '.join(str(a) for a in args)}", flush=True)

    last_write = time.monotonic()
    update_status_current()

    result_info = {"result": "UNKNOWN", "result_summary": "", "started_at": started_at_utc, "completed_at": "", "output_file": ""}

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Heartbeat thread: write elapsed time every 5 s even when the
        # experiment script has no new output (e.g. between ep-50 prints).
        _hb_stop = threading.Event()
        def _heartbeat():
            while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
                update_status_current()
        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)

            # Parse progress signals
            m = RE_SEED_CONDITION.search(line)
            if m:
                current_run_label = f"Seed {m.group(1)} / {m.group(2)}"
                episodes_in_run = 0

            m = RE_EP_PROGRESS.search(line)
            if m:
                episodes_in_run = int(m.group(1))

            if RE_RUN_DONE.search(line):
                runs_done += 1
                episodes_in_run = episodes_per_run

            m = RE_SAVED_TO.search(line)
            if m:
                result_info["output_file"] = m.group(1).strip()

            # Capture verdict
            if "verdict: PASS" in line:
                result_info["result"] = "PASS"
            elif "verdict: FAIL" in line:
                result_info["result"] = "FAIL"
            elif "verdict: IMPROVED" in line:
                result_info["result"] = "IMPROVED"

            # Keep recent non-blank lines
            stripped = line.strip()
            if stripped:
                recent_lines.append(stripped)
                if len(recent_lines) > 20:
                    recent_lines.pop(0)

            # Throttle status writes
            if time.monotonic() - last_write >= STATUS_WRITE_INTERVAL:
                update_status_current()
                last_write = time.monotonic()

        _hb_stop.set()
        proc.wait()
        exit_code = proc.returncode

        if exit_code != 0 and result_info["result"] == "UNKNOWN":
            result_info["result"] = "ERROR"
            result_info["result_summary"] = f"Non-zero exit code {exit_code}"

    except Exception as exc:
        result_info["result"] = "ERROR"
        result_info["result_summary"] = str(exc)
        print(f"[runner] ERROR running {item['queue_id']}: {exc}", flush=True)

    result_info["completed_at"] = now_utc()

    # Build result summary from recent output
    if not result_info["result_summary"]:
        # Grab summary lines (the final verdict block)
        summary_lines = [l for l in recent_lines if any(
            kw in l for kw in ["harm", "corr", "verdict", "Harm reduction", "Signals", "Separation"]
        )]
        result_info["result_summary"] = " | ".join(summary_lines[-3:])

    return result_info


def main():
    parser = argparse.ArgumentParser(description="REE Experiment Runner")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Path to write runner_status.json (default: auto-detect REE_assembly)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "After exhausting the queue, poll experiment_queue.json every "
            "--loop-interval seconds for new items instead of exiting. "
            "Lets the runner stay alive so clicking Start in the explorer "
            "keeps it running without a manual restart."
        ),
    )
    parser.add_argument(
        "--loop-interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Seconds between queue re-checks in --loop mode (default: 60)",
    )
    args = parser.parse_args()

    status_path = args.status_file or find_default_status_path()
    print(f"[runner] Status file: {status_path}", flush=True)
    print(f"[runner] Queue file: {QUEUE_FILE}", flush=True)

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    # Handle signals gracefully
    def handle_signal(sig, frame):
        print(f"\n[runner] Caught signal {sig}, shutting down.", flush=True)
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                status["idle"] = True
                status["current"] = None
                status["runner_pid"] = None
                # Reset any queue item stuck in "running" back to "pending"
                for qi in status.get("queue", []):
                    if qi.get("status") == "running":
                        qi["status"] = "pending"
                write_status(status, status_path)
            except Exception:
                pass
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    queue_data = load_queue()
    calibration = queue_data.get("calibration", {})
    items = queue_data.get("items", [])

    # Load existing status if present (to preserve completed list)
    existing_completed = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
            existing_completed = existing.get("completed", [])
        except Exception:
            pass

    status = build_initial_status(queue_data)
    status["completed"] = existing_completed
    write_status(status, status_path)

    if args.dry_run:
        print("[runner] Dry run — queue:")
        for item in items:
            script = REPO_ROOT / item["script"]
            runnable = script.exists()
            mins = estimate_minutes(item, calibration)
            print(f"  {item['queue_id']} {item['claim_id']:12s} ~{mins:.0f}min  "
                  f"{'READY' if runnable else 'NEEDS_SCRIPT'}: {item['title']}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return

    print(f"[runner] PID {os.getpid()} — ready to run {len(items)} queued experiments", flush=True)
    if args.loop:
        print(f"[runner] Loop mode: will poll queue every {args.loop_interval}s after exhaustion", flush=True)

    completed_ids = {c["queue_id"] for c in existing_completed}

    # Prune any already-completed items from the initial queue display
    status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] not in completed_ids]
    write_status(status, status_path)

    while True:
        ran_any = False

        for item in items:
            queue_id = item["queue_id"]

            # Skip already completed
            if queue_id in completed_ids:
                continue

            # Skip items with no script
            script = REPO_ROOT / item["script"]
            if not script.exists():
                print(f"[runner] Skipping {queue_id} ({item['claim_id']}) — script not found: {item['script']}", flush=True)
                # Update queue item status in live status file
                for qi in status["queue"]:
                    if qi["queue_id"] == queue_id:
                        qi["status"] = "needs_script"
                write_status(status, status_path)
                continue

            # Run it
            result = run_experiment(item, status, status_path, calibration)
            ran_any = True

            # Move to completed
            completed_entry = {
                "queue_id": queue_id,
                "backlog_id": item.get("backlog_id", ""),
                "claim_id": item.get("claim_id", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "result": result["result"],
                "result_summary": result["result_summary"],
                "started_at": result.get("started_at", ""),
                "completed_at": result["completed_at"],
                "output_file": result.get("output_file", ""),
            }
            status["completed"].append(completed_entry)
            completed_ids.add(queue_id)

            # Remove from queue display
            status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
            status["current"] = None

            write_status(status, status_path)
            print(f"[runner] Done: {queue_id} — {result['result']}", flush=True)

        # Queue pass complete
        if not args.loop:
            break

        # Loop mode: wait, then reload queue for any newly-added items
        status["idle"] = True
        status["current"] = None
        write_status(status, status_path)
        if ran_any:
            print(f"[runner] Pass complete. Waiting {args.loop_interval}s before re-checking queue…", flush=True)
        else:
            print(f"[runner] No new items. Waiting {args.loop_interval}s…", flush=True)

        time.sleep(args.loop_interval)

        # Reload queue — picks up any new items added while we slept
        queue_data = load_queue()
        calibration = queue_data.get("calibration", {})
        items = queue_data.get("items", [])

        new_pending = [i for i in items if i["queue_id"] not in completed_ids]
        if new_pending:
            print(f"[runner] Found {len(new_pending)} new item(s): "
                  f"{[i['queue_id'] for i in new_pending]}", flush=True)
            # Rebuild queue display in status
            new_queue_display = []
            for i in new_pending:
                new_queue_display.append({
                    "queue_id": i["queue_id"],
                    "backlog_id": i.get("backlog_id", ""),
                    "claim_id": i.get("claim_id", ""),
                    "title": i.get("title", ""),
                    "description": i.get("description", ""),
                    "estimated_minutes": round(estimate_minutes(i, calibration), 1),
                    "status": "pending",
                    "status_reason": i.get("status_reason", ""),
                    "ree_version": "v2",
                })
            status["queue"] = new_queue_display
            status["idle"] = False
            write_status(status, status_path)

    # All done (non-loop path)
    status["idle"] = True
    status["current"] = None
    status["runner_pid"] = None
    write_status(status, status_path)
    print("[runner] Queue exhausted. Runner idle.", flush=True)

    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
