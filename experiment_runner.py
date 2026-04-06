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
import socket
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

# Per-script timing learned from completed runs (fix 2)
SCRIPT_TIMING_FILE = REPO_ROOT / "script_timing.json"

# Regex patterns for parsing experiment stdout
RE_SEED_CONDITION = re.compile(r'Seed\s+(\d+)\s+Condition\s+(\w+)')
RE_EP_PROGRESS = re.compile(r'ep\s+(\d+)/(\d+)')
# Multiple run-completion signals — different experiment scripts emit different formats
RE_RUN_DONE_PATTERNS = [
    re.compile(r'harm\s+[\d.]+\s*→\s*[\d.]+'),          # V1/V2 parity scripts (harm X.XXX → Y.YYY)
    re.compile(r'calibration_gap:\s*[+-]?[\d.]+'),       # causal_attribution_calibration (EXQ-027)
    re.compile(r'final_harm:\s*[\d.]+'),                 # selective_residue_attribution (EXQ-028)
    re.compile(r'PATH_(?:MEMORY|ABLATED)\s+agent_harm'), # path_memory_ablation (EXQ-024)
    re.compile(r'(?:HIGH|LOW)_REGIME\s+precision='),     # precision_regime_probe (EXQ-025)
    re.compile(r'(?:HIGH|LOW)_CAUSAL\s+(?:harm|lift)='),  # action_doing_mode_probe (EXQ-026)
]
RE_SAVED_TO = re.compile(r'Results saved to:\s+(.+)')


def find_ree_assembly_path() -> Path | None:
    """Locate the REE_assembly repo (for git auto-sync pushes)."""
    candidates = [
        REPO_ROOT.parent / "REE_assembly",
        Path.home() / "Documents" / "GitHub" / "REE_Working" / "REE_assembly",
    ]
    for c in candidates:
        if c.is_dir() and (c / "evidence" / "experiments").is_dir():
            return c
    return None


def git_pull(repo_path: Path, label: str) -> None:
    """Pull latest changes. Retries on transient lock errors. Never raises."""
    import time
    _LOCK_HINTS = ("cannot lock ref", "unable to resolve reference",
                   "lock file", "index.lock")
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(repo_path), capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"
                print(f"[runner] git pull {label}: {msg}", flush=True)
                return
            stderr = r.stderr.strip()
            if any(h in stderr.lower() for h in _LOCK_HINTS) and attempt < 2:
                print(f"[runner] git pull {label}: transient lock, retrying "
                      f"({attempt + 1}/2)...", flush=True)
                time.sleep(2)
                continue
            print(f"[runner] git pull {label} warn: {stderr}", flush=True)
            return
        except Exception as e:
            print(f"[runner] git pull {label} error: {e}", flush=True)
            return


def git_push_results(ree_assembly_path: Path, result_files: list[str] | None = None) -> None:
    """Stage, commit, and push experiment results in REE_assembly.

    If result_files is provided, only those specific files are staged (selective
    commit).  Otherwise falls back to staging the entire evidence/experiments/
    directory.

    On push rejection, retries once with pull --rebase.  Never uses git reset --hard
    to avoid destroying uncommitted work from concurrent Claude sessions.

    Warns on failure; never raises.
    """
    try:
        if result_files:
            for f in result_files:
                try:
                    rel = str(Path(f).relative_to(ree_assembly_path))
                except ValueError:
                    rel = f
                subprocess.run(
                    ["git", "add", rel],
                    cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
                )
        else:
            subprocess.run(
                ["git", "add", "evidence/experiments/"],
                cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
            )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ree_assembly_path), timeout=5,
        )
        if diff.returncode == 0:
            print("[runner] auto-sync: nothing new to push", flush=True)
            return
        msg = f"auto-sync: v2 results {datetime.now(timezone.utc).isoformat()[:10]}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=15,
        )
        r = subprocess.run(
            ["git", "push", "origin", "HEAD:master"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print("[runner] auto-sync: pushed results -> REE_assembly", flush=True)
        elif "non-fast-forward" in r.stderr or "fetch first" in r.stderr:
            # Retry once with pull --rebase
            pull = subprocess.run(
                ["git", "pull", "--rebase", "origin", "master"],
                cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=30,
            )
            if pull.returncode == 0:
                r2 = subprocess.run(
                    ["git", "push", "origin", "HEAD:master"],
                    cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=30,
                )
                if r2.returncode == 0:
                    print("[runner] auto-sync: pushed results -> REE_assembly (after rebase)",
                          flush=True)
                else:
                    print(f"[runner] auto-sync push warn (retry): {r2.stderr.strip()}", flush=True)
            else:
                # Rebase failed -- abort and skip; don't use git reset --hard
                subprocess.run(["git", "rebase", "--abort"],
                               cwd=str(ree_assembly_path), capture_output=True, timeout=10)
                print(f"[runner] auto-sync: rebase conflict -- skipping push (will retry next sync)",
                      flush=True)
        else:
            print(f"[runner] auto-sync push warn: {r.stderr.strip()}", flush=True)
    except Exception as e:
        print(f"[runner] auto-sync push error: {e}", flush=True)


# ── Multi-machine coordination ────────────────────────────────────────────────

CLAIM_TTL_HOURS = 6


def _get_machine_name(override: str | None = None) -> str:
    return override or socket.gethostname()


def _affinity_matches(item: dict, machine: str) -> bool:
    affinity = item.get("machine_affinity", "any")
    return affinity in ("any", None, "") or affinity == machine


def _is_stale_claim(claimed_by: dict) -> bool:
    try:
        claimed_at = datetime.fromisoformat(claimed_by["claimed_at"])
        age = datetime.now(timezone.utc) - claimed_at
        return age.total_seconds() > CLAIM_TTL_HOURS * 3600
    except Exception:
        return True


def _git_undo_last_commit(repo: Path) -> None:
    subprocess.run(["git", "reset", "--soft", "HEAD~1"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "reset", "HEAD", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "checkout", "--", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)


def attempt_claim(queue_file: Path, queue_id: str, machine: str) -> str:
    repo = queue_file.parent
    try:
        r = subprocess.run(["git", "pull", "--ff-only"],
                           cwd=str(repo), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"[runner] claim pull warn ({queue_id}): {r.stderr.strip()}", flush=True)

        data = json.loads(queue_file.read_text())
        item = next((i for i in data.get("items", []) if i["queue_id"] == queue_id), None)
        if item is None:
            return "error"

        existing = item.get("claimed_by")
        if existing and existing.get("machine") != machine and not _is_stale_claim(existing):
            return "already_claimed"

        if not _affinity_matches(item, machine):
            return "already_claimed"

        item["claimed_by"] = {"machine": machine, "claimed_at": datetime.now(timezone.utc).isoformat()}
        item["status"] = "claimed"
        queue_file.write_text(json.dumps(data, indent=2))

        subprocess.run(["git", "add", queue_file.name],
                       cwd=str(repo), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", f"claim: {queue_id} → {machine}"],
                       cwd=str(repo), capture_output=True, check=True)

        push = subprocess.run(["git", "push", "origin", "HEAD:main"],
                               cwd=str(repo), capture_output=True, text=True, timeout=30)
        if push.returncode == 0:
            return "ok"

        _git_undo_last_commit(repo)
        stderr = push.stderr.lower()
        if "non-fast-forward" in stderr or "rejected" in stderr:
            return "already_claimed"
        print(f"[runner] claim push error ({queue_id}): {push.stderr.strip()}", flush=True)
        return "error"

    except Exception as e:
        print(f"[runner] claim exception ({queue_id}): {e}", flush=True)
        try:
            _git_undo_last_commit(repo)
        except Exception:
            pass
        return "error"


def release_claim(queue_file: Path, queue_id: str, machine: str) -> None:
    repo = queue_file.parent
    try:
        subprocess.run(["git", "pull", "--ff-only"],
                       cwd=str(repo), capture_output=True, timeout=30)
        data = json.loads(queue_file.read_text())
        changed = False
        for item in data.get("items", []):
            if item["queue_id"] == queue_id:
                cb = item.get("claimed_by")
                if cb and cb.get("machine") == machine:
                    item["claimed_by"] = None
                    item["status"] = "pending"
                    changed = True
                break
        if not changed:
            return
        queue_file.write_text(json.dumps(data, indent=2))
        subprocess.run(["git", "add", queue_file.name], cwd=str(repo), capture_output=True)
        subprocess.run(["git", "commit", "-m",
                        f"release claim: {queue_id} ← {machine} (shutdown)"],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "push", "origin", "HEAD:main"],
                       cwd=str(repo), capture_output=True, timeout=30)
        print(f"[runner] Released claim on {queue_id}", flush=True)
    except Exception as e:
        print(f"[runner] Release claim error ({queue_id}): {e}", flush=True)


def recover_stale_claims(queue_file: Path, machine: str) -> int:
    # STUB: see ree-v3/experiment_runner.py for full docstring
    try:
        data = json.loads(queue_file.read_text())
        stale = [(i["queue_id"], i["claimed_by"]["machine"])
                 for i in data.get("items", [])
                 if i.get("claimed_by")
                 and i["claimed_by"].get("machine") != machine
                 and _is_stale_claim(i["claimed_by"])]
        if stale:
            print(f"[runner] Stale claims (not auto-recovering): {[q for q, _ in stale]}",
                  flush=True)
        return len(stale)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────

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


def load_script_timing() -> dict:
    """Load per-script ms/episode-condition timing learned from past runs."""
    if SCRIPT_TIMING_FILE.exists():
        try:
            return json.loads(SCRIPT_TIMING_FILE.read_text())
        except Exception:
            pass
    return {}


def save_script_timing(script: str, actual_secs: float, seeds: int, conditions: int, episodes: int) -> None:
    """Back-calculate ms/episode-condition from a completed run and persist it."""
    total_ep_cond = seeds * conditions * episodes
    if total_ep_cond <= 0:
        return
    actual_ms_per = round((actual_secs * 1000) / total_ep_cond, 1)
    timing = load_script_timing()
    timing[script] = actual_ms_per
    SCRIPT_TIMING_FILE.write_text(json.dumps(timing, indent=2))
    print(f"[runner] Calibration updated: {script} → {actual_ms_per:.0f} ms/ep-cond", flush=True)


def estimate_minutes(item: dict, calibration: dict, script_timing: dict | None = None) -> float:
    seeds = item.get("seeds", 3)
    conditions = item.get("conditions", 2)
    episodes = item.get("episodes_per_run", 200)
    # Use per-script learned rate if available (fix 2), else global calibration constant
    script = item.get("script", "")
    if script_timing and script in script_timing:
        ms_per = script_timing[script]
    else:
        ms_per = calibration.get("ms_per_episode_condition", 750)
    return (seeds * conditions * episodes * ms_per) / 60_000


def build_initial_status(queue_data: dict, script_timing: dict | None = None) -> dict:
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
            "estimated_minutes": round(estimate_minutes(item, calibration, script_timing), 1),
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


def run_experiment(item: dict, status: dict, status_path: Path, calibration: dict, script_timing: dict | None = None) -> dict:
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
    run_end_times: list[float] = []  # fix 3: monotonic timestamp when each run completes

    started_at = time.monotonic()
    started_at_utc = now_utc()

    def overall_pct() -> float:
        run_frac = (runs_done + episodes_in_run / max(episodes_per_run, 1)) / max(total_runs, 1)
        return round(run_frac * 100, 1)

    def seconds_remaining() -> float:
        elapsed = time.monotonic() - started_at
        pct = overall_pct()
        static_secs = estimate_minutes(item, calibration, script_timing) * 60

        if pct <= 0:
            return static_secs

        # Fix 3: once at least one full run has completed, use average run duration
        # to project remaining time — captures per-run overhead better than pure
        # episode extrapolation.
        if run_end_times:
            avg_secs_per_run = run_end_times[-1] / runs_done  # time to Nth run / N runs
            remaining_run_units = total_runs - runs_done - episodes_in_run / max(episodes_per_run, 1)
            return max(0, avg_secs_per_run * remaining_run_units)

        # Fix 1: blend static estimate → live extrapolation over the first 20%.
        # At pct=0 we rely entirely on the static estimate; by pct=20 we trust
        # the live pace entirely. This kills the wild swings in the early phase
        # before enough data has accumulated for a stable extrapolation.
        live_total = elapsed / (pct / 100)
        blend = min(pct / 20.0, 1.0)
        total_estimated = (1 - blend) * static_secs + blend * live_total
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

    result_info = {"result": "UNKNOWN", "result_summary": "", "started_at": started_at_utc, "completed_at": "", "output_file": "", "actual_secs": 0.0}

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

            if any(p.search(line) for p in RE_RUN_DONE_PATTERNS):
                run_end_times.append(time.monotonic() - started_at)  # fix 3
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
    result_info["actual_secs"] = round(time.monotonic() - started_at, 1)

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
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help="Git-pull queue repo before each batch; git-push results to REE_assembly after. "
             "Also enables git-based experiment claiming (multi-machine coordination).",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine identity for experiment claiming (default: hostname).",
    )
    args = parser.parse_args()

    machine = _get_machine_name(args.machine)
    status_path = args.status_file or find_default_status_path()
    ree_assembly_path = find_ree_assembly_path()
    print(f"[runner] Machine identity: {machine}", flush=True)
    print(f"[runner] Status file: {status_path}", flush=True)
    print(f"[runner] Queue file: {QUEUE_FILE}", flush=True)
    if args.auto_sync:
        if ree_assembly_path:
            print(f"[runner] Auto-sync: ON (REE_assembly: {ree_assembly_path})", flush=True)
            git_pull(REPO_ROOT, "ree-v2")
            git_pull(ree_assembly_path, "REE_assembly")
        else:
            print("[runner] Auto-sync: ON but REE_assembly not found — sync disabled", flush=True)
        recover_stale_claims(QUEUE_FILE, machine)

    # Write PID file
    PID_FILE.write_text(str(os.getpid()))

    _current_claim: list[str] = []

    # Handle signals gracefully
    def handle_signal(sig, frame):
        print(f"\n[runner] Caught signal {sig}, shutting down.", flush=True)
        if args.auto_sync and _current_claim:
            release_claim(QUEUE_FILE, _current_claim[0], machine)
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
    script_timing = load_script_timing()  # fix 2: per-script learned rates
    if script_timing:
        print(f"[runner] Loaded per-script timing for {len(script_timing)} script(s)", flush=True)

    # Load existing status if present (to preserve completed list)
    existing_completed = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
            existing_completed = existing.get("completed", [])
        except Exception:
            pass

    status = build_initial_status(queue_data, script_timing)
    status["completed"] = existing_completed
    write_status(status, status_path)

    if args.dry_run:
        print("[runner] Dry run — queue:")
        for item in items:
            script = REPO_ROOT / item["script"]
            runnable = script.exists()
            mins = estimate_minutes(item, calibration, script_timing)
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

    _result_files_this_pass: list[str] = []

    while True:
        ran_any = False

        for item in items:
            queue_id = item["queue_id"]

            # Skip already completed
            if queue_id in completed_ids:
                continue

            # Skip experiments assigned to a different machine
            if not _affinity_matches(item, machine):
                print(f"[runner] Skipping {queue_id} — affinity={item.get('machine_affinity')} "
                      f"(this machine: {machine})", flush=True)
                continue

            # Skip experiments claimed by another active machine
            existing_claim = item.get("claimed_by")
            if (existing_claim
                    and existing_claim.get("machine") != machine
                    and not _is_stale_claim(existing_claim)):
                print(f"[runner] Skipping {queue_id} — claimed by "
                      f"{existing_claim['machine']}", flush=True)
                continue

            # Skip items with no script
            script = REPO_ROOT / item["script"]
            if not script.exists():
                print(f"[runner] Skipping {queue_id} ({item['claim_id']}) — script not found: {item['script']}", flush=True)
                for qi in status["queue"]:
                    if qi["queue_id"] == queue_id:
                        qi["status"] = "needs_script"
                write_status(status, status_path)
                continue

            # In auto-sync mode, use git claim as mutex before running
            if args.auto_sync:
                claim_result = attempt_claim(QUEUE_FILE, queue_id, machine)
                if claim_result == "already_claimed":
                    print(f"[runner] {queue_id} — claim lost to another machine, skipping",
                          flush=True)
                    continue
                if claim_result == "error":
                    print(f"[runner] {queue_id} — claim push failed (network?), "
                          f"running anyway", flush=True)
                _current_claim.clear()
                _current_claim.append(queue_id)

            # Run it
            result = run_experiment(item, status, status_path, calibration, script_timing)
            ran_any = True
            _current_claim.clear()

            # Collect output file for selective git staging
            if result.get("output_file"):
                _result_files_this_pass.append(result["output_file"])

            # Fix 2: update per-script calibration from actual run time
            if result["result"] not in ("ERROR", "UNKNOWN") and result.get("actual_secs"):
                save_script_timing(
                    item["script"],
                    result["actual_secs"],
                    item.get("seeds", 3),
                    item.get("conditions", 2),
                    item.get("episodes_per_run", 200),
                )
                script_timing = load_script_timing()  # reload so next queue item uses updated rate

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

        if args.auto_sync and ran_any and ree_assembly_path:
            git_push_results(ree_assembly_path, _result_files_this_pass or None)
            _result_files_this_pass.clear()

        # Loop mode: wait, then reload queue for any newly-added items
        status["idle"] = True
        status["current"] = None
        write_status(status, status_path)
        if ran_any:
            print(f"[runner] Pass complete. Waiting {args.loop_interval}s before re-checking queue…", flush=True)
        else:
            print(f"[runner] No new items. Waiting {args.loop_interval}s…", flush=True)

        time.sleep(args.loop_interval)

        if args.auto_sync and ree_assembly_path:
            git_pull(REPO_ROOT, "ree-v2")

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
                    "estimated_minutes": round(estimate_minutes(i, calibration, script_timing), 1),
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

    if args.auto_sync and ree_assembly_path:
        git_push_results(ree_assembly_path, _result_files_this_pass or None)

    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
