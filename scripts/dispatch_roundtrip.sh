#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILE="all"
MACHINE="macbook_air_m2_2022"
OUT_DIR="${REPO_ROOT}/jobs/outgoing"
INCOMING_DIR="${REPO_ROOT}/jobs/incoming"
RUNS_ROOT="${REPO_ROOT}/evidence/experiments"
DRY_RUN=0
STRICT_QUEUE=1
SKIP_VALIDATE=0

usage() {
  cat <<EOF
Usage: scripts/dispatch_roundtrip.sh [options]

Options:
  --profile <name>         Profile name or 'all' (default: all)
  --machine <name>         Machine policy key (default: macbook_air_m2_2022)
  --out-dir <path>         Outgoing job spec dir (default: jobs/outgoing)
  --incoming-dir <path>    Inbound bundle dir (default: jobs/incoming)
  --runs-root <path>       Experiment runs root (default: evidence/experiments)
  --dry-run                Validate/preview dispatch flow without importing bundles
  --no-strict-queue        Allow blocked inbound bundles without non-zero exit
  --skip-validate          Skip validate_experiment_pack.py at the end
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --machine)
      MACHINE="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --incoming-dir)
      INCOMING_DIR="$2"
      shift 2
      ;;
    --runs-root)
      RUNS_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-strict-queue)
      STRICT_QUEUE=0
      shift
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[1/5] Build remote job specs"
python3 "${REPO_ROOT}/scripts/build_remote_job_spec.py" \
  --profile "${PROFILE}" \
  --machine "${MACHINE}" \
  --out-dir "${OUT_DIR}"

echo "[2/5] Submit remote job specs"
SUBMIT_ARGS=(
  "${REPO_ROOT}/scripts/submit_remote_job.py"
  --job-spec-dir "${OUT_DIR}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  SUBMIT_ARGS+=(--dry-run)
fi
python3 "${SUBMIT_ARGS[@]}"

echo "[3/5] Check inbound handoff queue"
QUEUE_ARGS=(
  "${REPO_ROOT}/scripts/check_handoff_queue.py"
  --incoming-dir "${INCOMING_DIR}"
  --runs-root "${RUNS_ROOT}"
)
if [[ "${STRICT_QUEUE}" -eq 1 ]]; then
  QUEUE_ARGS+=(--strict)
fi
python3 "${QUEUE_ARGS[@]}"

echo "[4/5] Pull remote results"
PULL_ARGS=(
  "${REPO_ROOT}/scripts/pull_remote_results.py"
  --job-run-dir "${INCOMING_DIR}"
  --runs-root "${RUNS_ROOT}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  PULL_ARGS+=(--dry-run)
fi
python3 "${PULL_ARGS[@]}"

if [[ "${SKIP_VALIDATE}" -eq 1 ]]; then
  echo "[5/5] Validate run packs (skipped)"
  exit 0
fi

echo "[5/5] Validate run packs"
python3 "${REPO_ROOT}/scripts/validate_experiment_pack.py" --runs-root "${RUNS_ROOT}"

