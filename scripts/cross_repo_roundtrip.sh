#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REE_V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REE_ASSEMBLY_ROOT="/Users/dgolden/Documents/GitHub/REE_assembly"
PROFILE="all"
CONDITION_MODE="all_conditions"
SEEDS="11,29,47"
BACKEND="internal_minimal"
MAX_ABS_DELTA="1e-6"
WEEKLY_HANDOFF_OUT="evidence/planning/weekly_handoff/latest.md"
DISPATCH_DATE="auto"

DRY_RUN=0
SKIP_QUALIFICATION=0
SKIP_SYNC=0
SKIP_DISPATCH_EMIT=0
SKIP_DISPATCH_PULL=0
SKIP_PACK_VALIDATE=0

usage() {
  cat <<EOF
Usage: scripts/cross_repo_roundtrip.sh [options]

Run full ree-v2 <-> REE_assembly roundtrip:
1) ree-v2 qualification + handoff emission
2) REE_assembly handoff sync + ingestion
3) REE_assembly outbound dispatch emission
4) pull ree-v2 dispatch packet back into ree-v2 incoming dispatch location

Options:
  --ree-assembly-root <path>   REE_assembly repo root
                               (default: /Users/dgolden/Documents/GitHub/REE_assembly)
  --profile <name>             Qualification profile (default: all)
  --condition-mode <mode>      supports_only|all_conditions (default: all_conditions)
  --seeds <csv>                Comma-separated seeds (default: 11,29,47)
  --backend <name>             internal_minimal (default: internal_minimal)
  --max-abs-delta <float>      Determinism threshold (default: 1e-6)
  --weekly-handoff-out <path>  ree-v2 weekly handoff output path relative to ree-v2 root
                               (default: evidence/planning/weekly_handoff/latest.md)
  --dispatch-date <YYYY-MM-DD|auto>
                               Date folder for outbound dispatch pullback
                               (default: auto -> latest available)
  --skip-qualification         Skip ree-v2 qualification/gate steps
  --skip-sync                  Skip REE_assembly sync_weekly_handoffs step
  --skip-dispatch-emit         Skip REE_assembly emit_weekly_dispatches step
  --skip-dispatch-pull         Skip copying ree-v2 dispatch bundle back to ree-v2
  --skip-pack-validate         Skip final ree-v2 validate_experiment_pack
  --dry-run                    Print commands without executing
  -h, --help                   Show help
EOF
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %q' "$1"
    shift
    for token in "$@"; do
      printf ' %q' "${token}"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ree-assembly-root)
      REE_ASSEMBLY_ROOT="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --condition-mode)
      CONDITION_MODE="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --max-abs-delta)
      MAX_ABS_DELTA="$2"
      shift 2
      ;;
    --weekly-handoff-out)
      WEEKLY_HANDOFF_OUT="$2"
      shift 2
      ;;
    --dispatch-date)
      DISPATCH_DATE="$2"
      shift 2
      ;;
    --skip-qualification)
      SKIP_QUALIFICATION=1
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    --skip-dispatch-emit)
      SKIP_DISPATCH_EMIT=1
      shift
      ;;
    --skip-dispatch-pull)
      SKIP_DISPATCH_PULL=1
      shift
      ;;
    --skip-pack-validate)
      SKIP_PACK_VALIDATE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ ! -d "${REE_ASSEMBLY_ROOT}" ]]; then
  echo "REE_assembly root not found: ${REE_ASSEMBLY_ROOT}" >&2
  exit 2
fi

if [[ "${SKIP_QUALIFICATION}" -eq 0 ]]; then
  echo "[1/7] ree-v2 hook surface gate"
  run_cmd python3 "${REE_V2_ROOT}/scripts/validate_hook_surfaces.py" \
    --registry "${REE_V2_ROOT}/contracts/hook_registry.v1.json"

  echo "[2/7] ree-v2 determinism gate"
  run_cmd python3 "${REE_V2_ROOT}/scripts/check_seed_determinism.py" \
    --profile all \
    --max-abs-delta "${MAX_ABS_DELTA}"

  echo "[3/7] ree-v2 qualification batch"
  run_cmd python3 "${REE_V2_ROOT}/scripts/run_qualification_batch.py" \
    --profile "${PROFILE}" \
    --condition-mode "${CONDITION_MODE}" \
    --seeds "${SEEDS}" \
    --backend "${BACKEND}"

  echo "[4/7] ree-v2 weekly handoff generation"
  run_cmd python3 "${REE_V2_ROOT}/scripts/generate_weekly_handoff.py" \
    --output "${REE_V2_ROOT}/${WEEKLY_HANDOFF_OUT}"
fi

if [[ "${SKIP_SYNC}" -eq 0 ]]; then
  echo "[5/7] REE_assembly handoff sync + ingestion"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_cmd python3 "${REE_ASSEMBLY_ROOT}/evidence/planning/scripts/sync_weekly_handoffs.py" \
      --full-run \
      --run-ingestion \
      --dry-run \
      --skip-git-pull
  else
    run_cmd python3 "${REE_ASSEMBLY_ROOT}/evidence/planning/scripts/sync_weekly_handoffs.py" \
      --full-run \
      --run-ingestion
  fi
fi

if [[ "${SKIP_DISPATCH_EMIT}" -eq 0 ]]; then
  echo "[6/7] REE_assembly dispatch emission"
  run_cmd python3 "${REE_ASSEMBLY_ROOT}/evidence/planning/scripts/emit_weekly_dispatches.py"
fi

if [[ "${SKIP_DISPATCH_PULL}" -eq 0 ]]; then
  echo "[7/7] Pull ree-v2 dispatch bundle back into ree-v2"
  dispatch_root="${REE_ASSEMBLY_ROOT}/evidence/planning/outbound_dispatches"
  if [[ "${DISPATCH_DATE}" == "auto" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      dispatch_dir="${dispatch_root}/<latest-date>"
    else
      dispatch_dir="$(ls -1dt "${dispatch_root}"/* 2>/dev/null | head -n 1)"
      if [[ -z "${dispatch_dir}" ]]; then
        echo "No outbound dispatch directory found under ${dispatch_root}" >&2
        exit 1
      fi
    fi
  else
    dispatch_dir="${dispatch_root}/${DISPATCH_DATE}"
  fi

  dispatch_src="${dispatch_dir}/ree-v2_weekly_dispatch.md"
  incoming_dir="${REE_V2_ROOT}/evidence/planning/weekly_handoff/incoming_dispatches"
  incoming_day_dir="${incoming_dir}/$(basename "${dispatch_dir}")"

  run_cmd mkdir -p "${incoming_day_dir}"
  run_cmd cp "${dispatch_src}" "${incoming_dir}/latest.md"
  run_cmd cp "${dispatch_src}" "${incoming_day_dir}/ree-v2_weekly_dispatch.md"
fi

if [[ "${SKIP_PACK_VALIDATE}" -eq 0 ]]; then
  echo "[final] ree-v2 pack validation"
  run_cmd python3 "${REE_V2_ROOT}/scripts/validate_experiment_pack.py" \
    --runs-root "${REE_V2_ROOT}/evidence/experiments"
fi

echo "Done."
