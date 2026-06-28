#!/usr/bin/env bash
set -euo pipefail

# Run K=24,D=704 improvement experiments sequentially to avoid GPU OOM.
# Order: Run 2 -> Run 3 -> Run 4 -> Run 5.
# Each child script trains with random_ellipses val500 as primary validation and
# Shepp-Logan random100 as secondary validation by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OFFLINE_DATA_DIR=""
SECONDARY_VAL_SUBSAMPLE_SIZE=100
DRY_RUN=0

usage() {
    cat <<USAGE
Usage: bash $(basename "$0") [options] [-- extra-options-for-each-run]

Sequentially runs:
  run2_1stage_weakphys_rmax005
  run3_1stage_tinyphys_rmax003
  run4_1stage_weak_stage_dc
  run5_2stage_tied_small_nodc

Options:
  --project-root PATH              Project root containing models/ and data/
  --python-bin PATH                Python executable. Default: \$PYTHON_BIN or python
  --offline-data-dir PATH          Directory containing train/val/test offline .pt files
  --secondary-val-subsample-size N Shepp-Logan secondary validation random subset size. Default: 100
  --dry-run                        Print commands and run each child script in dry-run mode
  -h, --help                       Show this help

Any arguments after -- are forwarded unchanged to every child run script.

Example:
  bash /root/run_detector_KD/24_angle_det704_improve/run2_to_run5_sequential.sh \
    --project-root /root \
    --python-bin python \
    --offline-data-dir /root/run_detector_KD/24_angle_det704/offline_data

Background example:
  nohup bash /root/run_detector_KD/24_angle_det704_improve/run2_to_run5_sequential.sh \
    --project-root /root \
    --python-bin python \
    > /root/run_detector_KD/24_angle_det704_improve/sequential_run2_to_run5.out 2>&1 &
USAGE
}

FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --offline-data-dir) OFFLINE_DATA_DIR="$2"; shift 2 ;;
        --secondary-val-subsample-size) SECONDARY_VAL_SUBSAMPLE_SIZE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; FORWARDED_ARGS+=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PROJECT_ROOT" ]]; then
    if [[ -d "${DEFAULT_PROJECT_ROOT}/models" && -d "${DEFAULT_PROJECT_ROOT}/data" ]]; then
        PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
    elif [[ -d "/root/models" && -d "/root/data" ]]; then
        PROJECT_ROOT="/root"
    else
        PROJECT_ROOT="$DEFAULT_PROJECT_ROOT"
    fi
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi
if ! [[ "$SECONDARY_VAL_SUBSAMPLE_SIZE" =~ ^[0-9]+$ ]] || [[ "$SECONDARY_VAL_SUBSAMPLE_SIZE" -le 0 ]]; then
    echo "secondary-val-subsample-size must be a positive integer." >&2
    exit 1
fi

RUNS=(
    "run2_1stage_weakphys_rmax005"
    "run3_1stage_tinyphys_rmax003"
    "run4_1stage_weak_stage_dc"
    "run5_2stage_tied_small_nodc"
)

COMMON_ARGS=(
    --project-root "$PROJECT_ROOT"
    --python-bin "$PYTHON_BIN"
    --secondary-val-subsample-size "$SECONDARY_VAL_SUBSAMPLE_SIZE"
)
if [[ -n "$OFFLINE_DATA_DIR" ]]; then
    COMMON_ARGS+=(--offline-data-dir "$OFFLINE_DATA_DIR")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    COMMON_ARGS+=(--dry-run)
fi
if [[ "${#FORWARDED_ARGS[@]}" -gt 0 ]]; then
    COMMON_ARGS+=("${FORWARDED_ARGS[@]}")
fi

echo "[sequential] project=${PROJECT_ROOT}"
echo "[sequential] python=${PYTHON_BIN}"
echo "[sequential] offline_data_dir=${OFFLINE_DATA_DIR:-default}"
echo "[sequential] secondary_val_subsample_size=${SECONDARY_VAL_SUBSAMPLE_SIZE}"
echo "[sequential] dry_run=${DRY_RUN}"
echo "[sequential] runs=${RUNS[*]}"

for run_name in "${RUNS[@]}"; do
    run_script="${SCRIPT_DIR}/${run_name}.sh"
    run_log_dir="${SCRIPT_DIR}/${run_name}/log"
    run_outer_log="${run_log_dir}/sequential_train.out"

    if [[ ! -f "$run_script" ]]; then
        echo "[sequential] Missing run script: $run_script" >&2
        exit 1
    fi
    mkdir -p "$run_log_dir"

    echo "[sequential] ===== START ${run_name} $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "[sequential] command: bash ${run_script} ${COMMON_ARGS[*]}"

    if ! bash "$run_script" "${COMMON_ARGS[@]}" 2>&1 | tee "$run_outer_log"; then
        echo "[sequential] ===== FAILED ${run_name} $(date '+%Y-%m-%d %H:%M:%S') =====" >&2
        echo "[sequential] See outer log: $run_outer_log" >&2
        exit 1
    fi

    echo "[sequential] ===== DONE ${run_name} $(date '+%Y-%m-%d %H:%M:%S') ====="
done

echo "[sequential] All requested runs completed."
