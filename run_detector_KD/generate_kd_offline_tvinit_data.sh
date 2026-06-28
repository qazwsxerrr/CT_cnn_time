#!/usr/bin/env bash
set -euo pipefail

# Generate offline TV-init datasets for KD-threshold sparse-detector cases.
# Defaults: K=24,D=704. Other default pairs: K=8,D=2048 and K=16,D=1024.

K=24
DETECTOR_SAMPLES=0
NOISE_LEVEL="0.1"
TRAIN_SAMPLES=8000
VAL_SAMPLES=500
TEST_SAMPLES=500
TRAIN_BATCH_SIZE=4
VAL_BATCH_SIZE=4
TEST_BATCH_SIZE=4
SEED_OFFSET=0
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -d "${DEFAULT_PROJECT_ROOT}/models" && -d "${DEFAULT_PROJECT_ROOT}/data" ]]; then
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
elif [[ -d "/root/models" && -d "/root/data" ]]; then
    PROJECT_ROOT="/root"
else
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi

usage() {
    cat <<USAGE
Usage: bash $(basename "$0") [options]

Options:
  --k 8|16|24                 Number of selected angles. Default: 24
  --detector-samples N         Detector samples per angle. Default by K: 8->2048, 16->1024, 24->704
  --noise-level X              Multiplicative noise level. Default: 0.1
  --train-samples N            Train random_ellipses samples. Default: 8000
  --val-samples N              Validation random_ellipses samples. Default: 500
  --test-samples N             Test shepp_logan samples. Default: 500
  --train-batch-size N         Offline generation batch size. Default: 4
  --val-batch-size N           Offline generation batch size. Default: 4
  --test-batch-size N          Offline generation batch size. Default: 4
  --seed-offset N              First random seed offset. Default: 0
  --output-dir PATH            Output directory. Default: run_detector_KD/{K}_angle_det{D}/offline_data
  --project-root PATH          Project root containing models/ and data/. Default: parent of this script, or /root when detected
  --python-bin PATH            Python executable. Default: \$PYTHON_BIN or python
  --dry-run                    Prepare and print command without generating .pt files
  -h, --help                   Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k) K="$2"; shift 2 ;;
        --detector-samples) DETECTOR_SAMPLES="$2"; shift 2 ;;
        --noise-level) NOISE_LEVEL="$2"; shift 2 ;;
        --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
        --val-samples) VAL_SAMPLES="$2"; shift 2 ;;
        --test-samples) TEST_SAMPLES="$2"; shift 2 ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --val-batch-size) VAL_BATCH_SIZE="$2"; shift 2 ;;
        --test-batch-size) TEST_BATCH_SIZE="$2"; shift 2 ;;
        --seed-offset) SEED_OFFSET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$K" in
    8) DEFAULT_DETECTOR_SAMPLES=2048; ALPHA_JSON_REL="data/alpha8_tv/alpha_selected8_dopt_hard_gap_16_30.json" ;;
    16) DEFAULT_DETECTOR_SAMPLES=1024; ALPHA_JSON_REL="data/alpha16_tv/alpha_selected16_dopt_hard_gap_9_14.json" ;;
    24) DEFAULT_DETECTOR_SAMPLES=704; ALPHA_JSON_REL="data/alpha24_tv/alpha_selected24_dopt_hard_gap_6_9.json" ;;
    *) echo "Unsupported K=$K. Expected 8, 16, or 24." >&2; exit 1 ;;
esac
if [[ "$DETECTOR_SAMPLES" == "0" ]]; then
    DETECTOR_SAMPLES="$DEFAULT_DETECTOR_SAMPLES"
fi

for item in \
    "DETECTOR_SAMPLES:$DETECTOR_SAMPLES" \
    "TRAIN_SAMPLES:$TRAIN_SAMPLES" \
    "VAL_SAMPLES:$VAL_SAMPLES" \
    "TEST_SAMPLES:$TEST_SAMPLES" \
    "TRAIN_BATCH_SIZE:$TRAIN_BATCH_SIZE" \
    "VAL_BATCH_SIZE:$VAL_BATCH_SIZE" \
    "TEST_BATCH_SIZE:$TEST_BATCH_SIZE"; do
    name="${item%%:*}"
    value="${item#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        echo "$name must be a positive integer, got $value" >&2
        exit 1
    fi
done

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"

if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

ALPHA_JSON="${PROJECT_ROOT}/${ALPHA_JSON_REL}"
if [[ ! -f "$ALPHA_JSON" ]]; then
    echo "Selected-angle JSON not found: $ALPHA_JSON" >&2
    exit 1
fi

NOISE_TEXT="$NOISE_LEVEL"
NOISE_TAG="${NOISE_TEXT//./p}"
NOISE_TAG="${NOISE_TAG//-/m}"
if [[ -z "$OUTPUT_DIR" ]]; then
    RESOLVED_OUTPUT_DIR="${SCRIPT_DIR}/${K}_angle_det${DETECTOR_SAMPLES}/offline_data"
elif [[ "$OUTPUT_DIR" == /* ]]; then
    RESOLVED_OUTPUT_DIR="$OUTPUT_DIR"
else
    RESOLVED_OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi
mkdir -p "$RESOLVED_OUTPUT_DIR"

FILE_SUFFIX="alpha${K}_noise${NOISE_TAG}_det${DETECTOR_SAMPLES}_edgewsubset"
TRAIN_PATH="${RESOLVED_OUTPUT_DIR}/train${TRAIN_SAMPLES}_random_ellipses_tvinit_${FILE_SUFFIX}.pt"
VAL_PATH="${RESOLVED_OUTPUT_DIR}/val${VAL_SAMPLES}_random_ellipses_tvinit_${FILE_SUFFIX}.pt"
TEST_PATH="${RESOLVED_OUTPUT_DIR}/test${TEST_SAMPLES}_shepp_logan_tvinit_${FILE_SUFFIX}.pt"
MEASUREMENT_COUNT=$((K * DETECTOR_SAMPLES))

set_run_env() { export "$1=$2"; }

set_run_env "PROJECT_ROOT" "$PROJECT_ROOT"
set_run_env "PYTHONUTF8" "1"
set_run_env "PYTHON_BIN" "$PYTHON_BIN"
set_run_env "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
set_run_env "ALPHA_CONDITION_TOP_K_OVERRIDE" "$K"
set_run_env "ALPHA_CONDITION_JSON_OVERRIDE" "$ALPHA_JSON"
set_run_env "ALPHA_GRAM_CACHE_DIR_OVERRIDE" "${PROJECT_ROOT}/data/alpha_gram_cache"
set_run_env "NUM_ANGLES_TOTAL_OVERRIDE" "$K"
set_run_env "MULTI_ANGLE_SOLVER_MODE_OVERRIDE" "stacked_tikhonov"
set_run_env "THEORETICAL_FORMULA_MODE_OVERRIDE" "alpha_continuous"
set_run_env "SAMPLING_MODE_OVERRIDE" "shifted_lattice_edge_weighted_subset"
set_run_env "NUM_DETECTOR_SAMPLES_OVERRIDE" "$DETECTOR_SAMPLES"
set_run_env "DETECTOR_PHASE_OVERRIDE" "0.5"
set_run_env "DETECTOR_MARGIN_RATIO_OVERRIDE" "0.0"
set_run_env "INIT_METHOD_OVERRIDE" "l2_tv_admm"
set_run_env "LAMBDA_SELECT_MODE_OVERRIDE" "morozov"
set_run_env "MOROZOV_FORM_OVERRIDE" "constrained"
set_run_env "MOROZOV_NOISE_RADIUS_MODE_OVERRIDE" "rms"
set_run_env "MOROZOV_TAU_OVERRIDE" "1.0"
set_run_env "L1_INIT_ADMM_ITERS_OVERRIDE" "80"
set_run_env "L1_INIT_ADMM_CG_ITERS_OVERRIDE" "30"
set_run_env "L1_INIT_ADMM_CG_TOL_OVERRIDE" "1e-4"
set_run_env "L1_INIT_ADMM_RHO_DATA_OVERRIDE" "1.0"
set_run_env "L1_INIT_ADMM_RHO_REG_OVERRIDE" "1.0"
set_run_env "REGULARIZER_TYPE_OVERRIDE" "dirichlet"
set_run_env "NOISE_MODE_OVERRIDE" "multiplicative"
set_run_env "NOISE_LEVEL_OVERRIDE" "$NOISE_TEXT"

echo "[offline] project=${PROJECT_ROOT}"
echo "[offline] python=${PYTHON_BIN}"
echo "[offline] K=${K} detector_samples=${DETECTOR_SAMPLES} measurements=${MEASUREMENT_COUNT} sampling=shifted_lattice_edge_weighted_subset"
echo "[offline] alpha_json=${ALPHA_JSON}"
echo "[offline] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
echo "[offline] noise=multiplicative delta=${NOISE_TEXT}"
echo "[offline] train=${TRAIN_SAMPLES} val=${VAL_SAMPLES} test=${TEST_SAMPLES}"
echo "[offline] batch train=${TRAIN_BATCH_SIZE} val=${VAL_BATCH_SIZE} test=${TEST_BATCH_SIZE}"
echo "[offline] output_dir=${RESOLVED_OUTPUT_DIR}"
echo "[offline] train_output=${TRAIN_PATH}"
echo "[offline] val_output=${VAL_PATH}"
echo "[offline] test_output=${TEST_PATH}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[offline] DryRun enabled; environment prepared but dataset generation was not started."
    exit 0
fi

exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/data_genoration/offline_tvinit_data.py" \
    --train-val-test-splits \
    --num-angles "$K" \
    --alpha-json "$ALPHA_JSON" \
    --seed-offset "$SEED_OFFSET" \
    --train-output "$TRAIN_PATH" \
    --val-output "$VAL_PATH" \
    --test-output "$TEST_PATH" \
    --train-samples "$TRAIN_SAMPLES" \
    --val-random-ellipses-samples "$VAL_SAMPLES" \
    --test-shepp-logan-samples "$TEST_SAMPLES" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --val-batch-size "$VAL_BATCH_SIZE" \
    --test-batch-size "$TEST_BATCH_SIZE"
