#!/usr/bin/env bash
set -euo pipefail

# Evaluate KD-threshold RAD-UNet TV-init checkpoints.
# Defaults to the offline test500_shepp_logan_tvinit dataset for each K,D case.

K=24
DETECTOR_SAMPLES=0
NOISE_LEVEL="0.1"
OUTPUT_TAG=""
MODEL_CHOICE="best"
MODEL_PATH=""
NUM_SAMPLES=500
OFFLINE_BATCH_SIZE=20
OFFLINE_EVAL_DATASET=""
RESULT_PREFIX=""
RESULT_DIR=""
TEST_DATA_SOURCE="shepp_logan"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE_ONLINE=0
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
  --output-tag TAG             Output tag. Default: a{K}_d{D}_n{Noise}_offline
  --model-choice best|final    Model file to use when --model-path is empty. Default: best
  --model-path PATH            Explicit checkpoint path. Relative paths are resolved from project root
  --num-samples N              Number of samples to evaluate. Default: 500
  --offline-batch-size N       Batch size for offline evaluation. Default: 20
  --offline-eval-dataset PATH  Explicit offline eval .pt path; auto-detected when omitted
  --result-prefix PREFIX       Prefix for saved evaluation files. Default: output tag
  --result-dir PATH            Result directory. Default: run_detector_KD/{K}_angle_det{D}/result
  --test-data-source NAME      random_ellipses or shepp_logan. Default: shepp_logan
  --force-online               Ignore default offline .pt files and evaluate online
  --project-root PATH          Project root containing models/ and data/. Default: parent of this script, or /root when detected
  --python-bin PATH            Python executable. Default: \$PYTHON_BIN or python
  --dry-run                    Prepare and print environment without starting evaluation
  -h, --help                   Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k) K="$2"; shift 2 ;;
        --detector-samples) DETECTOR_SAMPLES="$2"; shift 2 ;;
        --noise-level) NOISE_LEVEL="$2"; shift 2 ;;
        --output-tag) OUTPUT_TAG="$2"; shift 2 ;;
        --model-choice) MODEL_CHOICE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --offline-batch-size) OFFLINE_BATCH_SIZE="$2"; shift 2 ;;
        --offline-eval-dataset) OFFLINE_EVAL_DATASET="$2"; shift 2 ;;
        --result-prefix) RESULT_PREFIX="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --test-data-source) TEST_DATA_SOURCE="$2"; shift 2 ;;
        --force-online) FORCE_ONLINE=1; shift ;;
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

case "$MODEL_CHOICE" in
    best|final) ;;
    *) echo "Unsupported model choice: $MODEL_CHOICE. Expected best or final." >&2; exit 1 ;;
esac
case "$TEST_DATA_SOURCE" in
    random_ellipses|shepp_logan) ;;
    *) echo "Unsupported test data source: $TEST_DATA_SOURCE. Expected random_ellipses or shepp_logan." >&2; exit 1 ;;
esac
for item in "DETECTOR_SAMPLES:$DETECTOR_SAMPLES" "NUM_SAMPLES:$NUM_SAMPLES" "OFFLINE_BATCH_SIZE:$OFFLINE_BATCH_SIZE"; do
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

CNN_ANGLE_INDICES="$(seq -s, 0 $((K - 1)))"
NOISE_TEXT="$NOISE_LEVEL"
OFFLINE_NOISE_TAG="${NOISE_TEXT//./p}"
OFFLINE_NOISE_TAG="${OFFLINE_NOISE_TAG//-/m}"
NOISE_TAG="n${OFFLINE_NOISE_TAG}"
if [[ -z "$OUTPUT_TAG" ]]; then
    OUTPUT_TAG="a${K}_d${DETECTOR_SAMPLES}_${NOISE_TAG}_offline"
fi

CASE_ROOT="${SCRIPT_DIR}/${K}_angle_det${DETECTOR_SAMPLES}"
LOG_DIR="${CASE_ROOT}/log"
DEFAULT_RESULT_DIR="${CASE_ROOT}/result"
CHECKPOINT_DIR="${CASE_ROOT}/checkpoints"
OFFLINE_DATA_DIR="${CASE_ROOT}/offline_data"

if [[ -z "$RESULT_DIR" ]]; then
    RESOLVED_RESULT_DIR="$DEFAULT_RESULT_DIR"
elif [[ "$RESULT_DIR" == /* ]]; then
    RESOLVED_RESULT_DIR="$RESULT_DIR"
else
    RESOLVED_RESULT_DIR="${PROJECT_ROOT}/${RESULT_DIR}"
fi

if [[ -z "$MODEL_PATH" ]]; then
    if [[ "$MODEL_CHOICE" == "best" ]]; then
        MODEL_FILE_KIND="best_model"
    else
        MODEL_FILE_KIND="model"
    fi
    DEFAULT_MODEL_PATH="${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_${MODEL_FILE_KIND}.pth"
    COMPACT_MODEL_PATH="${DEFAULT_MODEL_PATH%.pth}_compact.pth"
    if [[ -f "$COMPACT_MODEL_PATH" ]]; then
        RESOLVED_MODEL_PATH="$COMPACT_MODEL_PATH"
    else
        RESOLVED_MODEL_PATH="$DEFAULT_MODEL_PATH"
    fi
elif [[ "$MODEL_PATH" == /* ]]; then
    RESOLVED_MODEL_PATH="$MODEL_PATH"
else
    RESOLVED_MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
fi

if [[ -z "$RESULT_PREFIX" ]]; then
    RESOLVED_RESULT_PREFIX="$OUTPUT_TAG"
else
    RESOLVED_RESULT_PREFIX="$RESULT_PREFIX"
fi

FILE_SUFFIX="alpha${K}_noise${OFFLINE_NOISE_TAG}_det${DETECTOR_SAMPLES}_edgewsubset"
resolve_default_offline_dataset() {
    local directory="$1"
    local exact_name="$2"
    local pattern="$3"
    local exact_path="${directory}/${exact_name}"
    if [[ -f "$exact_path" ]]; then
        printf '%s\n' "$exact_path"
        return 0
    fi
    if [[ -d "$directory" ]]; then
        local matches=("$directory"/$pattern)
        if [[ -e "${matches[0]}" ]]; then
            local newest="${matches[0]}"
            local item
            for item in "${matches[@]}"; do
                if [[ "$item" -nt "$newest" ]]; then
                    newest="$item"
                fi
            done
            printf '%s\n' "$newest"
            return 0
        fi
    fi
    printf '\n'
}

if [[ "$FORCE_ONLINE" -eq 1 ]]; then
    RESOLVED_OFFLINE_EVAL=""
elif [[ -z "$OFFLINE_EVAL_DATASET" ]]; then
    if [[ "$TEST_DATA_SOURCE" == "shepp_logan" ]]; then
        RESOLVED_OFFLINE_EVAL="$(resolve_default_offline_dataset "$OFFLINE_DATA_DIR" "test500_shepp_logan_tvinit_${FILE_SUFFIX}.pt" "test*_shepp_logan_tvinit_${FILE_SUFFIX}.pt")"
    else
        RESOLVED_OFFLINE_EVAL="$(resolve_default_offline_dataset "$OFFLINE_DATA_DIR" "val500_random_ellipses_tvinit_${FILE_SUFFIX}.pt" "val*_random_ellipses_tvinit_${FILE_SUFFIX}.pt")"
    fi
elif [[ "$OFFLINE_EVAL_DATASET" == /* ]]; then
    RESOLVED_OFFLINE_EVAL="$OFFLINE_EVAL_DATASET"
else
    RESOLVED_OFFLINE_EVAL="${PROJECT_ROOT}/${OFFLINE_EVAL_DATASET}"
fi

if [[ -z "$RESOLVED_OFFLINE_EVAL" ]]; then
    USE_OFFLINE_EVAL=0
else
    USE_OFFLINE_EVAL=1
fi

mkdir -p "$LOG_DIR" "$RESOLVED_RESULT_DIR" "$CHECKPOINT_DIR"

set_run_env() { export "$1=$2"; }

set_run_env "PROJECT_ROOT" "$PROJECT_ROOT"
set_run_env "PYTHONUTF8" "1"
set_run_env "PYTHON_BIN" "$PYTHON_BIN"
set_run_env "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
set_run_env "ALPHA_CONDITION_TOP_K_OVERRIDE" "$K"
set_run_env "ALPHA_CONDITION_JSON_OVERRIDE" "$ALPHA_JSON"
set_run_env "ALPHA_GRAM_CACHE_DIR_OVERRIDE" "${PROJECT_ROOT}/data/alpha_gram_cache"
set_run_env "NUM_ANGLES_TOTAL_OVERRIDE" "$K"
set_run_env "CNN_ANGLE_INDICES_OVERRIDE" "$CNN_ANGLE_INDICES"
set_run_env "CNN_NUM_ANGLES_OVERRIDE" "$K"
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
set_run_env "DATA_SOURCE_OVERRIDE" "$TEST_DATA_SOURCE"
set_run_env "TEST_DATA_SOURCE_OVERRIDE" "$TEST_DATA_SOURCE"
set_run_env "NOISE_MODE_OVERRIDE" "multiplicative"
set_run_env "NOISE_LEVEL_OVERRIDE" "$NOISE_TEXT"
set_run_env "MODEL_ARCH_OVERRIDE" "tv_pc_cascade_unet"
set_run_env "REFINER_INPUT_MODE_OVERRIDE" "u2_stacked"
set_run_env "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE" "stacked_selected"
set_run_env "UNET_BACKBONE_OVERRIDE" "rad_unet"
set_run_env "UNET_BASE_CHANNELS_OVERRIDE" "64"
set_run_env "UNET_DEPTH_OVERRIDE" "4"
set_run_env "UNET_RESIDUAL_MAX_OVERRIDE" "0.25"
set_run_env "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE" "1"
set_run_env "PHYSICS_RESIDUAL_MODE_OVERRIDE" "stacked_selected_cg"
set_run_env "PHYSICS_RESIDUAL_DAMPING_OVERRIDE" "1e-2"
set_run_env "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE" "8"
set_run_env "PHYSICS_RESIDUAL_DETACH_OVERRIDE" "1"
set_run_env "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE" "1"
set_run_env "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE" "1"
set_run_env "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE" "0.13"
set_run_env "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE" "0.25"
set_run_env "PHYSICS_GATE_MODE_OVERRIDE" "spatial"
set_run_env "REFINER_STAGES_OVERRIDE" "2"
set_run_env "REFINER_SHARE_WEIGHTS_OVERRIDE" "1"
set_run_env "REFINER_STAGE_DC_ENABLED_OVERRIDE" "1"
set_run_env "REFINER_STAGE_DC_CG_ITERS_OVERRIDE" "4"
set_run_env "REFINER_STAGE_DC_DAMPING_OVERRIDE" "1e-2"
set_run_env "REFINER_STAGE_DC_DETACH_OVERRIDE" "1"
set_run_env "REFINER_STAGE_DC_NORMALIZE_OVERRIDE" "1"
set_run_env "OUTPUT_TAG_OVERRIDE" "$OUTPUT_TAG"
set_run_env "LOG_DIR_OVERRIDE" "$LOG_DIR"
set_run_env "RESULTS_DIR_OVERRIDE" "$RESOLVED_RESULT_DIR"
set_run_env "RESULT_DIR_OVERRIDE" "$RESOLVED_RESULT_DIR"
set_run_env "MODEL_LOAD_PATH_OVERRIDE" "$RESOLVED_MODEL_PATH"
set_run_env "MODEL_DIR_OVERRIDE" "$CHECKPOINT_DIR"
set_run_env "CHECKPOINT_DIR_OVERRIDE" "$CHECKPOINT_DIR"

echo "[test] project=${PROJECT_ROOT}"
echo "[test] python=${PYTHON_BIN}"
echo "[test] K=${K} detector_samples=${DETECTOR_SAMPLES} measurements=$((K * DETECTOR_SAMPLES))"
echo "[test] alpha_json=${ALPHA_JSON}"
echo "[test] model_choice=${MODEL_CHOICE} model=${RESOLVED_MODEL_PATH}"
echo "[test] num_samples=${NUM_SAMPLES} test_data_source=${TEST_DATA_SOURCE} noise=multiplicative delta=${NOISE_TEXT}"
echo "[test] offline_eval=${USE_OFFLINE_EVAL} offline_dataset='${RESOLVED_OFFLINE_EVAL}'"
echo "[test] output_tag=${OUTPUT_TAG} result_prefix=${RESOLVED_RESULT_PREFIX}"
echo "[test] result_dir=${RESOLVED_RESULT_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    [[ -f "$RESOLVED_MODEL_PATH" ]] && MODEL_EXISTS=yes || MODEL_EXISTS=no
    [[ -n "$RESOLVED_OFFLINE_EVAL" && -f "$RESOLVED_OFFLINE_EVAL" ]] && OFFLINE_EXISTS=yes || OFFLINE_EXISTS=no
    echo "[test] DryRun enabled; model_exists=${MODEL_EXISTS} offline_dataset_exists=${OFFLINE_EXISTS} and evaluation was not started."
    exit 0
fi

if [[ ! -f "$RESOLVED_MODEL_PATH" ]]; then
    echo "Model checkpoint not found: $RESOLVED_MODEL_PATH" >&2
    echo "Use --model-path, --output-tag, or --model-choice." >&2
    exit 1
fi

if [[ "$USE_OFFLINE_EVAL" -eq 1 ]]; then
    if [[ ! -f "$RESOLVED_OFFLINE_EVAL" ]]; then
        echo "Offline eval dataset not found: $RESOLVED_OFFLINE_EVAL" >&2
        echo "Generate offline data, pass --offline-eval-dataset, or pass --force-online." >&2
        exit 1
    fi
    exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/deep_learn/evaluate_best_model_offline_val.py" \
        --model-path "$RESOLVED_MODEL_PATH" \
        --offline-val "$RESOLVED_OFFLINE_EVAL" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$OFFLINE_BATCH_SIZE" \
        --result-dir "$RESOLVED_RESULT_DIR" \
        --result-prefix "$RESOLVED_RESULT_PREFIX"
fi

exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/deep_learn/test.py" \
    --model-path "$RESOLVED_MODEL_PATH" \
    --num-samples "$NUM_SAMPLES" \
    --result-dir "$RESOLVED_RESULT_DIR" \
    --result-prefix "$RESOLVED_RESULT_PREFIX"
