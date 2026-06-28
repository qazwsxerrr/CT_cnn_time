#!/usr/bin/env bash
set -euo pipefail

# Common Linux/bash runner for K=24,D=704 improvement experiments.
# Training uses random_ellipses val500 as the primary validation set and
# observes a random 100-sample Shepp-Logan subset as secondary validation.

EXPERIMENT=""
MODE="train"
NOISE_LEVEL="0.1"
OUTPUT_TAG=""
OFFLINE_DATA_DIR=""
OFFLINE_TRAIN_DATASET=""
OFFLINE_VAL_DATASET=""
OFFLINE_EVAL_DATASET=""
MODEL_CHOICE="best"
MODEL_PATH=""
NUM_SAMPLES=500
OFFLINE_BATCH_SIZE=20
RESULT_PREFIX=""
RESULT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_ONLINE_FALLBACK=0
DRY_RUN=0
SECONDARY_VAL_SUBSAMPLE_SIZE=100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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
Usage: bash $(basename "$0") --experiment NAME [options]

Experiments:
  run1_1stage_nodc_rmax005
  run2_1stage_weakphys_rmax005
  run3_1stage_tinyphys_rmax003
  run4_1stage_weak_stage_dc
  run5_2stage_tied_small_nodc

Options:
  --experiment NAME              Experiment name. Required for common runner.
  --mode train|test              Run training or offline evaluation. Default: train
  --noise-level X                Multiplicative noise level. Default: 0.1
  --output-tag TAG               Output tag. Default: a24_d704_n{Noise}_admm80_{Experiment}
  --offline-data-dir PATH        Directory containing KD offline .pt files. Default: run_detector_KD/24_angle_det704/offline_data
  --offline-train-dataset PATH   Explicit train .pt file
  --offline-val-dataset PATH     Explicit primary validation .pt file; default is val500_random_ellipses
  --offline-eval-dataset PATH    Explicit test/eval .pt file; default is test500_shepp_logan
  --model-choice best|final      Model file to evaluate in --mode test. Default: best
  --model-path PATH              Explicit checkpoint path for --mode test
  --num-samples N                Evaluation sample count for --mode test. Default: 500
  --offline-batch-size N         Evaluation batch size for --mode test. Default: 20
  --result-prefix PREFIX         Evaluation result prefix. Default: output tag
  --result-dir PATH              Evaluation result directory. Default: experiment result directory
  --secondary-val-subsample-size N Shepp-Logan secondary validation random subset size. Default: 100
  --allow-online-fallback        Allow online training if primary offline datasets are missing
  --project-root PATH            Project root containing models/ and data/. Default: inferred from script location
  --python-bin PATH              Python executable. Default: \$PYTHON_BIN or python
  --dry-run                      Prepare and print environment without starting train/test
  -h, --help                     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --noise-level) NOISE_LEVEL="$2"; shift 2 ;;
        --output-tag) OUTPUT_TAG="$2"; shift 2 ;;
        --offline-data-dir) OFFLINE_DATA_DIR="$2"; shift 2 ;;
        --offline-train-dataset) OFFLINE_TRAIN_DATASET="$2"; shift 2 ;;
        --offline-val-dataset) OFFLINE_VAL_DATASET="$2"; shift 2 ;;
        --offline-eval-dataset) OFFLINE_EVAL_DATASET="$2"; shift 2 ;;
        --model-choice) MODEL_CHOICE="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --offline-batch-size) OFFLINE_BATCH_SIZE="$2"; shift 2 ;;
        --result-prefix) RESULT_PREFIX="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --secondary-val-subsample-size) SECONDARY_VAL_SUBSAMPLE_SIZE="$2"; shift 2 ;;
        --allow-online-fallback) ALLOW_ONLINE_FALLBACK=1; shift ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --python-bin) PYTHON_BIN="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Missing required --experiment." >&2
    usage >&2
    exit 2
fi
case "$MODE" in
    train|test) ;;
    *) echo "Unsupported mode: $MODE. Expected train or test." >&2; exit 2 ;;
esac
case "$MODEL_CHOICE" in
    best|final) ;;
    *) echo "Unsupported model choice: $MODEL_CHOICE. Expected best or final." >&2; exit 2 ;;
esac
for item in "NUM_SAMPLES:$NUM_SAMPLES" "OFFLINE_BATCH_SIZE:$OFFLINE_BATCH_SIZE" "SECONDARY_VAL_SUBSAMPLE_SIZE:$SECONDARY_VAL_SUBSAMPLE_SIZE"; do
    name="${item%%:*}"
    value="${item#*:}"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -le 0 ]]; then
        echo "$name must be a positive integer, got $value" >&2
        exit 1
    fi
done

case "$EXPERIMENT" in
    run1_1stage_nodc_rmax005)
        DESCRIPTION="one-stage residual, no explicit update, no stage DC, residual max 0.05"
        REFINER_STAGES="1"; SHARE_WEIGHTS="1"; STAGE_DC_ENABLED="0"; STAGE_DC_CG_ITERS="4"; STAGE_DC_DAMPING="1e-2"
        EXPLICIT_ENABLED="0"; EXPLICIT_ALPHA_INIT="0.0"; EXPLICIT_MAX="0.0"; PHYSICS_GATE_MODE="spatial"
        RESIDUAL_MAX="0.05"; INTERMEDIATE_ENABLED="0"; INTERMEDIATE_START="0.0"; INTERMEDIATE_END="0.0"
        ;;
    run2_1stage_weakphys_rmax005)
        DESCRIPTION="one-stage residual, weak explicit physics update, no stage DC, residual max 0.05"
        REFINER_STAGES="1"; SHARE_WEIGHTS="1"; STAGE_DC_ENABLED="0"; STAGE_DC_CG_ITERS="4"; STAGE_DC_DAMPING="1e-2"
        EXPLICIT_ENABLED="1"; EXPLICIT_ALPHA_INIT="0.02"; EXPLICIT_MAX="0.05"; PHYSICS_GATE_MODE="spatial"
        RESIDUAL_MAX="0.05"; INTERMEDIATE_ENABLED="0"; INTERMEDIATE_START="0.0"; INTERMEDIATE_END="0.0"
        ;;
    run3_1stage_tinyphys_rmax003)
        DESCRIPTION="one-stage residual, tiny explicit physics update, no stage DC, residual max 0.03"
        REFINER_STAGES="1"; SHARE_WEIGHTS="1"; STAGE_DC_ENABLED="0"; STAGE_DC_CG_ITERS="4"; STAGE_DC_DAMPING="1e-2"
        EXPLICIT_ENABLED="1"; EXPLICIT_ALPHA_INIT="0.01"; EXPLICIT_MAX="0.03"; PHYSICS_GATE_MODE="spatial"
        RESIDUAL_MAX="0.03"; INTERMEDIATE_ENABLED="0"; INTERMEDIATE_START="0.0"; INTERMEDIATE_END="0.0"
        ;;
    run4_1stage_weak_stage_dc)
        DESCRIPTION="one-stage residual, weak stage DC, no explicit update, residual max 0.05"
        REFINER_STAGES="1"; SHARE_WEIGHTS="1"; STAGE_DC_ENABLED="1"; STAGE_DC_CG_ITERS="2"; STAGE_DC_DAMPING="1e-1"
        EXPLICIT_ENABLED="0"; EXPLICIT_ALPHA_INIT="0.0"; EXPLICIT_MAX="0.0"; PHYSICS_GATE_MODE="spatial"
        RESIDUAL_MAX="0.05"; INTERMEDIATE_ENABLED="0"; INTERMEDIATE_START="0.0"; INTERMEDIATE_END="0.0"
        ;;
    run5_2stage_tied_small_nodc)
        DESCRIPTION="two-stage shared residual, tiny explicit physics update, no stage DC, residual max 0.03"
        REFINER_STAGES="2"; SHARE_WEIGHTS="1"; STAGE_DC_ENABLED="0"; STAGE_DC_CG_ITERS="4"; STAGE_DC_DAMPING="1e-2"
        EXPLICIT_ENABLED="1"; EXPLICIT_ALPHA_INIT="0.01"; EXPLICIT_MAX="0.03"; PHYSICS_GATE_MODE="spatial"
        RESIDUAL_MAX="0.03"; INTERMEDIATE_ENABLED="1"; INTERMEDIATE_START="0.2"; INTERMEDIATE_END="0.5"
        ;;
    *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

BASE_LR="1e-4"
LR_SCHEDULE="constant_cosine"
LR_CONSTANT_STEPS="4500"
LR_MIN_FACTOR="0.2"
LR_WARMUP="300"
SCALAR_LR_RATIO="0.1"
BASE_CHANNELS="64"
DEPTH="4"
K=24
DETECTOR_SAMPLES=704

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
cd "$PROJECT_ROOT"

if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

ALPHA_JSON="${PROJECT_ROOT}/data/alpha24_tv/alpha_selected24_dopt_hard_gap_6_9.json"
if [[ ! -f "$ALPHA_JSON" ]]; then
    echo "Selected-angle JSON not found: $ALPHA_JSON" >&2
    exit 1
fi

resolve_input_path() {
    local input="$1"
    if [[ -z "$input" ]]; then
        printf '\n'
    elif [[ "$input" == /* ]]; then
        printf '%s\n' "$input"
    else
        printf '%s\n' "${PROJECT_ROOT}/${input}"
    fi
}

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

CNN_ANGLE_INDICES="$(seq -s, 0 $((K - 1)))"
NOISE_TEXT="$NOISE_LEVEL"
OFFLINE_NOISE_TAG="${NOISE_TEXT//./p}"
OFFLINE_NOISE_TAG="${OFFLINE_NOISE_TAG//-/m}"
NOISE_TAG="n${OFFLINE_NOISE_TAG}"
if [[ -z "$OUTPUT_TAG" ]]; then
    OUTPUT_TAG="a24_d704_${NOISE_TAG}_admm80_${EXPERIMENT}"
fi

EXPERIMENT_ROOT="${SCRIPT_DIR}/${EXPERIMENT}"
LOG_DIR="${EXPERIMENT_ROOT}/log"
DEFAULT_RESULT_DIR="${EXPERIMENT_ROOT}/result"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/checkpoints"
mkdir -p "$LOG_DIR" "$DEFAULT_RESULT_DIR" "$CHECKPOINT_DIR"

KD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESOLVED_OFFLINE_DATA_DIR="$(resolve_input_path "$OFFLINE_DATA_DIR")"
if [[ -z "$RESOLVED_OFFLINE_DATA_DIR" ]]; then
    RESOLVED_OFFLINE_DATA_DIR="${KD_ROOT}/24_angle_det704/offline_data"
fi

FILE_SUFFIX="alpha24_noise${OFFLINE_NOISE_TAG}_det704_edgewsubset"
OFFLINE_TRAIN_DATASET="$(resolve_input_path "$OFFLINE_TRAIN_DATASET")"
if [[ -z "$OFFLINE_TRAIN_DATASET" ]]; then
    OFFLINE_TRAIN_DATASET="$(resolve_default_offline_dataset "$RESOLVED_OFFLINE_DATA_DIR" "train8000_random_ellipses_tvinit_${FILE_SUFFIX}.pt" "train*_random_ellipses_tvinit_${FILE_SUFFIX}.pt")"
fi
OFFLINE_VAL_DATASET="$(resolve_input_path "$OFFLINE_VAL_DATASET")"
if [[ -z "$OFFLINE_VAL_DATASET" ]]; then
    OFFLINE_VAL_DATASET="$(resolve_default_offline_dataset "$RESOLVED_OFFLINE_DATA_DIR" "val500_random_ellipses_tvinit_${FILE_SUFFIX}.pt" "val*_random_ellipses_tvinit_${FILE_SUFFIX}.pt")"
fi
OFFLINE_SECONDARY_VAL_DATASET="$(resolve_default_offline_dataset "$RESOLVED_OFFLINE_DATA_DIR" "test500_shepp_logan_tvinit_${FILE_SUFFIX}.pt" "test*_shepp_logan_tvinit_${FILE_SUFFIX}.pt")"
OFFLINE_EVAL_DATASET="$(resolve_input_path "$OFFLINE_EVAL_DATASET")"
if [[ -z "$OFFLINE_EVAL_DATASET" ]]; then
    OFFLINE_EVAL_DATASET="$OFFLINE_SECONDARY_VAL_DATASET"
fi

if [[ "$MODE" == "train" && "$DRY_RUN" -eq 0 && "$ALLOW_ONLINE_FALLBACK" -eq 0 ]]; then
    if [[ -z "$OFFLINE_TRAIN_DATASET" || ! -f "$OFFLINE_TRAIN_DATASET" ]]; then
        echo "Offline train dataset not found. Pass --offline-data-dir/--offline-train-dataset or --allow-online-fallback." >&2
        exit 1
    fi
    if [[ -z "$OFFLINE_VAL_DATASET" || ! -f "$OFFLINE_VAL_DATASET" ]]; then
        echo "Offline validation dataset not found. Pass --offline-data-dir/--offline-val-dataset or --allow-online-fallback." >&2
        exit 1
    fi
    if [[ -z "$OFFLINE_SECONDARY_VAL_DATASET" || ! -f "$OFFLINE_SECONDARY_VAL_DATASET" ]]; then
        echo "Secondary Shepp-Logan validation dataset not found. Pass --offline-data-dir or generate offline data first." >&2
        exit 1
    fi
fi
if [[ "$ALLOW_ONLINE_FALLBACK" -eq 1 && ( -z "$OFFLINE_TRAIN_DATASET" || ! -f "$OFFLINE_TRAIN_DATASET" ) ]]; then
    OFFLINE_TRAIN_DATASET=""
fi
if [[ "$ALLOW_ONLINE_FALLBACK" -eq 1 && ( -z "$OFFLINE_VAL_DATASET" || ! -f "$OFFLINE_VAL_DATASET" ) ]]; then
    OFFLINE_VAL_DATASET=""
fi
if [[ "$ALLOW_ONLINE_FALLBACK" -eq 1 && ( -z "$OFFLINE_SECONDARY_VAL_DATASET" || ! -f "$OFFLINE_SECONDARY_VAL_DATASET" ) ]]; then
    OFFLINE_SECONDARY_VAL_DATASET=""
fi
ONLINE_DATA=0
if [[ -z "$OFFLINE_TRAIN_DATASET" && -z "$OFFLINE_VAL_DATASET" ]]; then
    ONLINE_DATA=1
fi

RESOLVED_RESULT_DIR="$(resolve_input_path "$RESULT_DIR")"
if [[ -z "$RESOLVED_RESULT_DIR" ]]; then
    RESOLVED_RESULT_DIR="$DEFAULT_RESULT_DIR"
fi
RESOLVED_RESULT_PREFIX="$RESULT_PREFIX"
if [[ -z "$RESOLVED_RESULT_PREFIX" ]]; then
    RESOLVED_RESULT_PREFIX="$OUTPUT_TAG"
fi

if [[ "$MODEL_CHOICE" == "best" ]]; then
    MODEL_FILE_KIND="best_model"
else
    MODEL_FILE_KIND="model"
fi
DEFAULT_MODEL_PATH="${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_${MODEL_FILE_KIND}.pth"
COMPACT_MODEL_PATH="${DEFAULT_MODEL_PATH%.pth}_compact.pth"
RESOLVED_MODEL_PATH="$(resolve_input_path "$MODEL_PATH")"
if [[ -z "$RESOLVED_MODEL_PATH" ]]; then
    if [[ -f "$COMPACT_MODEL_PATH" ]]; then
        RESOLVED_MODEL_PATH="$COMPACT_MODEL_PATH"
    else
        RESOLVED_MODEL_PATH="$DEFAULT_MODEL_PATH"
    fi
fi

set_run_env() { export "$1=$2"; }

set_run_env "PROJECT_ROOT" "$PROJECT_ROOT"
set_run_env "PYTHONUTF8" "1"
set_run_env "PYTHON_BIN" "$PYTHON_BIN"
set_run_env "EXPERIMENT_PROFILE_OVERRIDE" "alpha_condition"
set_run_env "ALPHA_CONDITION_TOP_K_OVERRIDE" "24"
set_run_env "ALPHA_CONDITION_JSON_OVERRIDE" "$ALPHA_JSON"
set_run_env "ALPHA_GRAM_CACHE_DIR_OVERRIDE" "${PROJECT_ROOT}/data/alpha_gram_cache"
set_run_env "NUM_ANGLES_TOTAL_OVERRIDE" "24"
set_run_env "CNN_ANGLE_INDICES_OVERRIDE" "$CNN_ANGLE_INDICES"
set_run_env "CNN_NUM_ANGLES_OVERRIDE" "24"
set_run_env "MULTI_ANGLE_SOLVER_MODE_OVERRIDE" "stacked_tikhonov"
set_run_env "THEORETICAL_FORMULA_MODE_OVERRIDE" "alpha_continuous"
set_run_env "SAMPLING_MODE_OVERRIDE" "shifted_lattice_edge_weighted_subset"
set_run_env "NUM_DETECTOR_SAMPLES_OVERRIDE" "704"
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
set_run_env "MODEL_ARCH_OVERRIDE" "tv_pc_cascade_unet"
set_run_env "REFINER_INPUT_MODE_OVERRIDE" "u2_stacked"
set_run_env "DATA_FIDELITY_CHANNEL_MODE_OVERRIDE" "stacked_selected"
set_run_env "UNET_BACKBONE_OVERRIDE" "rad_unet"
set_run_env "UNET_BASE_CHANNELS_OVERRIDE" "$BASE_CHANNELS"
set_run_env "UNET_DEPTH_OVERRIDE" "$DEPTH"
set_run_env "UNET_RESIDUAL_MAX_OVERRIDE" "$RESIDUAL_MAX"
set_run_env "PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE" "1"
set_run_env "PHYSICS_RESIDUAL_MODE_OVERRIDE" "stacked_selected_cg"
set_run_env "PHYSICS_RESIDUAL_DAMPING_OVERRIDE" "1e-2"
set_run_env "PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE" "8"
set_run_env "PHYSICS_RESIDUAL_DETACH_OVERRIDE" "1"
set_run_env "PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE" "1"
set_run_env "PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE" "$EXPLICIT_ENABLED"
set_run_env "PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE" "$EXPLICIT_ALPHA_INIT"
set_run_env "PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE" "$EXPLICIT_MAX"
set_run_env "PHYSICS_GATE_MODE_OVERRIDE" "$PHYSICS_GATE_MODE"
set_run_env "REFINER_STAGES_OVERRIDE" "$REFINER_STAGES"
set_run_env "REFINER_SHARE_WEIGHTS_OVERRIDE" "$SHARE_WEIGHTS"
set_run_env "REFINER_STAGE_DC_ENABLED_OVERRIDE" "$STAGE_DC_ENABLED"
set_run_env "REFINER_STAGE_DC_CG_ITERS_OVERRIDE" "$STAGE_DC_CG_ITERS"
set_run_env "REFINER_STAGE_DC_DAMPING_OVERRIDE" "$STAGE_DC_DAMPING"
set_run_env "REFINER_STAGE_DC_DETACH_OVERRIDE" "1"
set_run_env "REFINER_STAGE_DC_NORMALIZE_OVERRIDE" "1"
set_run_env "BASE_LR_OVERRIDE" "$BASE_LR"
set_run_env "LR_SCHEDULE_OVERRIDE" "$LR_SCHEDULE"
set_run_env "LR_CONSTANT_STEPS_OVERRIDE" "$LR_CONSTANT_STEPS"
set_run_env "LR_MIN_FACTOR_OVERRIDE" "$LR_MIN_FACTOR"
set_run_env "LR_WARMUP_STEPS_OVERRIDE" "$LR_WARMUP"
set_run_env "SCALAR_LR_RATIO_OVERRIDE" "$SCALAR_LR_RATIO"
set_run_env "INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE" "$INTERMEDIATE_ENABLED"
set_run_env "INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE" "$INTERMEDIATE_START"
set_run_env "INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE" "$INTERMEDIATE_END"
set_run_env "DETACH_PHYSICAL_GRADS_OVERRIDE" "0"
set_run_env "N_DATA_OVERRIDE" "4"
set_run_env "N_TRAIN_OVERRIDE" "8000"
set_run_env "VALIDATION_INTERVAL_OVERRIDE" "50"
set_run_env "VAL_RANDOM_SUBSAMPLE_OVERRIDE" "1"
set_run_env "VAL_SUBSAMPLE_SIZE_OVERRIDE" "500"
set_run_env "VAL_BATCH_SIZE_OVERRIDE" "100"
set_run_env "COMPACT_CHECKPOINTS_OVERRIDE" "1"
set_run_env "OUTPUT_TAG_OVERRIDE" "$OUTPUT_TAG"
set_run_env "LOG_DIR_OVERRIDE" "$LOG_DIR"
set_run_env "RESULTS_DIR_OVERRIDE" "$RESOLVED_RESULT_DIR"
set_run_env "RESULT_DIR_OVERRIDE" "$RESOLVED_RESULT_DIR"
set_run_env "MODEL_DIR_OVERRIDE" "$CHECKPOINT_DIR"
set_run_env "CHECKPOINT_DIR_OVERRIDE" "$CHECKPOINT_DIR"
set_run_env "MODEL_PATH_OVERRIDE" "${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_model.pth"
set_run_env "BEST_MODEL_PATH_OVERRIDE" "${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_best_model.pth"
set_run_env "MODEL_LOAD_PATH_OVERRIDE" "$RESOLVED_MODEL_PATH"

echo "[$MODE] experiment=${EXPERIMENT}"
echo "[$MODE] description=${DESCRIPTION}"
echo "[$MODE] project=${PROJECT_ROOT}"
echo "[$MODE] python=${PYTHON_BIN}"
echo "[$MODE] K=24 detector_samples=704 measurements=$((K * DETECTOR_SAMPLES))"
echo "[$MODE] alpha_json=${ALPHA_JSON}"
echo "[$MODE] offline_data_dir=${RESOLVED_OFFLINE_DATA_DIR}"
echo "[$MODE] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
echo "[$MODE] model=tv_pc_cascade_unet input=u2_stacked backbone=rad_unet stages=${REFINER_STAGES} residual_max=${RESIDUAL_MAX}"
echo "[$MODE] explicit_update=${EXPLICIT_ENABLED} alpha_init=${EXPLICIT_ALPHA_INIT} max=${EXPLICIT_MAX} gate=${PHYSICS_GATE_MODE}"
echo "[$MODE] stage_dc=${STAGE_DC_ENABLED} dc_cg=${STAGE_DC_CG_ITERS} dc_damping=${STAGE_DC_DAMPING}"
echo "[$MODE] lr=${BASE_LR} schedule=${LR_SCHEDULE} constant_steps=${LR_CONSTANT_STEPS} min_factor=${LR_MIN_FACTOR} warmup=${LR_WARMUP}"
echo "[$MODE] output_tag=${OUTPUT_TAG}"
echo "[$MODE] log_dir=${LOG_DIR}"
echo "[$MODE] result_dir=${RESOLVED_RESULT_DIR}"
echo "[$MODE] checkpoint_dir=${CHECKPOINT_DIR}"

if [[ "$MODE" == "train" ]]; then
    set_run_env "OFFLINE_TRAIN_DATASET_OVERRIDE" "$OFFLINE_TRAIN_DATASET"
    set_run_env "OFFLINE_VAL_DATASET_OVERRIDE" "$OFFLINE_VAL_DATASET"
    set_run_env "OFFLINE_SECONDARY_VAL_DATASET_OVERRIDE" "$OFFLINE_SECONDARY_VAL_DATASET"
    set_run_env "SECONDARY_VAL_DATA_SOURCE_OVERRIDE" "shepp_logan"
    set_run_env "SECONDARY_VAL_LABEL_OVERRIDE" "shepp_logan_random100"
    set_run_env "SECONDARY_VAL_SUBSAMPLE_SIZE_OVERRIDE" "$SECONDARY_VAL_SUBSAMPLE_SIZE"
    set_run_env "SECONDARY_VAL_BATCH_SIZE_OVERRIDE" "$SECONDARY_VAL_SUBSAMPLE_SIZE"
    set_run_env "SECONDARY_VAL_RANDOM_SUBSAMPLE_OVERRIDE" "1"
    set_run_env "SECONDARY_VAL_REPRODUCIBLE_OVERRIDE" "1"
    set_run_env "SECONDARY_VAL_SEED_OVERRIDE" "42"
    set_run_env "TRAIN_DATA_SOURCE_OVERRIDE" "random_ellipses"
    set_run_env "VAL_DATA_SOURCE_OVERRIDE" "random_ellipses"
    echo "[train] online_data=${ONLINE_DATA} offline_train='${OFFLINE_TRAIN_DATASET}' offline_val='${OFFLINE_VAL_DATASET}'"
    echo "[train] secondary_val='${OFFLINE_SECONDARY_VAL_DATASET}' secondary_samples=${SECONDARY_VAL_SUBSAMPLE_SIZE}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[train] DryRun enabled; environment prepared but training was not started."
        exit 0
    fi
    exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/deep_learn/train.py"
fi

echo "[test] model_choice=${MODEL_CHOICE} model=${RESOLVED_MODEL_PATH}"
echo "[test] offline_eval='${OFFLINE_EVAL_DATASET}' num_samples=${NUM_SAMPLES} batch_size=${OFFLINE_BATCH_SIZE} result_prefix=${RESOLVED_RESULT_PREFIX}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    [[ -f "$RESOLVED_MODEL_PATH" ]] && MODEL_EXISTS=yes || MODEL_EXISTS=no
    [[ -n "$OFFLINE_EVAL_DATASET" && -f "$OFFLINE_EVAL_DATASET" ]] && EVAL_EXISTS=yes || EVAL_EXISTS=no
    echo "[test] DryRun enabled; model_exists=${MODEL_EXISTS} offline_eval_exists=${EVAL_EXISTS} and evaluation was not started."
    exit 0
fi
if [[ ! -f "$RESOLVED_MODEL_PATH" ]]; then
    echo "Model checkpoint not found. Use --model-path, --output-tag, or --model-choice. Missing: $RESOLVED_MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$OFFLINE_EVAL_DATASET" ]]; then
    echo "Offline eval dataset not found. Use --offline-data-dir or --offline-eval-dataset. Missing: $OFFLINE_EVAL_DATASET" >&2
    exit 1
fi
exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/deep_learn/evaluate_best_model_offline_val.py" \
    --model-path "$RESOLVED_MODEL_PATH" \
    --offline-val "$OFFLINE_EVAL_DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$OFFLINE_BATCH_SIZE" \
    --result-dir "$RESOLVED_RESULT_DIR" \
    --result-prefix "$RESOLVED_RESULT_PREFIX"
