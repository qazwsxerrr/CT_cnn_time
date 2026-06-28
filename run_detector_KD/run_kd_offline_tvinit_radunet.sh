#!/usr/bin/env bash
set -euo pipefail

# Train RAD-UNet TV-init model for KD-threshold sparse-detector cases.
# Defaults: K=24,D=704. Other default pairs: K=8,D=2048 and K=16,D=1024.

K=24
DETECTOR_SAMPLES=0
NOISE_LEVEL="0.1"
OUTPUT_TAG=""
OFFLINE_TRAIN_DATASET=""
OFFLINE_VAL_DATASET=""
OFFLINE_SECONDARY_VAL_DATASET=""
DUAL_VAL_OBSERVATION=0
SECONDARY_VAL_SUBSAMPLE_SIZE=100
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_ONLINE_FALLBACK=0
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
  --offline-train-dataset PATH Explicit precomputed train .pt file
  --offline-val-dataset PATH   Explicit precomputed validation .pt file
  --dual-val-observation       Keep random_ellipses val500 as main validation and also observe shepp_logan test500 with a random 100-sample subset
  --offline-secondary-val-dataset PATH Explicit secondary validation .pt file. Default: test*_shepp_logan_tvinit_*.pt under offline_data
  --secondary-val-subsample-size N Secondary validation random subset size. Default: 100
  --allow-online-fallback      If offline .pt files are absent, allow online generation
  --project-root PATH          Project root containing models/ and data/. Default: parent of this script, or /root when detected
  --python-bin PATH            Python executable. Default: \$PYTHON_BIN or python
  --dry-run                    Prepare and print environment without starting training
  -h, --help                   Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k) K="$2"; shift 2 ;;
        --detector-samples) DETECTOR_SAMPLES="$2"; shift 2 ;;
        --noise-level) NOISE_LEVEL="$2"; shift 2 ;;
        --output-tag) OUTPUT_TAG="$2"; shift 2 ;;
        --offline-train-dataset) OFFLINE_TRAIN_DATASET="$2"; shift 2 ;;
        --offline-val-dataset) OFFLINE_VAL_DATASET="$2"; shift 2 ;;
        --dual-val-observation) DUAL_VAL_OBSERVATION=1; shift ;;
        --offline-secondary-val-dataset) OFFLINE_SECONDARY_VAL_DATASET="$2"; shift 2 ;;
        --secondary-val-subsample-size) SECONDARY_VAL_SUBSAMPLE_SIZE="$2"; shift 2 ;;
        --allow-online-fallback) ALLOW_ONLINE_FALLBACK=1; shift ;;
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
if ! [[ "$DETECTOR_SAMPLES" =~ ^[0-9]+$ ]] || [[ "$DETECTOR_SAMPLES" -le 0 ]]; then
    echo "DetectorSamples must be a positive integer." >&2
    exit 1
fi
if ! [[ "$SECONDARY_VAL_SUBSAMPLE_SIZE" =~ ^[0-9]+$ ]] || [[ "$SECONDARY_VAL_SUBSAMPLE_SIZE" -le 0 ]]; then
    echo "secondary-val-subsample-size must be a positive integer." >&2
    exit 1
fi

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
RESULT_DIR="${CASE_ROOT}/result"
CHECKPOINT_DIR="${CASE_ROOT}/checkpoints"
OFFLINE_DATA_DIR="${CASE_ROOT}/offline_data"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CHECKPOINT_DIR"

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

if [[ -z "$OFFLINE_TRAIN_DATASET" ]]; then
    OFFLINE_TRAIN_DATASET="$(resolve_default_offline_dataset "$OFFLINE_DATA_DIR" "train8000_random_ellipses_tvinit_${FILE_SUFFIX}.pt" "train*_random_ellipses_tvinit_${FILE_SUFFIX}.pt")"
elif [[ "$OFFLINE_TRAIN_DATASET" != /* ]]; then
    OFFLINE_TRAIN_DATASET="${PROJECT_ROOT}/${OFFLINE_TRAIN_DATASET}"
fi
if [[ -z "$OFFLINE_VAL_DATASET" ]]; then
    OFFLINE_VAL_DATASET="$(resolve_default_offline_dataset "$OFFLINE_DATA_DIR" "val500_random_ellipses_tvinit_${FILE_SUFFIX}.pt" "val*_random_ellipses_tvinit_${FILE_SUFFIX}.pt")"
elif [[ "$OFFLINE_VAL_DATASET" != /* ]]; then
    OFFLINE_VAL_DATASET="${PROJECT_ROOT}/${OFFLINE_VAL_DATASET}"
fi
if [[ "$DUAL_VAL_OBSERVATION" -eq 1 ]]; then
    if [[ -z "$OFFLINE_SECONDARY_VAL_DATASET" ]]; then
        OFFLINE_SECONDARY_VAL_DATASET="$(resolve_default_offline_dataset "$OFFLINE_DATA_DIR" "test500_shepp_logan_tvinit_${FILE_SUFFIX}.pt" "test*_shepp_logan_tvinit_${FILE_SUFFIX}.pt")"
    elif [[ "$OFFLINE_SECONDARY_VAL_DATASET" != /* ]]; then
        OFFLINE_SECONDARY_VAL_DATASET="${PROJECT_ROOT}/${OFFLINE_SECONDARY_VAL_DATASET}"
    fi
fi

if [[ "$ALLOW_ONLINE_FALLBACK" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    if [[ -z "$OFFLINE_TRAIN_DATASET" || ! -f "$OFFLINE_TRAIN_DATASET" ]]; then
        echo "Offline train dataset not found. Generate offline data first or pass --allow-online-fallback." >&2
        exit 1
    fi
    if [[ -z "$OFFLINE_VAL_DATASET" || ! -f "$OFFLINE_VAL_DATASET" ]]; then
        echo "Offline validation dataset not found. Generate offline data first or pass --allow-online-fallback." >&2
        exit 1
    fi
    if [[ "$DUAL_VAL_OBSERVATION" -eq 1 && ( -z "$OFFLINE_SECONDARY_VAL_DATASET" || ! -f "$OFFLINE_SECONDARY_VAL_DATASET" ) ]]; then
        echo "Secondary offline validation dataset not found. Generate offline data first, pass --offline-secondary-val-dataset, or disable --dual-val-observation." >&2
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
if [[ -z "$OFFLINE_TRAIN_DATASET" && -z "$OFFLINE_VAL_DATASET" ]]; then
    ONLINE_DATA=1
else
    ONLINE_DATA=0
fi

set_run_env() { export "$1=$2"; }

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
set_run_env "TRAIN_DATA_SOURCE_OVERRIDE" "random_ellipses"
set_run_env "VAL_DATA_SOURCE_OVERRIDE" "random_ellipses"
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
set_run_env "BASE_LR_OVERRIDE" "3e-4"
set_run_env "LR_SCHEDULE_OVERRIDE" "constant_cosine"
set_run_env "LR_CONSTANT_STEPS_OVERRIDE" "4500"
set_run_env "LR_MIN_FACTOR_OVERRIDE" "0.3"
set_run_env "LR_WARMUP_STEPS_OVERRIDE" "100"
set_run_env "SCALAR_LR_RATIO_OVERRIDE" "1.0"
set_run_env "INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE" "1"
set_run_env "INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE" "0.4"
set_run_env "INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE" "1.0"
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
set_run_env "RESULTS_DIR_OVERRIDE" "$RESULT_DIR"
set_run_env "MODEL_DIR_OVERRIDE" "$CHECKPOINT_DIR"
set_run_env "CHECKPOINT_DIR_OVERRIDE" "$CHECKPOINT_DIR"
set_run_env "MODEL_PATH_OVERRIDE" "${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_model.pth"
set_run_env "BEST_MODEL_PATH_OVERRIDE" "${CHECKPOINT_DIR}/theoretical_ct_${OUTPUT_TAG}_best_model.pth"

echo "[run] project=${PROJECT_ROOT}"
echo "[run] python=${PYTHON_BIN}"
echo "[run] K=${K} detector_samples=${DETECTOR_SAMPLES} measurements=$((K * DETECTOR_SAMPLES))"
echo "[run] alpha_json=${ALPHA_JSON}"
echo "[run] online_data=${ONLINE_DATA} offline_train='${OFFLINE_TRAIN_DATASET}' offline_val='${OFFLINE_VAL_DATASET}'"
echo "[run] dual_val_observation=${DUAL_VAL_OBSERVATION} secondary_val='${OFFLINE_SECONDARY_VAL_DATASET}' secondary_samples=${SECONDARY_VAL_SUBSAMPLE_SIZE}"
echo "[run] init=l2_tv_admm morozov=constrained radius=rms admm=80 cg=30 tol=1e-4"
echo "[run] model=tv_pc_cascade_unet refiner=u2_stacked backbone=rad_unet stages=2 gate=spatial stage_dc=1"
echo "[run] lr=3e-4 schedule=constant_cosine constant_steps=4500 min_factor=0.3 warmup=100 scalar_lr_ratio=1.0"
echo "[run] output_tag=${OUTPUT_TAG}"
echo "[run] log_dir=${LOG_DIR}"
echo "[run] result_dir=${RESULT_DIR}"
echo "[run] checkpoint_dir=${CHECKPOINT_DIR}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[run] DryRun enabled; environment prepared but training was not started."
    exit 0
fi

exec "$PYTHON_BIN" "${PROJECT_ROOT}/models/deep_learn/train.py"
