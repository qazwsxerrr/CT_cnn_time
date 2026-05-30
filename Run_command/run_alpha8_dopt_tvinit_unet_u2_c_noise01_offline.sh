#!/usr/bin/env bash
set -euo pipefail

# Offline training command for TV-PC Residual U-Net variant C with alpha-stack U2 input:
#   x_pred = x_tv + alpha * physics_corr0
#            + U-Net([x_tv, data_grad_alpha_1..8, physics_corr0, reg_grad0])
#
# The offline tensor files are expected to contain:
#   coeff_true, g_observed, coeff_initial

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${ALPHA_CONDITION_JSON_OVERRIDE:-}" ]]; then
  ANGLE_JSON="${ALPHA_CONDITION_JSON_OVERRIDE}"
elif [[ -f "${PROJECT_ROOT}/data/alpha8_tv/alpha_selected8_dopt_soft_g25.json" ]]; then
  ANGLE_JSON="${PROJECT_ROOT}/data/alpha8_tv/alpha_selected8_dopt_soft_g25.json"
elif [[ -f "${PROJECT_ROOT}/results/shepp_logan_condition_vs_dopt_tv_noise01_8/alpha_selected8_dopt_soft_g25.json" ]]; then
  ANGLE_JSON="${PROJECT_ROOT}/results/shepp_logan_condition_vs_dopt_tv_noise01_8/alpha_selected8_dopt_soft_g25.json"
else
  ANGLE_JSON="${PROJECT_ROOT}/data/alpha8_tv/alpha_selected8_dopt_soft_g25.json"
fi

if [[ ! -f "${ANGLE_JSON}" ]]; then
  echo "[error] alpha JSON not found: ${ANGLE_JSON}" >&2
  echo "        Set ALPHA_CONDITION_JSON_OVERRIDE to alpha_selected8_dopt_soft_g25.json" >&2
  exit 1
fi

export OFFLINE_TRAIN_DATASET_OVERRIDE="${OFFLINE_TRAIN_DATASET_OVERRIDE:-${PROJECT_ROOT}/data/data_genoration/train8000_tvinit_alpha8_noise01.pt}"
export OFFLINE_VAL_DATASET_OVERRIDE="${OFFLINE_VAL_DATASET_OVERRIDE:-${PROJECT_ROOT}/data/data_genoration/val500_tvinit_alpha8_noise01.pt}"

if [[ ! -f "${OFFLINE_TRAIN_DATASET_OVERRIDE}" ]]; then
  echo "[error] offline train dataset not found: ${OFFLINE_TRAIN_DATASET_OVERRIDE}" >&2
  echo "        Set OFFLINE_TRAIN_DATASET_OVERRIDE or generate the .pt file first." >&2
  exit 1
fi
if [[ ! -f "${OFFLINE_VAL_DATASET_OVERRIDE}" ]]; then
  echo "[error] offline val dataset not found: ${OFFLINE_VAL_DATASET_OVERRIDE}" >&2
  echo "        Set OFFLINE_VAL_DATASET_OVERRIDE or generate the .pt file first." >&2
  exit 1
fi

export EXPERIMENT_PROFILE_OVERRIDE="alpha_condition"
export ALPHA_CONDITION_TOP_K_OVERRIDE="8"
export ALPHA_CONDITION_JSON_OVERRIDE="${ANGLE_JSON}"
export ALPHA_GRAM_CACHE_DIR_OVERRIDE="${ALPHA_GRAM_CACHE_DIR_OVERRIDE:-${PROJECT_ROOT}/data/alpha_gram_cache}"

export MULTI_ANGLE_SOLVER_MODE_OVERRIDE="stacked_tikhonov"
export THEORETICAL_FORMULA_MODE_OVERRIDE="alpha_continuous"

export CNN_ANGLE_INDICES_OVERRIDE="0,1,2,3,4,5,6,7"
export CNN_NUM_ANGLES_OVERRIDE="8"

export INIT_METHOD_OVERRIDE="l2_tv_admm"
export LAMBDA_SELECT_MODE_OVERRIDE="morozov"
export MOROZOV_FORM_OVERRIDE="constrained"
export MOROZOV_NOISE_RADIUS_MODE_OVERRIDE="rms"
export MOROZOV_TAU_OVERRIDE="1.0"

export L1_INIT_ADMM_ITERS_OVERRIDE="40"
export L1_INIT_ADMM_CG_ITERS_OVERRIDE="15"
export L1_INIT_ADMM_CG_TOL_OVERRIDE="1e-4"
export L1_INIT_ADMM_RHO_DATA_OVERRIDE="1.0"
export L1_INIT_ADMM_RHO_REG_OVERRIDE="1.0"

export REGULARIZER_TYPE_OVERRIDE="dirichlet"

export NOISE_MODE_OVERRIDE="multiplicative"
export NOISE_LEVEL_OVERRIDE="0.1"

export MODEL_ARCH_OVERRIDE="tv_pc_unet"
export REFINER_INPUT_MODE_OVERRIDE="u2_alpha_stack"
export UNET_BASE_CHANNELS_OVERRIDE="${UNET_BASE_CHANNELS_OVERRIDE:-64}"
export UNET_DEPTH_OVERRIDE="${UNET_DEPTH_OVERRIDE:-4}"
# Direct residual U-Net updates are larger than the old LGD step-scaled CNN updates,
# so use a lower LR and cap the learned residual for stable refinement.
export BASE_LR_OVERRIDE="${BASE_LR_OVERRIDE:-3e-4}"
export UNET_RESIDUAL_MAX_OVERRIDE="${UNET_RESIDUAL_MAX_OVERRIDE:-0.25}"

# u2_alpha_stack keeps one image-domain data-gradient channel per selected alpha.
export DATA_FIDELITY_CHANNEL_MODE_OVERRIDE="per_angle"
export PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE="1"
export PHYSICS_RESIDUAL_MODE_OVERRIDE="stacked_selected_cg"
export PHYSICS_RESIDUAL_DAMPING_OVERRIDE="1e-2"
export PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE="8"
export PHYSICS_RESIDUAL_DETACH_OVERRIDE="1"
export PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE="1"

export PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE="1"
export PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE="${PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE:-0.13}"
export PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE="${PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE:-0.25}"

export INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE="0"
export DETACH_PHYSICAL_GRADS_OVERRIDE="${DETACH_PHYSICAL_GRADS_OVERRIDE:-0}"

export N_DATA_OVERRIDE="${N_DATA_OVERRIDE:-4}"
export VALIDATION_INTERVAL_OVERRIDE="${VALIDATION_INTERVAL_OVERRIDE:-50}"
export VAL_RANDOM_SUBSAMPLE_OVERRIDE="${VAL_RANDOM_SUBSAMPLE_OVERRIDE:-1}"
export VAL_SUBSAMPLE_SIZE_OVERRIDE="${VAL_SUBSAMPLE_SIZE_OVERRIDE:-50}"
export VAL_BATCH_SIZE_OVERRIDE="${VAL_BATCH_SIZE_OVERRIDE:-${VAL_SUBSAMPLE_SIZE_OVERRIDE}}"
export N_TRAIN_OVERRIDE="${N_TRAIN_OVERRIDE:-5000}"

export OUTPUT_TAG_OVERRIDE="${OUTPUT_TAG_OVERRIDE:-alpha8_dopt_tvinit_unet_u2_alpha_stack_c_base64d4_lr3e4_rescap025_noise01_offline}"
export MODEL_DIR_NAME_OVERRIDE="${MODEL_DIR_NAME_OVERRIDE:-${OUTPUT_TAG_OVERRIDE}}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[run] project=${PROJECT_ROOT}"
echo "[run] python=${PYTHON_BIN}"
echo "[run] alpha_json=${ALPHA_CONDITION_JSON_OVERRIDE}"
echo "[run] offline_train=${OFFLINE_TRAIN_DATASET_OVERRIDE}"
echo "[run] offline_val=${OFFLINE_VAL_DATASET_OVERRIDE}"
echo "[run] model_arch=${MODEL_ARCH_OVERRIDE} refiner=${REFINER_INPUT_MODE_OVERRIDE} unet_base=${UNET_BASE_CHANNELS_OVERRIDE} unet_depth=${UNET_DEPTH_OVERRIDE} residual_max=${UNET_RESIDUAL_MAX_OVERRIDE} base_lr=${BASE_LR_OVERRIDE}"
echo "[run] init=${INIT_METHOD_OVERRIDE} regularizer=${REGULARIZER_TYPE_OVERRIDE} detach_physical_grads=${DETACH_PHYSICAL_GRADS_OVERRIDE}"
echo "[run] data_fidelity_channel=${DATA_FIDELITY_CHANNEL_MODE_OVERRIDE} physics_residual=${PHYSICS_RESIDUAL_MODE_OVERRIDE} explicit_update=${PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE} alpha_init=${PHYSICS_EXPLICIT_UPDATE_ALPHA_INIT_OVERRIDE} alpha_max=${PHYSICS_EXPLICIT_UPDATE_MAX_OVERRIDE}"
echo "[run] batch=${N_DATA_OVERRIDE} val_batch=${VAL_BATCH_SIZE_OVERRIDE} val_random_subsample=${VAL_RANDOM_SUBSAMPLE_OVERRIDE} validation_interval=${VALIDATION_INTERVAL_OVERRIDE} n_train=${N_TRAIN_OVERRIDE} output_tag=${OUTPUT_TAG_OVERRIDE}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/models/deep_learn/train.py"
