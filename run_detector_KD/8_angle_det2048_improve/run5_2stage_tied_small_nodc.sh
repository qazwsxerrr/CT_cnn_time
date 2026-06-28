#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_k8_improve_common.sh" \
    --experiment run5_2stage_tied_small_nodc \
    "$@"
