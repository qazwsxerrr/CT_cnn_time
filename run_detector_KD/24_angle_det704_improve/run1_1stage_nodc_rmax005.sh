#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_k24_improve_common.sh" \
    --experiment run1_1stage_nodc_rmax005 \
    "$@"
