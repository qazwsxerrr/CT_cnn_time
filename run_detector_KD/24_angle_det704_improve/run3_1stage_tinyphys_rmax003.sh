#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_k24_improve_common.sh" \
    --experiment run3_1stage_tinyphys_rmax003 \
    "$@"
