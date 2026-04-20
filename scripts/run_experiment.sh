#!/usr/bin/env bash
# scripts/run_experiment.sh
#
# CLIP text re-conditioning experiment: 6 runs total.  All use
# backend/images/standard_benchmark.jpg as input.  SEVA runs first so its
# ~90s cold load is amortized across the 3 SEVA cells, then ViewCrafter
# runs its 3 cells.
#
#   exp_01 seva   promptA           clip_lambda=0.25
#   exp_02 seva   promptB           clip_lambda=0.25
#   exp_03 seva   (no prompt)       baseline
#   exp_04 vc     promptA           (VC ignores clip_lambda internally)
#   exp_05 vc     promptB
#   exp_06 vc     (no prompt)       baseline
#
# Soft-fail: each cell is a run_cell call; a failed cell logs a FAILED line
# but the sweep continues.
#
# curl/jq only for HTTP interaction, set -euo pipefail (with soft-fail
# handled inside run_cell so -e doesn't kill the whole sweep on first failure).
#
# Output:
#   sweep.log            -- full teed log
#   exp_session_map.tsv  -- label -> session_id mapping
#   ~/sweep_results.tar.gz  (via package_results.sh experiment)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Tee all output to sweep.log (user spec).  Guard re-exec so `source`d callers
# don't trigger nested tees.
if [ "${_SWEEP_LOG_TEE:-0}" != "1" ]; then
    export _SWEEP_LOG_TEE=1
    exec > >(tee -a "${REPO_ROOT}/sweep.log") 2>&1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_common.sh"

log_info "================================================================"
log_info " CLIP EXPERIMENT (6 cells): SEVA x3 + ViewCrafter x3"
log_info " Input: backend/images/standard_benchmark.jpg"
log_info " Repo root: ${REPO_ROOT}"
log_info " Log:       ${REPO_ROOT}/sweep.log"
log_info "================================================================"

# Tool dependencies (curl + jq only per spec).
for dep in curl jq; do
    if ! command -v "${dep}" >/dev/null 2>&1; then
        log_err "Missing required tool: ${dep}"
        exit 2
    fi
done

INPUT_IMG="${REPO_ROOT}/backend/images/standard_benchmark.jpg"
if [ ! -f "${INPUT_IMG}" ]; then
    log_err "Input image missing: ${INPUT_IMG}"
    exit 2
fi

# Bring up uvicorn (no-op if the caller already did).
if ! ensure_server_running; then
    log_err "Could not bring up the backend server; aborting."
    exit 3
fi

# Configure lib_common's cell runner.
export CELL_TOTAL=6
export LOG_PREFIX="EXP"
init_session_map "${REPO_ROOT}/exp_session_map.tsv"
log_info "Session map will be written to ${REPO_ROOT}/exp_session_map.tsv"

T_SWEEP_START="$(date +%s)"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPT_A="a photo of a modern living room at golden hour with warm sunset light streaming through the tall windows"
PROMPT_B="a photo of a modern living room at night with cool moonlight outside and the brass arc lamp glowing warmly"
NEUTRAL_PROMPT="a photo of a living room interior"
CLIP_LAMBDA="0.25"

# ---------------------------------------------------------------------------
# SEVA cells (3)
# ---------------------------------------------------------------------------
# SEVA uses clip_lambda + neutral_prompt to inject the CLIP-direction; the
# SevaSynthesizer only cold-loads once, then cells 2 and 3 reuse the
# in-memory model.

run_cell "exp_01_seva_promptA_l0p25" \
    "seva" "${INPUT_IMG}" "${PROMPT_A}" "${CLIP_LAMBDA}" "${NEUTRAL_PROMPT}"

run_cell "exp_02_seva_promptB_l0p25" \
    "seva" "${INPUT_IMG}" "${PROMPT_B}" "${CLIP_LAMBDA}" "${NEUTRAL_PROMPT}"

run_cell "exp_03_seva_baseline" \
    "seva" "${INPUT_IMG}" "" "" ""

# ---------------------------------------------------------------------------
# ViewCrafter cells (3)
# ---------------------------------------------------------------------------
# ViewCrafter consumes `prompt` directly as its video-diffusion text guide.
# It does not use clip_lambda / neutral_prompt; the /process endpoint
# silently ignores those fields for non-SEVA backends, so we just omit them.

run_cell "exp_04_vc_promptA" \
    "viewcrafter" "${INPUT_IMG}" "${PROMPT_A}" "" ""

run_cell "exp_05_vc_promptB" \
    "viewcrafter" "${INPUT_IMG}" "${PROMPT_B}" "" ""

run_cell "exp_06_vc_baseline" \
    "viewcrafter" "${INPUT_IMG}" "" "" ""

T_SWEEP_END="$(date +%s)"
ELAPSED=$((T_SWEEP_END - T_SWEEP_START))

log_info ""
log_info "================================================================"
log_info " CLIP EXPERIMENT done in ${ELAPSED}s"
log_info " Session map: ${REPO_ROOT}/exp_session_map.tsv"
log_info "================================================================"

# Dump the session map so it's visible in the log tail.
if [ -f "${REPO_ROOT}/exp_session_map.tsv" ]; then
    log_info "Session map contents:"
    cat "${REPO_ROOT}/exp_session_map.tsv"
fi

# Per user spec: package a tarball of backend/uploads/ (minus _seva_work) plus
# the log into ~/sweep_results.tar.gz and print its path.  Soft-fail here
# too: packaging problems should not mask the fact that the sweep finished.
set +e
bash "${SCRIPT_DIR}/package_results.sh" experiment
PKG_RC=$?
set -e
if [ "${PKG_RC}" -ne 0 ]; then
    log_warn "package_results.sh experiment exited ${PKG_RC}; artifacts are still on disk under backend/uploads/"
fi

exit 0
