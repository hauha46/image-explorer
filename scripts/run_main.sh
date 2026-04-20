#!/usr/bin/env bash
# scripts/run_main.sh
#
# The 6 main pipeline runs for the final-project sweep.  Each cell invokes
# the full backend/app.py pipeline via POST /process, which covers:
#
#   DepthPro -> NVS (SEVA or ViewCrafter) -> DUSt3R -> Mesh -> Scene compose
#
# Every cell gets its own uploads/<session_id>/ folder containing
# run_report.txt, run_info.json, depth.png, depth_metric.npy, views/view_*.png,
# views/trajectory.json, scene.glb, scene_mesh.glb, scene.json.  The label ->
# session_id mapping is recorded in main_session_map.tsv so the downloaded
# tarball can be re-keyed by human label.
#
# Soft-fail: each cell is wrapped in run_cell (see lib_common.sh) which
# catches submit/wait errors and logs them but never aborts the script.
#
# All stdout + stderr is teed to main.log.  Safe to re-run; each invocation
# creates fresh session folders (new session_ids), so nothing is clobbered.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Re-exec ourselves with output teed to main.log unless we're already teed.
if [ "${_MAIN_LOG_TEE:-0}" != "1" ]; then
    export _MAIN_LOG_TEE=1
    exec > >(tee -a "${REPO_ROOT}/main.log") 2>&1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_common.sh"

log_info "================================================================"
log_info " MAIN RUN (6 cells): SEVA x2 + ViewCrafter x4"
log_info " Repo root: ${REPO_ROOT}"
log_info " Log:       ${REPO_ROOT}/main.log"
log_info "================================================================"

# Sanity checks.
for dep in curl jq; do
    if ! command -v "${dep}" >/dev/null 2>&1; then
        log_err "Missing required tool: ${dep}"
        exit 2
    fi
done

STANDARD_IMG="${REPO_ROOT}/backend/images/standard_benchmark.jpg"
TECHBRO_IMG="${REPO_ROOT}/backend/images/techbro_room.jpg"
for f in "${STANDARD_IMG}" "${TECHBRO_IMG}"; do
    if [ ! -f "${f}" ]; then
        log_err "Input image missing: ${f}"
        exit 2
    fi
done

# Bring up uvicorn if not already running.  If the caller (run_all.sh)
# already did this, ensure_server_running is a no-op + reuses it.
if ! ensure_server_running; then
    log_err "Could not bring up the backend server; aborting."
    exit 3
fi

# Configure lib_common's cell runner.
export CELL_TOTAL=6
export LOG_PREFIX="MAIN"
init_session_map "${REPO_ROOT}/main_session_map.tsv"
log_info "Session map will be written to ${REPO_ROOT}/main_session_map.tsv"

T_SWEEP_START="$(date +%s)"

# ---------------------------------------------------------------------------
# Prompts (kept here so they show up grep-ably in one place)
# ---------------------------------------------------------------------------
PROMPT_STEP_LEFT="Please keep the same room and same furniture, the camera has stepped 2 feet to the left and is looking slightly right towards the original center."
PROMPT_ORBIT_360="Please generate a view around the room as if the camera has rotated 360 degrees around."

# ---------------------------------------------------------------------------
# The 6 cells (SEVA runs first so SEVA only cold-loads once before the sweep
# flips to ViewCrafter for the remaining cells).
# ---------------------------------------------------------------------------

# 1) SEVA on standard_benchmark (no prompt => pure image-conditioned SEVA)
run_cell "main_01_seva_standard" \
    "seva" "${STANDARD_IMG}" "" "" ""

# 2) SEVA on techbro_room (no prompt)
run_cell "main_02_seva_techbro" \
    "seva" "${TECHBRO_IMG}" "" "" ""

# 3) ViewCrafter on standard, no prompt
run_cell "main_03_vc_standard_noprompt" \
    "viewcrafter" "${STANDARD_IMG}" "" "" ""

# 4) ViewCrafter on standard, "stepped 2 feet to the left"
run_cell "main_04_vc_standard_step_left" \
    "viewcrafter" "${STANDARD_IMG}" "${PROMPT_STEP_LEFT}" "" ""

# 5) ViewCrafter on standard, "360 degree orbit"
run_cell "main_05_vc_standard_orbit360" \
    "viewcrafter" "${STANDARD_IMG}" "${PROMPT_ORBIT_360}" "" ""

# 6) ViewCrafter on techbro_room, "360 degree orbit"
run_cell "main_06_vc_techbro_orbit360" \
    "viewcrafter" "${TECHBRO_IMG}" "${PROMPT_ORBIT_360}" "" ""

T_SWEEP_END="$(date +%s)"
ELAPSED=$((T_SWEEP_END - T_SWEEP_START))

log_info ""
log_info "================================================================"
log_info " MAIN RUN done in ${ELAPSED}s"
log_info " Session map: ${REPO_ROOT}/main_session_map.tsv"
log_info " Per-run artifacts: ${REPO_ROOT}/backend/uploads/<session_id>/"
log_info "================================================================"

# Dump the session map so it's visible in the log tail for operators.
if [ -f "${REPO_ROOT}/main_session_map.tsv" ]; then
    log_info "Session map contents:"
    cat "${REPO_ROOT}/main_session_map.tsv"
fi

exit 0
