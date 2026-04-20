#!/usr/bin/env bash
# scripts/run_all.sh
#
# Sequencer: runs the main sweep, then the CLIP experiment sweep, then
# packages everything into ~/a100_results.tar.gz.  Leave it running in tmux.
#
# A single uvicorn process is brought up once at the start and torn down at
# the end, so SEVA / ViewCrafter / DepthPro / DUSt3R only cold-load once
# across all 12 cells (6 main + 6 experiment).
#
# Soft-fail across scripts: if run_main.sh hits a fatal error, run_experiment.sh
# still runs; if either one exits non-zero, packaging still runs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Tee the run_all-level summary to run_all.log so an operator attaching to
# tmux sees a clean timeline even if the child scripts are teeing to their
# own logs.
if [ "${_RUN_ALL_TEE:-0}" != "1" ]; then
    export _RUN_ALL_TEE=1
    exec > >(tee -a "${REPO_ROOT}/run_all.log") 2>&1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_common.sh"

log_info "################################################################"
log_info " A100 SWEEP (preflight + main + experiment + package)"
log_info " Repo root: ${REPO_ROOT}"
log_info " Phases:"
log_info "   Phase 0: preflight_check.sh  (fast dep sanity, ~10s)"
log_info "   Phase A: ensure_server_running (uvicorn cold-load, ~3-5 min)"
log_info "   Phase B: run_main.sh         (6 cells)"
log_info "   Phase C: run_experiment.sh   (6 cells)"
log_info "   Phase D: package_results.sh  (tarball to ~/a100_results.tar.gz)"
log_info " Logs:"
log_info "   ${REPO_ROOT}/run_all.log         (this file)"
log_info "   ${REPO_ROOT}/main.log            (run_main.sh)"
log_info "   ${REPO_ROOT}/sweep.log           (run_experiment.sh)"
log_info "   ${REPO_ROOT}/server.log          (uvicorn)"
log_info " Bypass preflight (NOT recommended) with: PREFLIGHT_SKIP=1 bash scripts/run_all.sh"
log_info "################################################################"

T_START="$(date +%s)"

# Cleanly stop the uvicorn we start here, even on Ctrl-C / kill.
cleanup() {
    log_info "Cleanup: stopping uvicorn if we started it ..."
    stop_server_if_we_started_it
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Phase 0: preflight (fast dependency sanity check, ~10s)
# ---------------------------------------------------------------------------
# This runs BEFORE we spawn uvicorn because the server cold-load takes
# ~3-5 min (SEVA weights + DepthPro + open_clip), and historically every
# version-drift failure surfaces only after that wait.  Preflight catches
# all 9 known drift bugs (pillow, av, open_clip, numpy, diffusers,
# pytorch_lightning, moviepy, torchvision-torch pair, VC DUSt3R .pth)
# in ~10 seconds of pure imports, so a broken pod aborts immediately.
#
# Skip with PREFLIGHT_SKIP=1 (e.g. when you're debugging inside a
# known-good env and just want to rerun the sweep).
log_info "--- Phase 0: preflight ---"
if [ "${PREFLIGHT_SKIP:-0}" = "1" ]; then
    log_warn "PREFLIGHT_SKIP=1 - skipping preflight; hope you know what you're doing."
else
    set +e
    bash "${SCRIPT_DIR}/preflight_check.sh"
    PREFLIGHT_RC=$?
    set -e
    if [ "${PREFLIGHT_RC}" -ne 0 ]; then
        log_err "Preflight FAILED (rc=${PREFLIGHT_RC}); aborting sweep BEFORE starting server."
        log_err "Fix the reported issues and re-run, or bypass with PREFLIGHT_SKIP=1 bash scripts/run_all.sh."
        exit 2
    fi
    log_info "Preflight OK."
fi

# ---------------------------------------------------------------------------
# Phase A: Start the server once; both sub-scripts reuse it.
# ---------------------------------------------------------------------------
log_info "--- Phase A: starting backend server ---"
if ! ensure_server_running; then
    log_err "Backend server did not start; aborting sweep."
    exit 3
fi

# 2. Main run (6 cells).
log_info "--- Phase B: run_main.sh ---"
set +e
bash "${SCRIPT_DIR}/run_main.sh"
MAIN_RC=$?
set -e
if [ "${MAIN_RC}" -ne 0 ]; then
    log_warn "run_main.sh exited with code ${MAIN_RC}; continuing to experiment anyway."
else
    log_info "run_main.sh OK."
fi

# 3. CLIP experiment (6 cells).  run_experiment.sh also writes its own
# ~/sweep_results.tar.gz via package_results.sh experiment, which is fine -
# the final run_all package subsumes that with the combined manifest.
log_info "--- Phase C: run_experiment.sh ---"
set +e
bash "${SCRIPT_DIR}/run_experiment.sh"
EXP_RC=$?
set -e
if [ "${EXP_RC}" -ne 0 ]; then
    log_warn "run_experiment.sh exited with code ${EXP_RC}; continuing to packaging anyway."
else
    log_info "run_experiment.sh OK."
fi

# 4. Combined tarball (includes both session maps and both logs).
log_info "--- Phase D: package_results.sh all ---"
set +e
bash "${SCRIPT_DIR}/package_results.sh" all
PKG_RC=$?
set -e

T_END="$(date +%s)"
ELAPSED=$((T_END - T_START))

log_info "################################################################"
log_info " A100 SWEEP finished in ${ELAPSED}s"
log_info " run_main.sh rc=${MAIN_RC}"
log_info " run_experiment.sh rc=${EXP_RC}"
log_info " package_results.sh rc=${PKG_RC}"
if [ "${PKG_RC}" -eq 0 ]; then
    log_info " Download the tarball with e.g.:"
    log_info "   cd ~ && runpodctl send a100_results.tar.gz"
fi
log_info "################################################################"

# cleanup() runs via trap; exit 0 so a failed cell doesn't make systemd-style
# supervisors think the sweep script itself crashed.
exit 0
