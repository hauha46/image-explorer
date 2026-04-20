# scripts/lib_common.sh
#
# Shared helpers for the A100 sweep scripts (run_main.sh, run_experiment.sh,
# run_all.sh, package_results.sh).  Source this file; do not execute it.
#
#   source "$(dirname "$0")/lib_common.sh"
#
# Contract:
#   - Does not set -e / -u / -o pipefail itself; each caller decides.
#   - Does not cd anywhere; caller controls working directory.
#   - Single fixed server URL: http://127.0.0.1:9876
#   - Single fixed session-map file per caller (caller sets SESSION_MAP_FILE).

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SERVER_URL="${SERVER_URL:-http://127.0.0.1:9876}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-9876}"
SERVER_PID_FILE="${SERVER_PID_FILE:-/tmp/image_explorer_server.pid}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-server.log}"
SERVER_STARTUP_TIMEOUT_S="${SERVER_STARTUP_TIMEOUT_S:-180}"

# Per-cell watchdog: each /process call runs Depth + NVS + DUSt3R + Mesh +
# Compose.  On A100 the slowest realistic cell is ~8 min; pad to 30 min so a
# slow ViewCrafter checkpoint download on first use doesn't falsely trip the
# timeout.
RUN_CELL_TIMEOUT_S="${RUN_CELL_TIMEOUT_S:-1800}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    # log LEVEL message...
    local level="$1"; shift
    printf '[%s %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${level}" "$*"
}
log_info() { log INFO "$@"; }
log_warn() { log WARN "$@"; }
log_err()  { log ERR  "$@"; }

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_server_is_up() {
    # True iff the FastAPI root responds 2xx.  Uses a short connect timeout
    # so the "is it up?" probe is fast on cold boxes.
    curl -fsS --max-time 3 "${SERVER_URL}/" >/dev/null 2>&1
}

ensure_server_running() {
    # If a uvicorn instance is already responding, reuse it.  Otherwise spawn
    # one in the background, record the PID, wait up to
    # SERVER_STARTUP_TIMEOUT_S for it to start answering /, and return 0.
    if _server_is_up; then
        log_info "Server already responding at ${SERVER_URL}; reusing it."
        return 0
    fi

    # backend/app.py does `from scene_processor import SceneProcessor` as a
    # top-level absolute import, so uvicorn must be launched with backend/ as
    # the working directory (same convention as docs/SETUP_LINUX.md).
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    log_info "Launching uvicorn backend.app (logs -> ${repo_root}/${SERVER_LOG_FILE}) ..."
    # nohup + & so the server survives the caller's shell exit.  We capture
    # $! directly (the uvicorn python PID) rather than using setsid, which
    # is less portable and forks by default, making PID capture racy.
    (
        cd "${repo_root}/backend"
        nohup python -u -m uvicorn app:app \
            --host "${SERVER_HOST}" --port "${SERVER_PORT}" \
            --log-level info \
            > "${repo_root}/${SERVER_LOG_FILE}" 2>&1 &
        echo $! > "${SERVER_PID_FILE}"
        # Detach from the subshell so the python process is reparented to
        # init and won't receive SIGHUP when this outer script exits.
        disown
    )
    local pid
    pid="$(cat "${SERVER_PID_FILE}" 2>/dev/null || true)"
    log_info "uvicorn pid=${pid}; waiting up to ${SERVER_STARTUP_TIMEOUT_S}s for /..."

    local waited=0
    while [ "${waited}" -lt "${SERVER_STARTUP_TIMEOUT_S}" ]; do
        if _server_is_up; then
            log_info "Server is up after ${waited}s."
            return 0
        fi
        # Bail early if the process has already died (install error etc).
        if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
            log_err "uvicorn pid=${pid} exited before /; tail of ${SERVER_LOG_FILE}:"
            tail -n 40 "${repo_root}/${SERVER_LOG_FILE}" || true
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    log_err "Server did not come up within ${SERVER_STARTUP_TIMEOUT_S}s; tail of ${SERVER_LOG_FILE}:"
    tail -n 40 "${repo_root}/${SERVER_LOG_FILE}" || true
    return 1
}

stop_server_if_we_started_it() {
    # Only kill a uvicorn we spawned ourselves.  If the user started the
    # server manually in another tmux window, do not touch it.
    if [ ! -f "${SERVER_PID_FILE}" ]; then
        return 0
    fi
    local pid
    pid="$(cat "${SERVER_PID_FILE}" 2>/dev/null || true)"
    rm -f "${SERVER_PID_FILE}"
    if [ -z "${pid}" ]; then
        return 0
    fi
    if kill -0 "${pid}" 2>/dev/null; then
        log_info "Stopping uvicorn pid=${pid} ..."
        kill -TERM "${pid}" 2>/dev/null || true
        # Give it a grace period, then SIGKILL if still alive.
        for _ in 1 2 3 4 5; do
            kill -0 "${pid}" 2>/dev/null || return 0
            sleep 1
        done
        kill -KILL "${pid}" 2>/dev/null || true
    fi
}

# ---------------------------------------------------------------------------
# /process submit + /status poll
# ---------------------------------------------------------------------------

submit() {
    # submit MODEL IMAGE_PATH [PROMPT] [CLIP_LAMBDA] [NEUTRAL_PROMPT]
    # Echoes the session_id on stdout; non-zero exit on HTTP or parse error.
    local model="$1"
    local image_path="$2"
    local prompt="${3:-}"
    local clip_lambda="${4:-}"
    local neutral_prompt="${5:-}"

    if [ ! -f "${image_path}" ]; then
        log_err "submit: image not found: ${image_path}"
        return 1
    fi

    # Build the curl -F arg list; optional fields are only passed if set.
    local -a args=(-sS -X POST "${SERVER_URL}/process")
    args+=(-F "file=@${image_path}")
    args+=(-F "model=${model}")
    if [ -n "${prompt}" ]; then
        args+=(-F "prompt=${prompt}")
    fi
    if [ -n "${clip_lambda}" ]; then
        args+=(-F "clip_lambda=${clip_lambda}")
    fi
    if [ -n "${neutral_prompt}" ]; then
        args+=(-F "neutral_prompt=${neutral_prompt}")
    fi

    local resp
    if ! resp="$(curl "${args[@]}")"; then
        log_err "submit: curl failed for ${model} ${image_path}"
        return 1
    fi
    local sid
    sid="$(printf '%s' "${resp}" | jq -r '.session_id // empty' 2>/dev/null || true)"
    if [ -z "${sid}" ] || [ "${sid}" = "null" ]; then
        log_err "submit: could not parse session_id from response: ${resp}"
        return 1
    fi
    printf '%s\n' "${sid}"
}

wait_for() {
    # wait_for SESSION_ID [TIMEOUT_S]
    # Polls /status/{sid} every 5s.
    #   0 -> status==complete
    #   1 -> status==error   (prints current_step / reason)
    #   2 -> timed out
    local sid="$1"
    local timeout_s="${2:-${RUN_CELL_TIMEOUT_S}}"

    local waited=0
    local last_step=""
    while [ "${waited}" -lt "${timeout_s}" ]; do
        local resp status step
        resp="$(curl -fsS --max-time 10 "${SERVER_URL}/status/${sid}" 2>/dev/null || true)"
        if [ -n "${resp}" ]; then
            status="$(printf '%s' "${resp}" | jq -r '.status // "unknown"')"
            step="$(printf '%s' "${resp}" | jq -r '.current_step // ""')"
            if [ "${step}" != "${last_step}" ] && [ -n "${step}" ]; then
                log_info "  sid=${sid} step=\"${step}\" status=${status}"
                last_step="${step}"
            fi
            case "${status}" in
                complete) return 0 ;;
                error)
                    log_err "  sid=${sid} errored: ${step}"
                    return 1
                    ;;
            esac
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log_err "  sid=${sid} timed out after ${timeout_s}s"
    return 2
}

# ---------------------------------------------------------------------------
# Cell runner (soft-fail wrapper used by run_main.sh and run_experiment.sh)
# ---------------------------------------------------------------------------

# Caller populates these before calling run_cell:
#   CELL_TOTAL       -- how many cells in this script (for === N/TOTAL ===)
#   SESSION_MAP_FILE -- path to write label<TAB>session_id pairs
#   LOG_PREFIX       -- short tag used in run headers, e.g. "MAIN" or "EXP"
# run_cell increments its own counter across calls.
_CELL_INDEX=0

init_session_map() {
    # init_session_map PATH
    # Truncate and write a TSV header so downstream tooling can parse it.
    SESSION_MAP_FILE="$1"
    : > "${SESSION_MAP_FILE}"
    printf 'label\tsession_id\tmodel\timage\tprompt\tclip_lambda\tstatus\truntime_s\n' \
        > "${SESSION_MAP_FILE}"
}

_append_session_map() {
    # _append_session_map LABEL SID MODEL IMAGE PROMPT CLIP_LAMBDA STATUS RUNTIME_S
    local label="$1" sid="$2" model="$3" image="$4"
    local prompt="$5" clip_lambda="$6" status="$7" runtime="$8"
    # Strip tabs/newlines from user-supplied prompt so the TSV stays clean.
    prompt="${prompt//	/ }"
    prompt="${prompt//$'\n'/ }"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${sid}" "${model}" "${image}" "${prompt}" "${clip_lambda}" "${status}" "${runtime}" \
        >> "${SESSION_MAP_FILE}"
}

run_cell() {
    # run_cell LABEL MODEL IMAGE_PATH [PROMPT] [CLIP_LAMBDA] [NEUTRAL_PROMPT]
    # Never exits non-zero; soft-fail is the whole point.  Exit code reflects
    # only catastrophic bugs in this helper itself (unlikely).
    local label="$1"
    local model="$2"
    local image_path="$3"
    local prompt="${4:-}"
    local clip_lambda="${5:-}"
    local neutral_prompt="${6:-}"

    _CELL_INDEX=$((_CELL_INDEX + 1))
    local total="${CELL_TOTAL:-?}"
    local prefix="${LOG_PREFIX:-CELL}"

    printf '\n'
    log_info "=== ${prefix} Run ${_CELL_INDEX}/${total} (${label}) === started"
    log_info "  model=${model} image=${image_path}"
    if [ -n "${prompt}" ];       then log_info "  prompt=\"${prompt}\""; fi
    if [ -n "${clip_lambda}" ];  then log_info "  clip_lambda=${clip_lambda}"; fi
    if [ -n "${neutral_prompt}" ]; then log_info "  neutral_prompt=\"${neutral_prompt}\""; fi

    local t_start t_end runtime
    t_start="$(date +%s)"

    # Submit.
    local sid
    set +e
    sid="$(submit "${model}" "${image_path}" "${prompt}" "${clip_lambda}" "${neutral_prompt}")"
    local submit_rc=$?
    set -e
    if [ "${submit_rc}" -ne 0 ] || [ -z "${sid}" ]; then
        t_end="$(date +%s)"; runtime=$((t_end - t_start))
        log_err "=== ${prefix} Run ${_CELL_INDEX}/${total} (${label}) === FAILED (submit) elapsed=${runtime}s"
        if [ -n "${SESSION_MAP_FILE:-}" ]; then
            _append_session_map "${label}" "-" "${model}" "${image_path}" \
                "${prompt}" "${clip_lambda}" "submit_failed" "${runtime}"
        fi
        return 0
    fi
    log_info "  submitted sid=${sid}"

    # Wait.
    set +e
    wait_for "${sid}" "${RUN_CELL_TIMEOUT_S}"
    local wait_rc=$?
    set -e

    t_end="$(date +%s)"; runtime=$((t_end - t_start))

    local status_word
    case "${wait_rc}" in
        0) status_word="complete" ;;
        1) status_word="error" ;;
        2) status_word="timeout" ;;
        *) status_word="unknown_rc_${wait_rc}" ;;
    esac

    if [ -n "${SESSION_MAP_FILE:-}" ]; then
        _append_session_map "${label}" "${sid}" "${model}" "${image_path}" \
            "${prompt}" "${clip_lambda}" "${status_word}" "${runtime}"
    fi

    if [ "${wait_rc}" -eq 0 ]; then
        log_info "=== ${prefix} Run ${_CELL_INDEX}/${total} (${label}) === done sid=${sid} elapsed=${runtime}s"
    else
        log_err "=== ${prefix} Run ${_CELL_INDEX}/${total} (${label}) === FAILED status=${status_word} sid=${sid} elapsed=${runtime}s"
    fi
    return 0
}
