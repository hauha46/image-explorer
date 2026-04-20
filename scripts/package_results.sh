#!/usr/bin/env bash
# scripts/package_results.sh [main|experiment|all]
#
# Builds a downloadable tarball of the A100 sweep artifacts.
#
# Usage:
#   bash scripts/package_results.sh main        -> ~/main_results.tar.gz
#   bash scripts/package_results.sh experiment  -> ~/sweep_results.tar.gz
#   bash scripts/package_results.sh all         -> ~/a100_results.tar.gz
#
# Contents include everything useful for the report:
#   - backend/uploads/<session_id>/run_report.txt   (human)
#   - backend/uploads/<session_id>/run_info.json    (machine)
#   - backend/uploads/<session_id>/views/*.png
#   - backend/uploads/<session_id>/views/trajectory.json
#   - backend/uploads/<session_id>/{scene.glb,scene_mesh.glb,scene.json}
#   - backend/uploads/<session_id>/{depth.png,depth_metric.npy,depth.npy}
#   - backend/uploads/<session_id>/input.jpg
#   - backend/uploads/sessions_index.jsonl          (one JSON row per run)
#   - main_session_map.tsv / exp_session_map.tsv    (label -> session_id)
#   - main.log / sweep.log / run_all.log / server.log
#   - MANIFEST.md                                   (what each file means)
#
# Excludes:
#   - backend/uploads/<session_id>/views/_seva_work/  (large SEVA staging,
#       ~50-200 MB per run; the final view PNGs are elsewhere under views/)
#   - *.pyc, __pycache__

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-all}"
case "${MODE}" in
    main)       OUT_NAME="main_results.tar.gz" ;;
    experiment) OUT_NAME="sweep_results.tar.gz" ;;
    all)        OUT_NAME="a100_results.tar.gz" ;;
    *)
        printf 'usage: %s [main|experiment|all]\n' "$0" >&2
        exit 2
        ;;
esac

OUT_PATH="${HOME}/${OUT_NAME}"

log() { printf '[package %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log "mode=${MODE} -> ${OUT_PATH}"

# ---------------------------------------------------------------------------
# Build a staging directory that mirrors what we want in the tarball.  We
# copy (not symlink) so the tarball is self-contained regardless of how the
# user extracts it.
# ---------------------------------------------------------------------------
STAGE="$(mktemp -d -t image_explorer_pkg_XXXXXX)"
trap 'rm -rf "${STAGE}"' EXIT

STAGE_ROOT="${STAGE}/a100_results"
mkdir -p "${STAGE_ROOT}"

# ---------------------------------------------------------------------------
# Copy session folders (filtered).
# ---------------------------------------------------------------------------
if [ -d "${REPO_ROOT}/backend/uploads" ]; then
    log "Copying backend/uploads/ (excluding views/_seva_work/) ..."
    mkdir -p "${STAGE_ROOT}/backend/uploads"
    # rsync is available on every RunPod PyTorch image; it handles the
    # _seva_work exclusion cleanly and preserves file sizes.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a \
            --exclude '*/views/_seva_work/' \
            --exclude '*/views/_seva_work' \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            "${REPO_ROOT}/backend/uploads/" \
            "${STAGE_ROOT}/backend/uploads/"
    else
        # Fallback: tar | tar.  Portable but slower.
        ( cd "${REPO_ROOT}" && \
          tar -cf - \
              --exclude='backend/uploads/*/views/_seva_work' \
              --exclude='__pycache__' --exclude='*.pyc' \
              backend/uploads ) \
        | ( cd "${STAGE_ROOT}" && tar -xf - )
    fi
else
    log "WARN: backend/uploads/ does not exist; there are no session artifacts to package."
    mkdir -p "${STAGE_ROOT}/backend/uploads"
fi

# ---------------------------------------------------------------------------
# Copy logs + session maps at the top level of the archive.
# ---------------------------------------------------------------------------
copy_if_exists() {
    if [ -f "${REPO_ROOT}/$1" ]; then
        cp "${REPO_ROOT}/$1" "${STAGE_ROOT}/$1"
        log "  + $1"
    fi
}

case "${MODE}" in
    main)
        copy_if_exists "main.log"
        copy_if_exists "main_session_map.tsv"
        copy_if_exists "server.log"
        ;;
    experiment)
        copy_if_exists "sweep.log"
        copy_if_exists "exp_session_map.tsv"
        copy_if_exists "server.log"
        ;;
    all)
        copy_if_exists "main.log"
        copy_if_exists "sweep.log"
        copy_if_exists "run_all.log"
        copy_if_exists "server.log"
        copy_if_exists "main_session_map.tsv"
        copy_if_exists "exp_session_map.tsv"
        ;;
esac

# ---------------------------------------------------------------------------
# Generate MANIFEST.md explaining every file in the archive.  This is the
# "documented file for each part" deliverable: an operator can extract the
# tarball, open MANIFEST.md, and know what every artifact is without
# reading any source code.
# ---------------------------------------------------------------------------
MANIFEST="${STAGE_ROOT}/MANIFEST.md"
{
    printf '# A100 Sweep Results Manifest\n\n'
    printf '- Generated: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    printf '- Host:      %s\n' "$(hostname || echo unknown)"
    printf '- Mode:      %s\n' "${MODE}"
    if command -v git >/dev/null 2>&1 && [ -d "${REPO_ROOT}/.git" ]; then
        printf '- Git SHA:   %s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    fi
    printf '\n'

    cat <<'EOF'
## Top-level files

- `MANIFEST.md`                -- this file
- `main.log`                   -- full stdout+stderr from the 6 "main run" pipeline cells
- `sweep.log`                  -- full stdout+stderr from the 6 CLIP experiment cells
- `run_all.log`                -- run_all.sh orchestration timeline
- `server.log`                 -- FastAPI / uvicorn server log (covers both sweeps)
- `main_session_map.tsv`       -- TSV: human label -> session_id for the main run
- `exp_session_map.tsv`        -- TSV: human label -> session_id for the CLIP experiment
- `backend/uploads/`           -- one subdirectory per completed /process call
- `backend/uploads/sessions_index.jsonl` -- one JSON row per run (compact summary
                                  produced by backend/app.py::_append_sessions_index)

## Per-session folder layout

Every `backend/uploads/<session_id>/` directory contains everything from one
run of the full pipeline: Depth -> NVS (SEVA or ViewCrafter) -> DUSt3R ->
Mesh -> Scene composition.  Files:

- `input.jpg`                  -- the uploaded image (possibly downscaled to 1920px
                                  on the longest side for pipeline VRAM headroom)
- `depth.png`                  -- DepthPro visualization (grayscale, 8-bit)
- `depth_metric.npy`           -- DepthPro raw metric depth in meters, float32,
                                  same spatial resolution as the resized input
- `depth.npy`                  -- (optional) intermediate depth array
- `views/view_000.png` ...     -- novel views output by the chosen NVS backend
                                  (10 frames for both SEVA and ViewCrafter)
- `views/trajectory.json`      -- camera trajectory metadata.  For SEVA: every
                                  c2w matrix + K + fov.  For ViewCrafter: the
                                  parametric (d_phi, d_theta, d_r) trajectory.
- `scene.glb`                  -- DUSt3R reconstructed point cloud (GLB format,
                                  viewable in https://gltf-viewer.donmccurdy.com/)
- `scene_mesh.glb`             -- ball-pivoting surface mesh from scene.glb
                                  (may be missing if Stage 5 soft-failed; see
                                  run_report.txt for status)
- `scene.json`                 -- final scene composition used by the viewer UI
- `run_report.txt`             -- human-readable multi-section report:
                                  per-stage timings, file manifest with sizes,
                                  chosen model + prompt + hyperparameters
- `run_info.json`              -- machine-readable version of run_report.txt;
                                  includes git_sha, input dims, nvs_params,
                                  mesh status, per-stage timings in seconds

## NVS-backend-specific notes

- SEVA runs include `dtype`, `num_steps`, `clip_lambda`, `neutral_prompt` in
  run_info.json -> nvs_params.  When `clip_lambda > 0` and `prompt != null`,
  the CLIP-direction injection (see docs/RESEARCH_NOTES.md in the repo) was
  active.  Baseline SEVA cells set `prompt=null`; the field is just absent.
- ViewCrafter runs include a `vc_params` block with the exact d_phi / d_theta
  / d_r trajectory used, diffusion hyperparameters, point-cloud filtering
  thresholds, and the checkpoint paths.  Great for the report appendix.

## How to load this in pandas for your report

```python
import json, pathlib
rows = []
for p in pathlib.Path("backend/uploads").rglob("run_info.json"):
    rows.append(json.loads(p.read_text()))
import pandas as pd
df = pd.DataFrame(rows)
df = df[["session_id", "model_name", "nvs_params", "timings_seconds", "status"]]
```

or one-line-per-run via `backend/uploads/sessions_index.jsonl`:

```python
import pandas as pd
df = pd.read_json("backend/uploads/sessions_index.jsonl", lines=True)
```

## Session map (label -> session_id)

Files `main_session_map.tsv` and `exp_session_map.tsv` map each cell's
human-readable label (e.g. `main_04_vc_standard_step_left`) to the session_id
used as the folder name under `backend/uploads/`.  Use these to find the
directory matching a given row in your report.

EOF

    # Inline the session maps into the manifest so a user reading MANIFEST.md
    # has a complete view without opening extra files.
    for map in main_session_map.tsv exp_session_map.tsv; do
        if [ -f "${STAGE_ROOT}/${map}" ]; then
            printf '### %s\n\n```\n' "${map}"
            cat "${STAGE_ROOT}/${map}"
            printf '```\n\n'
        fi
    done

    # Summary statistics: total sessions, total size, tarball size on disk.
    if [ -d "${STAGE_ROOT}/backend/uploads" ]; then
        count="$(find "${STAGE_ROOT}/backend/uploads" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
        size="$(du -sh "${STAGE_ROOT}/backend/uploads" 2>/dev/null | awk '{print $1}')"
        printf '## Summary\n\n- Session folders: %s\n- uploads/ size:  %s\n' "${count}" "${size}"
    fi
} > "${MANIFEST}"
log "  + MANIFEST.md ($(wc -l < "${MANIFEST}") lines)"

# ---------------------------------------------------------------------------
# Build the tarball.  gzip level 6 is a good quality/speed tradeoff for the
# PNG+binary mix we're shipping.
# ---------------------------------------------------------------------------
log "Creating ${OUT_PATH} ..."
( cd "${STAGE}" && tar -czf "${OUT_PATH}" a100_results )

if [ ! -f "${OUT_PATH}" ]; then
    log "ERROR: tarball was not created."
    exit 1
fi

SIZE_BYTES="$(stat -c '%s' "${OUT_PATH}" 2>/dev/null || stat -f '%z' "${OUT_PATH}" 2>/dev/null || echo 0)"
SIZE_HUMAN="$(ls -lh "${OUT_PATH}" | awk '{print $5}')"
log "Packaged results -> ${OUT_PATH} (${SIZE_HUMAN} / ${SIZE_BYTES} bytes)"
log ""
log "To download from this pod to Windows:"
log "  (on pod)      cd ~ && runpodctl send ${OUT_NAME}"
log "  (on Windows)  .\\runpodctl.exe receive <code>"
log "  (on Windows)  tar -xzf ${OUT_NAME}"

# The final printed path is the primary machine-readable output of this
# script; keep it on its own line so callers can `tail -n1` cleanly.
printf '%s\n' "${OUT_PATH}"
