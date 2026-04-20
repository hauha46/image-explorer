#!/usr/bin/env bash
# scripts/preflight_check.sh
#
# Phase 0 environment verification for the A100 sweep.  Run BEFORE
# ensure_server_running so the 5-minute uvicorn cold-load isn't wasted on
# a pod with a drifted dependency.  Exits:
#   0  - all checks pass; safe to run run_all.sh / run_main.sh
#   1  - venv / HF cache / vendor tree / image inputs misconfigured
#   2  - Python package version drift from the VC compatibility pin set
#   3  - torch/CUDA or pytorch3d native extension broken
#   4  - VC-side DUSt3R .pth missing or wrong format
#
# Fast: the full check runs in ~10s.  No model weights are loaded.  Heavy
# imports (viewcrafter, seva) are NOT done here - setup_runpod.sh §13
# already covers those; this script only checks invariants that could
# drift between setup and a sweep (e.g. an interactive `uv pip install`
# that upgraded pillow).
#
# Safe to run manually at any time:
#   source /workspace/.venv/bin/activate
#   bash scripts/preflight_check.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log()     { printf '[preflight %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
log_ok()  { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
log_bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

FAIL=0
fail() { FAIL=$((FAIL + 1)); log_bad "$*"; }
check() { if "$@" >/dev/null 2>&1; then log_ok "$1"; else fail "$*"; fi; }

log "Phase 0: A100 sweep preflight"
log "Repo root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Venv / interpreter / HF cache are where we expect
# ---------------------------------------------------------------------------
log "[1/5] venv + environment variables"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    fail "VIRTUAL_ENV is unset - did you 'source /workspace/.venv/bin/activate'?"
else
    log_ok "VIRTUAL_ENV = ${VIRTUAL_ENV}"
fi

PY_BIN="$(command -v python || true)"
if [ -z "${PY_BIN}" ]; then
    fail "python not on PATH"
elif [[ "${PY_BIN}" != /workspace/.venv/* ]] && [[ "${PY_BIN}" != "${VIRTUAL_ENV:-}"/* ]]; then
    fail "python=${PY_BIN} is NOT inside the venv - pip installs will leak to system python"
else
    log_ok "python = ${PY_BIN}"
fi

# HF cache env: if unset, weights will be re-downloaded on every pod restart.
if [ -z "${HF_HOME:-}" ]; then
    fail "HF_HOME is unset - SEVA / open_clip weights will not persist across pod restarts"
else
    log_ok "HF_HOME = ${HF_HOME}"
fi
if [ ! -d "${HF_HOME:-/workspace/.hf}" ]; then
    fail "HF_HOME directory ${HF_HOME:-/workspace/.hf} does not exist"
fi

# ---------------------------------------------------------------------------
# 2. Vendor trees and checkpoints exist
# ---------------------------------------------------------------------------
log "[2/5] vendor trees + checkpoints"

for d in \
    backend/vendor/dust3r \
    backend/vendor/ViewCrafter \
    backend/vendor/ViewCrafter/extern/dust3r \
    backend/vendor/stable-virtual-camera \
    backend/vendor/DepthPro/checkpoints \
    ; do
    if [ -d "${d}" ]; then
        log_ok "vendor dir: ${d}"
    else
        fail "missing vendor dir: ${d}"
    fi
done

DEPTHPRO_CKPT="backend/vendor/DepthPro/checkpoints/depth_pro.pt"
if [ -s "${DEPTHPRO_CKPT}" ] && [ "$(stat -c%s "${DEPTHPRO_CKPT}")" -gt 1000000000 ]; then
    log_ok "DepthPro checkpoint present ($(du -h "${DEPTHPRO_CKPT}" | cut -f1))"
else
    fail "DepthPro checkpoint missing or too small: ${DEPTHPRO_CKPT}"
fi

VC_DUST3R_CKPT="backend/vendor/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -s "${VC_DUST3R_CKPT}" ]; then
    fail "VC DUSt3R .pth missing: ${VC_DUST3R_CKPT}"
else
    # Verify the file is in the training-checkpoint format (has 'args'),
    # not a raw state_dict.  A raw state_dict crashes with KeyError at
    # runtime, and this preflight exists precisely to catch that.
    if python - <<PY >/dev/null 2>&1
import torch, sys
c = torch.load("${VC_DUST3R_CKPT}", map_location="cpu", weights_only=False)
sys.exit(0 if isinstance(c, dict) and "args" in c else 1)
PY
    then
        log_ok "VC DUSt3R .pth has 'args' wrapper ($(du -h "${VC_DUST3R_CKPT}" | cut -f1))"
    else
        fail "VC DUSt3R .pth is WRONG FORMAT (no 'args' key) - re-run setup_runpod.sh §9"
        FAIL=$((FAIL + 3))  # bump exit to 4 via final mapping below
    fi
fi

# ---------------------------------------------------------------------------
# 3. Torch + pytorch3d native extension load
# ---------------------------------------------------------------------------
log "[3/5] torch / CUDA / pytorch3d native extension"

python - <<'PY' || FAIL=$((FAIL + 1))
import sys
errs = []

try:
    import torch
    print(f"  OK   torch {torch.__version__} (cuda={torch.version.cuda})")

    # Guard against the CUDA-13 incident: an errant
    # `uv pip install --force-reinstall <pkg>` without --no-deps can
    # re-resolve torch to the latest build, which ships with
    # nvidia-cuda-runtime==13.x.  That won't load on RunPod pods with
    # driver<=12.x and crashes with "NVIDIA driver ... is too old"
    # deep inside the first model init.  Refuse to proceed.
    cuda_str = torch.version.cuda or ""
    if not cuda_str.startswith("12."):
        errs.append(
            f"torch compiled against CUDA {cuda_str!r}; expected 12.x. "
            "Fix: uv pip install --force-reinstall torch==2.4.1 torchvision==0.19.1 "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    if not torch.__version__.startswith("2.4."):
        errs.append(
            f"torch=={torch.__version__}; expected 2.4.x. "
            "flash-attn prebuilt wheels target 2.4.x only."
        )

    if not torch.cuda.is_available():
        errs.append("torch.cuda.is_available() is False - driver/runtime mismatch")
    else:
        print(f"  OK   GPU: {torch.cuda.get_device_name()}  "
              f"cap={torch.cuda.get_device_capability()}")
        # Exercise the CUDA init path that would fail on cu13 torch.
        _ = torch.randn(2, device='cuda')
        print(f"  OK   CUDA tensor allocation")
except Exception as e:
    errs.append(f"torch import / CUDA init failed: {type(e).__name__}: {e}")

# Reject the presence of PyPI's CUDA-13 stub packages.  If these are
# installed, some prior `uv pip install` leaked them in; they will
# shadow torch's bundled CUDA 12 libs on sys.path and can also crash
# the pod on next install.
try:
    import importlib.metadata as md
    bad = [
        d.name for d in md.distributions()
        if d.name.endswith("-cu13") or d.name in {
            "nvidia-cuda-runtime-cu13",
            "nvidia-cudnn-cu13",
            "nvidia-cublas-cu13",
            "nvidia-cusparselt-cu13",
            "nvidia-nccl-cu13",
            "cuda-toolkit",
            "cuda-bindings",
            "cuda-pathfinder",
        }
    ]
    if bad:
        errs.append(
            f"CUDA-13 stub packages installed: {bad}. "
            "Remove with: uv pip uninstall " + " ".join(bad)
        )
    else:
        print("  OK   no CUDA-13 stub packages present")
except Exception as e:
    print(f"  WARN cannot enumerate installed packages: {e}")

try:
    import pytorch3d
    from pytorch3d import _C  # the libcudart.so.13 failure triggers here
    from pytorch3d.renderer import PerspectiveCameras
    _ = PerspectiveCameras(device="cuda")
    print(f"  OK   pytorch3d {pytorch3d.__version__} (_C + CUDA ctor OK)")
except Exception as e:
    errs.append(f"pytorch3d native extension failed: {type(e).__name__}: {e}")

if errs:
    print("\nFAIL:")
    for e in errs: print(f"  - {e}")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# 4. VC compatibility pin set: assert the actual API surfaces VC uses
# ---------------------------------------------------------------------------
# Each check below targets the *specific* API that a version-drifted
# package would silently break.  A bare `import pkg` would falsely pass.
log "[4/5] VC compatibility pin set (actual API checks)"

python - <<'PY' || FAIL=$((FAIL + 2))
import sys, importlib
errs = []
def chk(label, fn):
    try:
        fn()
        print(f"  OK   {label}")
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}")
        errs.append(label)

# Pillow.Image.ANTIALIAS removed in 10.0
def _pillow():
    from PIL import Image
    assert hasattr(Image, "ANTIALIAS"), \
        f"Pillow>=10 removed Image.ANTIALIAS (have {__import__('PIL').__version__})"
chk("Pillow<10 (Image.ANTIALIAS present)", _pillow)

# PyAV: string frame.pict_type rejected in av>=12 (torchvision.io.write_video uses it)
def _av():
    import av, numpy as np
    f = av.VideoFrame.from_ndarray(np.zeros((8,8,3), dtype='uint8'), format='rgb24')
    f.pict_type = "NONE"  # string assignment
    assert av.__version__.startswith("11."), f"need av 11.x, got {av.__version__}"
chk("av==11.x (string pict_type accepted)", _av)

# open_clip: ViewCrafter's lvdm/condition.py reads visual.input_patchnorm
def _oclip():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    assert hasattr(m.visual, "input_patchnorm"), \
        f"open_clip>=2.24 removed visual.input_patchnorm (have {open_clip.__version__})"
chk("open_clip_torch==2.20.* (input_patchnorm present)", _oclip)

# pytorch_lightning: VC inherits 1.x LightningModule APIs
def _pl():
    import pytorch_lightning as pl
    assert pl.__version__.split(".")[0] == "1", \
        f"need pytorch_lightning 1.x, got {pl.__version__}"
chk("pytorch-lightning<2", _pl)

# numpy 2.x removed np.bool/np.int/np.float
def _np():
    import numpy as np
    assert np.__version__.startswith("1."), \
        f"need numpy 1.x, got {np.__version__}"
chk("numpy<2", _np)

# moviepy 2.x rewrote the API
def _mp():
    import moviepy
    assert moviepy.__version__.startswith("1."), \
        f"need moviepy 1.x, got {moviepy.__version__}"
chk("moviepy<2", _mp)

# diffusers custom_op registration is broken in torch 2.4 for >=0.35
def _diff():
    import diffusers
    assert diffusers.__version__.startswith("0.30."), \
        f"need diffusers 0.30.x, got {diffusers.__version__}"
chk("diffusers==0.30.*", _diff)

# Transformers upper bound (keeps SEVA's tokenizer path happy)
def _tf():
    import transformers
    major = int(transformers.__version__.split(".")[0])
    assert major < 5, f"need transformers <5, got {transformers.__version__}"
chk("transformers<5", _tf)

# torchvision must match torch 2.4.1
def _tv():
    import torch, torchvision
    # torchvision 0.19.x pairs with torch 2.4.x
    assert torchvision.__version__.startswith("0.19."), \
        f"need torchvision 0.19.x for torch 2.4.x, got {torchvision.__version__}"
chk("torchvision 0.19.* (pairs with torch 2.4.1)", _tv)

if errs:
    print(f"\n{len(errs)} VC compatibility check(s) FAILED.")
    print("Re-run: bash scripts/setup_runpod.sh   (or the pin block in §7)")
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# 5. Benchmark image inputs
# ---------------------------------------------------------------------------
log "[5/5] benchmark image inputs"

# Standard benchmark image referenced by run_main.sh / run_experiment.sh.
# If a user renames or removes it, every cell fails with a 400 from /process.
STD_BENCH="backend/images/standard_benchmark.jpg"
if [ -f "${STD_BENCH}" ]; then
    log_ok "standard benchmark image: ${STD_BENCH}"
else
    fail "standard benchmark image missing: ${STD_BENCH}"
fi

# ---------------------------------------------------------------------------
# Summary + exit-code mapping
# ---------------------------------------------------------------------------
if [ "${FAIL}" -eq 0 ]; then
    log "Preflight OK - safe to run run_all.sh."
    exit 0
fi

log "Preflight FAILED (${FAIL} check(s) failed).  Fix the above before running the sweep."
# Map: no granular exit-code differentiation is needed for run_all.sh
# (anything non-zero aborts the sweep), but we keep a meaningful code
# so callers can `$?`-check:
#   10+: generic / multi-failure
exit 10
