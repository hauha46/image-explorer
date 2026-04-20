#!/usr/bin/env bash
# scripts/setup_runpod.sh
#
# Idempotent bootstrap for a fresh RunPod A100 pod (or any Ubuntu 22.04 box
# with an Ampere+ NVIDIA GPU and CUDA 12.1 preinstalled).
#
# Creates /workspace/.venv, installs torch cu121 + the project deps, persists
# the HuggingFace cache on the /workspace volume so weights survive pod
# restarts, clones every vendor repo the pipeline imports, downloads the
# Apple DepthPro checkpoint, and ends with an A100 SDPA sanity check.
#
# Safe to re-run: every install / clone / download step is idempotent.

set -euo pipefail

# ----------------------------------------------------------------------------
# 0. Preamble
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() { printf '[setup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "Repo root: ${REPO_ROOT}"
log "Pod info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    log "WARNING: nvidia-smi failed - are you actually on a GPU pod?"
}

# ----------------------------------------------------------------------------
# 1. Persistent HuggingFace cache on the /workspace volume
# ----------------------------------------------------------------------------
# Without this the SEVA gated weights (~8 GB) get wiped every time the pod
# stops, and re-downloading them eats a chunk of your A100 rental.

HF_HOME_DIR="/workspace/.hf"
mkdir -p "${HF_HOME_DIR}/hub"

persist_export() {
    local line="$1"
    if ! grep -Fqx "${line}" "${HOME}/.bashrc" 2>/dev/null; then
        printf '%s\n' "${line}" >> "${HOME}/.bashrc"
        log "persisted: ${line}"
    fi
}
persist_export 'export HF_HOME=/workspace/.hf'
persist_export 'export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub'

export HF_HOME=/workspace/.hf
export HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub

# ----------------------------------------------------------------------------
# 2. apt system tools (jq + git + wget + curl + tmux + FFmpeg dev libs)
# ----------------------------------------------------------------------------
# jq               - required by scripts/lib_common.sh to parse FastAPI /status JSON
# git              - vendor repo clones
# wget             - DepthPro / DUSt3R checkpoint downloads
# curl             - generic fallback + uv installer
# tmux             - optional, for detaching the sweep from a flaky SSH session
# pkg-config       - required for PyAV source build (no cp311 wheel for av==11)
# libav*-dev       - FFmpeg dev headers PyAV compiles against
# fuser (psmisc)   - used by the preflight to free port 9876 from stale uvicorns
log "Installing system tools (jq, git, wget, curl, tmux, pkg-config, FFmpeg dev libs) via apt ..."
APT_PKGS=(
    jq git wget curl tmux
    pkg-config psmisc
    libavformat-dev libavcodec-dev libavdevice-dev
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
)
if command -v apt-get >/dev/null 2>&1; then
    NEED_APT=()
    # Binary tools
    for tool in jq git wget curl tmux pkg-config; do
        command -v "${tool}" >/dev/null 2>&1 || NEED_APT+=("${tool}")
    done
    # psmisc (for fuser): check the binary, install psmisc
    command -v fuser >/dev/null 2>&1 || NEED_APT+=("psmisc")
    # FFmpeg dev headers: probe via pkg-config once pkg-config is present
    # (do a conservative install regardless if any header file is missing)
    for pkg in libavformat-dev libavcodec-dev libavdevice-dev \
               libavutil-dev libswscale-dev libswresample-dev libavfilter-dev; do
        dpkg -s "${pkg}" >/dev/null 2>&1 || NEED_APT+=("${pkg}")
    done
    if [ "${#NEED_APT[@]}" -gt 0 ]; then
        log "  missing: ${NEED_APT[*]} - running apt-get install ..."
        DEBIAN_FRONTEND=noninteractive apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${NEED_APT[@]}"
    else
        log "  all apt tools already present."
    fi
else
    log "  WARNING: apt-get not available - ensure all apt tools are installed some other way."
    log "           Required: ${APT_PKGS[*]}"
fi

# ----------------------------------------------------------------------------
# 3. uv (fast Python package manager)
# ----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# The installer drops an env script here; source it so `uv` is on PATH in this
# shell even on a fresh install.
# shellcheck disable=SC1091
[ -f "${HOME}/.local/bin/env" ] && source "${HOME}/.local/bin/env"
export PATH="${HOME}/.local/bin:${PATH}"
log "uv version: $(uv --version)"

# ----------------------------------------------------------------------------
# 4. Persistent venv on /workspace
# ----------------------------------------------------------------------------
VENV_DIR="/workspace/.venv"
if [ ! -d "${VENV_DIR}" ]; then
    log "Creating venv at ${VENV_DIR} (python 3.11) ..."
    uv venv "${VENV_DIR}" --python 3.11
else
    log "Reusing existing venv at ${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "python: $(python --version) | $(which python)"

# ----------------------------------------------------------------------------
# 5. Vendor repos (cloned into backend/vendor/)
# ----------------------------------------------------------------------------
# The pipeline adds these directories to sys.path at runtime and imports
# them directly (seva, viewcrafter, dust3r, etc.), so they must be on disk.
# Cloning early so later steps (DUSt3R requirements.txt install, VC-side
# DUSt3R .pth generation) can rely on them being present.
VENDOR="backend/vendor"
mkdir -p "${VENDOR}"

clone_repo() {
    # clone_repo <name> <url> [git-clone-flags...]
    local name="$1" url="$2"; shift 2
    local dest="${VENDOR}/${name}"
    if [ -d "${dest}/.git" ]; then
        log "  ${name}: already cloned, skipping."
        return 0
    fi
    if [ -d "${dest}" ] && [ -n "$(ls -A "${dest}" 2>/dev/null)" ]; then
        log "  ${name}: directory exists but is not a git repo - leaving as-is."
        return 0
    fi
    log "  Cloning ${name} from ${url} ..."
    rm -rf "${dest}"
    git clone "$@" "${url}" "${dest}"
}

log "Cloning vendor repositories ..."
clone_repo "dust3r"                "https://github.com/naver/dust3r.git"
if [ -d "${VENDOR}/dust3r/.git" ]; then
    ( cd "${VENDOR}/dust3r" && git submodule update --init --recursive ) || true
fi
clone_repo "ViewCrafter"           "https://github.com/Drexubery/ViewCrafter.git"          --recursive
clone_repo "stable-virtual-camera" "https://github.com/Stability-AI/stable-virtual-camera.git" --recursive

# DepthPro is Apple's repo; we use it here just as a stable location for the
# pretrained checkpoint (backend/vendor/DepthPro/checkpoints/depth_pro.pt).
# The actual Python package (depth_pro) is pip-installed in section 9 below.
mkdir -p "${VENDOR}/DepthPro/checkpoints"

# ----------------------------------------------------------------------------
# 6. PyTorch for A100 (Ampere / sm_80)
# ----------------------------------------------------------------------------
# cu121 wheels cover A100 natively; do NOT use cu128/Blackwell wheels here.
# Pinning to torch 2.4.1 matches the torch version the flash-attn prebuilt
# wheel index expects, which saves a from-source compile.
log "Installing torch 2.4.1 + torchvision 0.19.1 (CUDA 12.1) ..."
uv pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------------------------------------------------------------------
# 7. Project requirements (strict pin set)
# ----------------------------------------------------------------------------
# backend/requirements.txt carries the "VC compatibility pin set"
# (pillow<10, numpy<2, open-clip-torch==2.20.0, av==11.0.0,
# pytorch-lightning<2, moviepy<2, diffusers==0.30.0).  Each of those
# strict pins exists because the looser version drifted forward and broke
# a specific code path deep inside ViewCrafter or DUSt3R.  Do NOT loosen
# them here - if you bump one, bump it in backend/requirements.txt and
# run the full sweep end-to-end before merging.
#
# ---------- The CUDA-13 incident, and how this step prevents it ----------
# The first pass of this script did:
#     uv pip install --force-reinstall -r backend/requirements.txt
# That is DANGEROUS.  uv's resolver re-evaluates the FULL dep graph with
# --force-reinstall: if any downstream package (e.g. open-clip-torch)
# lists `torch>=1.9` as a dep, uv is free to pick the newest torch from
# PyPI, which pulls nvidia-cuda-runtime==13.x.  A cu13 torch wheel
# cannot `torch.cuda.init()` on a RunPod pod whose driver is CUDA 12.x,
# and the whole stack crashes at first model load.  We have already hit
# this exact failure once.
#
# Defenses below:
#   1. We write an explicit uv constraints file pinning torch/torchvision
#      to the cu121 versions installed in §6.  uv honours --constraints
#      during dependency resolution, so even if a package says
#      `torch>=1.9`, uv will still pick 2.4.1.
#   2. The defensive reassert at the end uses --no-deps so a single-
#      package install can NEVER re-resolve torch to latest.
CONSTRAINTS_FILE="/tmp/image_explorer_constraints.txt"
cat > "${CONSTRAINTS_FILE}" <<'EOF'
# Managed by setup_runpod.sh - do not edit by hand.
# Pins torch/torchvision/CUDA-runtime stubs so no resolver transaction
# can silently upgrade them to CUDA 13 and break the A100 pod.
torch==2.4.1
torchvision==0.19.1
# Block the PyPI CUDA-13 stub packages entirely.  If uv tries to pull
# any of these, force it to an impossible version so the install fails
# loudly instead of silently.
nvidia-cuda-runtime-cu13<0
nvidia-cudnn-cu13<0
nvidia-cublas-cu13<0
nvidia-cusparselt-cu13<0
nvidia-nccl-cu13<0
EOF
log "Wrote uv constraints file: ${CONSTRAINTS_FILE}"

log "Installing backend/requirements.txt (strict pin set, with torch constraints) ..."
uv pip install \
    --constraints "${CONSTRAINTS_FILE}" \
    --force-reinstall \
    -r backend/requirements.txt

# DUSt3R has its own requirements.txt - its __init__.py transitively imports
# some of these on first use.  NOTE: dust3r/requirements.txt can relax pins
# we just made strict (e.g. opencv-python, numpy).  Install WITHOUT
# force-reinstall so it can't stomp the VC compat pin set; any genuinely
# missing dep will still be picked up.  Constraints file still applies
# so torch can't drift here either.
if [ -f backend/vendor/dust3r/requirements.txt ]; then
    log "Installing backend/vendor/dust3r/requirements.txt (constrained, no force-reinstall) ..."
    uv pip install \
        --constraints "${CONSTRAINTS_FILE}" \
        -r backend/vendor/dust3r/requirements.txt || \
        log "  WARN: dust3r requirements install returned non-zero; continuing."
fi

# Re-assert the VC compatibility pin set after dust3r's requirements.txt,
# in case any transitive dep (e.g. scipy, opencv) pulled a NumPy 2.x
# upgrade back in.  --no-deps is CRITICAL here: it tells uv "install
# exactly these versions, do not touch any other package".  Without it,
# `uv pip install --force-reinstall open-clip-torch==2.20.0` is free to
# re-resolve torch to latest -> CUDA 13 -> broken pod.
log "Re-asserting VC compatibility pin set (--no-deps, defensive) ..."
uv pip install --no-deps --force-reinstall \
    "diffusers==0.30.0" \
    "open-clip-torch==2.20.0" \
    "pytorch-lightning<2.0" \
    "av==11.0.0" \
    "moviepy<2.0" \
    "Pillow<10" \
    "numpy<2"

# Sanity: after all installs, torch must still be the cu121 build.
log "Verifying torch is still on cu121 after requirements install ..."
python - <<'PY' || { echo "ERROR: torch is not on cu121 after requirements install; aborting."; exit 5; }
import torch, sys
assert "+cu12" in torch.__version__ or torch.version.cuda.startswith("12."), \
    f"torch={torch.__version__} / cuda={torch.version.cuda} - expected cu12x"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"  OK  torch {torch.__version__} cuda={torch.version.cuda}")
PY

# ----------------------------------------------------------------------------
# 8. Apple DepthPro (depth_pro Python package + 1.5 GB checkpoint)
# ----------------------------------------------------------------------------
# backend/depth_pro_estimator.py does `import depth_pro` at module import
# time - so uvicorn cannot even start unless this package is installed.
log "Installing depth_pro Python package from apple/ml-depth-pro ..."
if python -c "import depth_pro" 2>/dev/null; then
    log "  depth_pro already importable, skipping."
else
    uv pip install "git+https://github.com/apple/ml-depth-pro.git"
fi

# The checkpoint lives on Apple's CDN, not HuggingFace, so `hf download`
# can't help here.  1.5 GB; takes ~10-60s on a decent pod network.
DEPTHPRO_CKPT="${VENDOR}/DepthPro/checkpoints/depth_pro.pt"
if [ -s "${DEPTHPRO_CKPT}" ] && [ "$(stat -c%s "${DEPTHPRO_CKPT}")" -gt 1000000000 ]; then
    log "  DepthPro checkpoint already present ($(du -h "${DEPTHPRO_CKPT}" | cut -f1))."
else
    log "  Downloading DepthPro checkpoint (~1.5 GB) ..."
    rm -f "${DEPTHPRO_CKPT}"
    wget --tries=3 --retry-connrefused --progress=dot:giga \
        -O "${DEPTHPRO_CKPT}" \
        "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    log "  DepthPro checkpoint downloaded: $(du -h "${DEPTHPRO_CKPT}" | cut -f1)"
fi

# ----------------------------------------------------------------------------
# 9. ViewCrafter-side DUSt3R .pth checkpoint (REAL Naver Labs training file)
# ----------------------------------------------------------------------------
# ViewCrafter's bundled `extern/dust3r/dust3r/inference.py::load_model`
# expects the ORIGINAL DUSt3R training checkpoint format, which is a dict:
#
#     {"args": Namespace(model="AsymmetricCroCo3DStereo(...)", ...),
#      "model": state_dict, "epoch": int, ...}
#
# The `args` field holds the string representation of the model
# constructor; load_model parses it at load time to pick the architecture.
# A raw state_dict saved from `AsymmetricCroCo3DStereo.from_pretrained`
# is NOT a substitute - it lacks `args`, and VC crashes with
# `KeyError: 'args'` the first time a ViewCrafter cell fires.
#
# The only public source of the properly-wrapped .pth is Naver Labs' CDN.
# We validate the existing file's shape before skipping to avoid carrying
# over a stale raw-state-dict version from an earlier setup run.
VC_CKPT_DIR="${VENDOR}/ViewCrafter/checkpoints"
VC_DUST3R_CKPT="${VC_CKPT_DIR}/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
VC_DUST3R_URL="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
mkdir -p "${VC_CKPT_DIR}"

vc_dust3r_is_valid() {
    # True iff the file has the training-checkpoint wrapper dict with
    # an 'args' key (what ViewCrafter's load_model actually reads).
    [ -s "${VC_DUST3R_CKPT}" ] || return 1
    python - <<PY 2>/dev/null
import sys, torch
try:
    c = torch.load("${VC_DUST3R_CKPT}", map_location="cpu", weights_only=False)
    sys.exit(0 if isinstance(c, dict) and "args" in c else 1)
except Exception:
    sys.exit(1)
PY
}

if vc_dust3r_is_valid; then
    log "  VC-side DUSt3R .pth already present with correct 'args' wrapper; skipping."
else
    if [ -s "${VC_DUST3R_CKPT}" ]; then
        log "  WARN: existing VC-side DUSt3R .pth is in the wrong format (no 'args' key) - removing and re-downloading."
        rm -f "${VC_DUST3R_CKPT}"
    fi
    log "  Downloading VC-side DUSt3R .pth from Naver Labs (~2.3 GB) ..."
    wget --tries=3 --retry-connrefused --progress=dot:giga \
        -O "${VC_DUST3R_CKPT}" "${VC_DUST3R_URL}"
    log "  VC-side DUSt3R .pth downloaded: $(du -h "${VC_DUST3R_CKPT}" | cut -f1)"
    # Validate what we just downloaded so we catch partial writes / 404s.
    if ! vc_dust3r_is_valid; then
        log "  ERROR: downloaded VC DUSt3R .pth failed format validation (no 'args' key)."
        exit 4
    fi
fi

# ----------------------------------------------------------------------------
# 9b. pytorch3d (built from source against the installed torch/CUDA)
# ----------------------------------------------------------------------------
# ViewCrafter's top-level `import pytorch3d` means uvicorn cannot even
# lazy-load the synthesizer without this.  We MUST build from source:
#   - The PyPI wheels are CPU-only.
#   - The miropsota torch_packages_builder wheels are built against
#     cu130 (libcudart.so.13), which is incompatible with our
#     torch 2.4.1+cu121 install (libcudart.so.12).
# Source build against torch 2.4.1+cu121 on an A100 takes ~4-6 min.
# We pin to v0.7.9 (the newest tag with our prefixed casing) and install
# into the venv with `uv pip install --no-build-isolation` so the
# compile sees the already-installed torch.
log "Installing pytorch3d v0.7.9 from source (this takes several minutes) ..."
if python -c "import pytorch3d; from pytorch3d import _C" 2>/dev/null; then
    log "  pytorch3d already importable (native _C loads); skipping rebuild."
else
    # Tell the build where CUDA lives and to target sm_80 only (A100).
    # MAX_JOBS caps parallel nvcc processes; without it, peak RAM on a
    # small pod can OOM.  FORCE_CUDA=1 stops pytorch3d from falling back
    # to a CPU build if it can't auto-detect CUDA.
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    uv pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
    # Verify the native extension actually loads against our CUDA runtime.
    python - <<'PY'
import pytorch3d
from pytorch3d import _C
from pytorch3d.renderer import PerspectiveCameras
_ = PerspectiveCameras(device="cuda")
print(f"  pytorch3d {pytorch3d.__version__} _C + CUDA ctor OK")
PY
fi

# ----------------------------------------------------------------------------
# 10. flash-attn (optional; SDPA fallback covers us if this fails)
# ----------------------------------------------------------------------------
# Prebuilt wheels for A100 + torch 2.4 are on PyPI; --no-build-isolation is
# required because flash-attn's build backend needs torch already installed
# to query its ABI.  If the wheel resolution fails, we log and continue: the
# SDPA_BACKENDS fallback in seva/modules/transformer.py still picks FLASH on
# A100 via PyTorch's built-in scaled_dot_product_attention.
log "Attempting flash-attn install ..."
if uv pip install flash-attn --no-build-isolation; then
    log "flash-attn installed."
else
    log "flash-attn install FAILED (non-fatal) - SDPA fallback in SEVA will use torch's built-in FLASH path."
fi

# ----------------------------------------------------------------------------
# 11. Reminder for HuggingFace login (interactive, can't automate)
# ----------------------------------------------------------------------------
cat <<'EOF'

---------------------------------------------------------------
  Next step (manual, one-time):

    source /workspace/.venv/bin/activate
    hf auth login     # or: huggingface-cli login

  Paste a HF token with access to:
    stabilityai/stable-virtual-camera
  (Accept the license in the browser first if you have not.)
---------------------------------------------------------------

EOF

# ----------------------------------------------------------------------------
# 12. A100 SDPA sanity check
# ----------------------------------------------------------------------------
# Replicates the user's Step 9 check.  If FLASH prints OK, the sweep will hit
# the fast path; if only MATH is OK, you're not actually on an A100 (or CUDA
# is misconfigured) and should stop before burning rental hours.
log "Running A100 SDPA sanity check ..."
python - <<'PY'
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

print("torch:   ", torch.__version__)
print("cuda:    ", torch.version.cuda)
print("device:  ", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
print("cap:     ", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")

if not torch.cuda.is_available():
    raise SystemExit("CUDA not available - cannot proceed.")

q = torch.randn(1, 16, 4096, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn_like(q); v = torch.randn_like(q)
any_ok = False
for name, be in [
    ("FLASH",     SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN",     SDPBackend.CUDNN_ATTENTION),
    ("MATH",      SDPBackend.MATH),
]:
    try:
        with sdpa_kernel(be):
            _ = F.scaled_dot_product_attention(q, k, v)
        print(f"{name:9} OK")
        any_ok = True
    except Exception as e:
        print(f"{name:9} FAIL {str(e)[:60]}")

if not any_ok:
    raise SystemExit("No SDPA backend works - something is very wrong with the CUDA setup.")
PY

# ----------------------------------------------------------------------------
# 13. Final dependency self-check
# ----------------------------------------------------------------------------
# Import-probe every package that earlier runs failed on, so setup can fail
# loudly here rather than 4 minutes into the first uvicorn startup.
log "Final import self-check (the exact imports that failed in earlier runs) ..."
python - <<'PY'
import sys, os

# Vendor dirs need to be on sys.path the same way the runtime code puts them
# there (seva_synthesizer._ensure_vendor_on_path, viewcrafter_synthesizer
# generate_views, dust3r_reconstructor module-level).
sys.path.insert(0, os.path.abspath("backend/vendor/stable-virtual-camera"))
sys.path.insert(0, os.path.abspath("backend/vendor/ViewCrafter"))
sys.path.insert(0, os.path.abspath("backend/vendor/dust3r"))

errs = []
def chk(label, stmt):
    try:
        exec(stmt, {})
        print(f"  OK    {label}")
    except Exception as e:
        print(f"  FAIL  {label}: {type(e).__name__}: {str(e)[:220]}")
        errs.append(label)

# Cheap standalone packages first
chk("colorama",                       "import colorama")
chk("splines",                        "import splines")
chk("tyro",                           "import tyro")
chk("fire",                           "import fire")
chk("open_clip",                      "import open_clip")
chk("lpips",                          "import lpips")
chk("depth_pro",                      "import depth_pro")

# The actual failing imports from earlier server.log tracebacks
chk("diffusers.models.AutoencoderKL", "from diffusers.models import AutoencoderKL")
chk("seva.modules.autoencoder",       "from seva.modules.autoencoder import AutoEncoder")
chk("seva.eval",                      "from seva.eval import IS_TORCH_NIGHTLY")
chk("viewcrafter.ViewCrafter",        "from viewcrafter import ViewCrafter")
chk("dust3r.model",                   "from dust3r.model import AsymmetricCroCo3DStereo")

# pytorch3d native extension (CUDA runtime mismatch regression)
chk("pytorch3d._C",                   "import pytorch3d; from pytorch3d import _C")

# ----- VC compatibility pin set: probe actual API-breakage surfaces -----
# Each block below asserts that the specific API ViewCrafter actually
# *uses* still exists at the installed version.  Pure "import X" would
# pass even with a silently-upgraded package that drops an attribute.

# Pillow: Image.ANTIALIAS was removed in 10.0.  DUSt3R uses it.
chk("Pillow.ANTIALIAS",
    "from PIL import Image; assert hasattr(Image, 'ANTIALIAS'), 'Pillow>=10 removed ANTIALIAS'")

# PyAV: frame.pict_type='NONE' (string) was rejected in av>=12.
# torchvision.io.write_video still does this; VC's save_video calls it.
chk("av.pict_type str assign",
    "import av, numpy as np;"
    "f = av.VideoFrame.from_ndarray(np.zeros((8,8,3), dtype='uint8'), format='rgb24');"
    "f.pict_type = 'NONE'")

# open_clip: ViewCrafter's lvdm/condition.py reads visual.input_patchnorm.
chk("open_clip.input_patchnorm attr",
    "import open_clip;"
    "m,_,_ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=None);"
    "assert hasattr(m.visual, 'input_patchnorm'), 'open_clip>=2.24 removed input_patchnorm'")

# pytorch_lightning: VC inherits LightningModule from 1.x; 2.x breaks load.
chk("pytorch_lightning<2",
    "import pytorch_lightning as pl;"
    "assert pl.__version__.split('.')[0] == '1', f'need pytorch_lightning 1.x, got {pl.__version__}'")

# numpy: VC/DUSt3R still reference np.bool / np.int / np.float.
chk("numpy<2",
    "import numpy as np;"
    "assert np.__version__.startswith('1.'), f'need numpy 1.x, got {np.__version__}'")

# moviepy: VC uses 1.x API (moviepy.editor.ImageSequenceClip).
chk("moviepy<2",
    "import moviepy; "
    "assert moviepy.__version__.startswith('1.'), f'need moviepy 1.x, got {moviepy.__version__}'")

# diffusers custom_op registration regression (torch 2.4 infer_schema).
chk("diffusers==0.30.*",
    "import diffusers;"
    "assert diffusers.__version__.startswith('0.30.'), f'need diffusers 0.30.x, got {diffusers.__version__}'")

# DUSt3R training-checkpoint format check: ViewCrafter's load_model
# expects a dict with an 'args' key.  A raw state_dict (saved from
# `AsymmetricCroCo3DStereo.from_pretrained`) is NOT accepted.
chk("VC DUSt3R .pth has 'args' wrapper",
    "import torch, os;"
    "p = 'backend/vendor/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth';"
    "assert os.path.exists(p), 'VC DUSt3R .pth missing';"
    "c = torch.load(p, map_location='cpu', weights_only=False);"
    "assert isinstance(c, dict) and 'args' in c, 'VC DUSt3R .pth has no args key (wrong format)'")

if errs:
    print("\nSELF-CHECK FAILED - the above imports need fixing before run_all.sh will work:")
    for e in errs: print(f"  - {e}")
    raise SystemExit(2)
print("\nAll self-check imports OK.")
PY

log ""
log "Setup complete.  Activate the venv in future shells with:"
log "    source /workspace/.venv/bin/activate"
log ""
log "Then run the full sweep with:"
log "    tmux new -s sweep"
log "    cd ${REPO_ROOT} && bash scripts/run_all.sh"
