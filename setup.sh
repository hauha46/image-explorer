#!/usr/bin/env bash
# setup.sh — One-time setup for image-explorer (Linux / macOS / WSL)
# Run from the repo root:  bash setup.sh
set -euo pipefail

VENDOR="backend/vendor"

echo ""
echo "============================================"
echo "  image-explorer setup (Linux/macOS/WSL)    "
echo "============================================"

# ── Prerequisites check ──────────────────────────────────────────────
echo -e "\n[prereq] Checking for required tools..."

command -v git  >/dev/null 2>&1 || { echo "ERROR: git is not installed."; exit 1; }
command -v uv   >/dev/null 2>&1 || { echo "ERROR: uv is not installed. Install from https://docs.astral.sh/uv/"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
else
    echo "  WARNING: nvidia-smi not found. An NVIDIA GPU with CUDA 12.8+ drivers is required."
fi

# ── Step 1: Clone vendor repositories ────────────────────────────────
echo -e "\n[1/6] Cloning vendor repositories..."
mkdir -p "$VENDOR"

clone_repo() {
    local name="$1" url="$2" flags="${3:-}"
    local dest="$VENDOR/$name"
    if [ -d "$dest" ]; then
        echo "  $name — already exists, skipping."
        return
    fi
    echo "  Cloning $name..."
    git clone $flags "$url" "$dest"
}

clone_repo "dust3r"                "https://github.com/naver/dust3r.git"
pushd "$VENDOR/dust3r" >/dev/null && git submodule update --init --recursive && popd >/dev/null

clone_repo "ml-depth-pro"          "https://github.com/apple/ml-depth-pro.git"
clone_repo "DepthPro"              "https://github.com/apple/ml-depth-pro.git"
clone_repo "ViewCrafter"           "https://github.com/Drexubery/ViewCrafter.git"          "--recursive"
clone_repo "ml-vivid"              "https://github.com/apple/ml-vivid.git"
clone_repo "PanoDreamer"           "https://github.com/avinashpaliwal/PanoDreamer.git"
clone_repo "sv3d-diffusers"        "https://github.com/chenguolin/sv3d-diffusers.git"
clone_repo "stable-virtual-camera" "https://github.com/Stability-AI/stable-virtual-camera.git" "--recursive"

# ── Step 2: Download model checkpoints ───────────────────────────────
echo -e "\n[2/6] Downloading model checkpoints..."

DEPTHPRO_DIR="$VENDOR/DepthPro/checkpoints"
DEPTHPRO_CKPT="$DEPTHPRO_DIR/depth_pro.pt"
if [ ! -f "$DEPTHPRO_CKPT" ]; then
    echo "  Downloading DepthPro checkpoint (~1.5 GB) — this may take a few minutes..."
    mkdir -p "$DEPTHPRO_DIR"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$DEPTHPRO_CKPT" "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$DEPTHPRO_CKPT" "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
    else
        echo "  ERROR: neither wget nor curl found. Install one and re-run." ; exit 1
    fi
    SIZE_MB=$(du -m "$DEPTHPRO_CKPT" | cut -f1)
    echo "  DepthPro checkpoint downloaded (${SIZE_MB} MB)."
else
    echo "  DepthPro checkpoint already exists, skipping."
fi

# ── Step 3: Install Python dependencies ──────────────────────────────
echo -e "\n[3/6] Installing Python dependencies with uv..."
uv sync

# ── Step 4: Install SEVA dependencies (Linux only) ───────────────────
echo -e "\n[4/6] Installing SEVA dependencies (flash-attn + open-clip-torch)..."
if [ "$(uname -s)" = "Linux" ]; then
    if uv run python -c "import flash_attn" 2>/dev/null; then
        echo "  flash-attn already installed, skipping."
    else
        TORCH_VER=$(uv run python -c "import torch; print(torch.__version__.split('+')[0])")
        PY_VER=$(uv run python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        WHEEL_URL="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch${TORCH_VER}/flash_attn-2.8.3%2Bcu12torch${TORCH_VER}cxx11abiTRUE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
        echo "  Trying prebuilt flash-attn wheel for torch ${TORCH_VER} / ${PY_VER}..."
        if uv pip install "$WHEEL_URL" 2>/dev/null; then
            echo "  flash-attn installed from prebuilt wheel."
        else
            echo "  Prebuilt wheel not available. Compiling from source (needs CUDA toolkit + several minutes)..."
            uv pip install flash-attn --no-build-isolation
            echo "  flash-attn compiled and installed."
        fi
    fi
    echo "  Installing additional SEVA dependencies..."
    uv pip install open-clip-torch viser tyro fire splines "imageio[ffmpeg]" --quiet
    echo "  SEVA dependencies installed."
else
    echo "  Skipping — flash-attn only builds on Linux. SEVA will not be available on this platform."
fi

# ── Step 5: Download DUSt3R checkpoint for ViewCrafter ───────────────
echo -e "\n[5/6] Downloading DUSt3R checkpoint for ViewCrafter..."
VC_CKPT_DIR="$VENDOR/ViewCrafter/checkpoints"
DUST3R_CKPT="$VC_CKPT_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CKPT" ]; then
    mkdir -p "$VC_CKPT_DIR"
    echo "  Downloading from HuggingFace and converting to .pth format..."
    DUST3R_VENDOR="$VENDOR/dust3r"
    uv run python -c "
import sys, torch
sys.path.insert(0, '$DUST3R_VENDOR')
from dust3r.model import AsymmetricCroCo3DStereo
model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')
torch.save({'model': model.state_dict()}, '$DUST3R_CKPT')
print('  DUSt3R checkpoint downloaded and saved as .pth')
"
else
    echo "  DUSt3R ViewCrafter checkpoint already exists, skipping."
fi

# ── Step 5: HuggingFace login reminder ───────────────────────────────
echo -e "\n[6/6] HuggingFace authentication"
echo "  Some models (SVD, SEVA) are gated and require a HuggingFace account."
echo "  If you haven't already, run:"
echo "    uv run huggingface-cli login"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!                           "
echo "============================================"
echo ""
echo "To start the server:"
echo "  uv run uvicorn backend.app:app --host 0.0.0.0 --port 9876"
echo ""
echo "Then open:  http://localhost:9876/app"
echo ""
echo "Notes:"
echo "  - Other model weights (SVD, SV3D, Zero123++, VIVID, ViewCrafter) auto-download on first use"
echo "  - SEVA requires Linux/WSL (flash-attn is not available on native Windows)"
echo "  - First run of each model will be slow due to weight downloads"
echo ""
