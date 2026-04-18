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
echo -e "\n[1/5] Cloning vendor repositories..."
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
echo -e "\n[2/5] Downloading model checkpoints..."

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
echo -e "\n[3/5] Installing Python dependencies with uv..."
uv sync

# ── Step 4: Download DUSt3R checkpoint for ViewCrafter ───────────────
echo -e "\n[4/5] Downloading DUSt3R checkpoint for ViewCrafter..."
VC_CKPT_DIR="$VENDOR/ViewCrafter/checkpoints"
DUST3R_CKPT="$VC_CKPT_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CKPT" ]; then
    mkdir -p "$VC_CKPT_DIR"
    uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt',
    'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
    local_dir='$VC_CKPT_DIR',
)
"
    echo "  DUSt3R checkpoint downloaded."
else
    echo "  DUSt3R ViewCrafter checkpoint already exists, skipping."
fi

# ── Step 5: HuggingFace login reminder ───────────────────────────────
echo -e "\n[5/5] HuggingFace authentication"
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
