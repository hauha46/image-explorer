# Linux Setup Guide

This guide walks through setting up image-explorer on a native Linux machine. All models including SEVA are supported.

---

## Prerequisites

- **Ubuntu 22.04/24.04** (or similar Debian-based distro; other distros should work with minor adjustments)
- **NVIDIA GPU** with drivers that support CUDA 12.8+
- **CUDA Toolkit 12.8** installed (provides `nvcc` -- needed only if the flash-attn prebuilt wheel is unavailable)
- **Git**
- ~30 GB free disk space (vendor repos + model weights + Python environment)

---

## Step 1: Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

---

## Step 2: (Optional) Install CUDA Toolkit

If you don't already have `nvcc` available, install the CUDA toolkit. This is only needed as a fallback if the flash-attn prebuilt wheel doesn't match your environment.

```bash
# Ubuntu 24.04
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

Add to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

Then `source ~/.bashrc`.

---

## Step 3: Clone the repository

```bash
git clone https://github.com/hauha46/image-explorer.git
cd image-explorer
```

---

## Step 4: Run the setup script

```bash
bash setup.sh
```

This takes 10-20 minutes and handles everything:

1. **Clones 8 vendor repositories** into `backend/vendor/`:
   - `dust3r` -- 3D reconstruction (from Naver)
   - `ml-depth-pro` -- depth estimation Python package (from Apple)
   - `DepthPro` -- depth estimation checkpoint location (from Apple)
   - `ViewCrafter` -- novel view synthesis via video diffusion
   - `ml-vivid` -- novel view synthesis (from Apple)
   - `PanoDreamer` -- panoramic novel view synthesis
   - `sv3d-diffusers` -- SV3D model wrapper
   - `stable-virtual-camera` -- SEVA model (from Stability AI)

2. **Downloads the DepthPro checkpoint** (~1.5 GB from Apple CDN)

3. **Installs Python dependencies** via `uv sync` (PyTorch with CUDA 12.8)

4. **Installs flash-attn** for SEVA support:
   - First tries a prebuilt wheel matching your PyTorch + Python version
   - Falls back to compiling from source if no prebuilt wheel is available

5. **Installs additional SEVA dependencies** (open-clip-torch, viser, tyro, etc.)

6. **Downloads the DUSt3R checkpoint** for ViewCrafter (~2.3 GB from HuggingFace)

---

## Step 5: HuggingFace authentication

Some models (SVD, SEVA) are gated and require a HuggingFace account.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the license for gated models:
   - [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
   - [stabilityai/stable-virtual-camera](https://huggingface.co/stabilityai/stable-virtual-camera)
4. Log in:

```bash
uv run python -c "from huggingface_hub import login; login()"
# Paste your token when prompted (input will be hidden)
```

---

## Step 6: Start the server

```bash
cd backend
uv run --project .. uvicorn app:app --host 0.0.0.0 --port 9876
```

Open your browser and go to:

```
http://localhost:9876/app
```

---

## Available Models

| Model | Type | Notes |
|-------|------|-------|
| SVD | Video Diffusion | Fast, general purpose. Weights auto-download on first use. |
| ViewCrafter | Video Diffusion | Supports text prompts. Weights auto-download. |
| VIVID | Video Diffusion | Apple model. Weights download from Apple CDN on first use. |
| PanoDreamer | Panorama Diffusion | Supports text prompts. SD2 weights auto-download. |
| Zero123++ | Multi-view Diffusion | Weights auto-download on first use. |
| SV3D | 3D Video Diffusion | Weights auto-download on first use. |
| SEVA | Stable Virtual Camera | Requires flash-attn. Gated HF model -- needs HF login. |

---

## Quick Start (TL;DR)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone and setup
git clone https://github.com/hauha46/image-explorer.git
cd image-explorer
bash setup.sh

# Login to HuggingFace (for gated models)
uv run python -c "from huggingface_hub import login; login()"

# Run
cd backend
uv run --project .. uvicorn app:app --host 0.0.0.0 --port 9876
# Open http://localhost:9876/app
```

---

## Troubleshooting

**"Warning, cannot find cuda-compiled version of RoPE2D"**
Harmless warning from DUSt3R. Falls back to a slower PyTorch implementation. No impact on output quality.

**flash-attn compilation fails / WSL crashes during install**
The source compilation is very memory-intensive and will crash WSL if it runs out of memory. Always install from the prebuilt wheel instead. For PyTorch 2.11 + Python 3.11 + CUDA 12:
```bash
uv pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```
For other PyTorch versions, find the matching wheel at [github.com/lesj0610/flash-attention/releases](https://github.com/lesj0610/flash-attention/releases).

**ModuleNotFoundError: No module named 'scene_processor'**
You're running uvicorn from the wrong directory. Run from `backend/`:
```bash
cd backend
uv run --project .. uvicorn app:app --host 0.0.0.0 --port 9876
```

**CUDA out of memory**
Some models (ViewCrafter, SEVA, PanoDreamer) are VRAM-intensive. The pipeline automatically offloads other models to CPU during inference, but you still need ~8-12 GB VRAM for the largest models.

**First run is very slow**
Model weights for SVD, SV3D, Zero123++, VIVID, and ViewCrafter are downloaded from HuggingFace/Apple CDN on first use. Subsequent runs use cached weights and start much faster.
