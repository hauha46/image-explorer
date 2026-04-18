# Windows Setup Guide (WSL)

This guide walks through setting up image-explorer on Windows using WSL2 (Windows Subsystem for Linux). WSL is required to run **all** models including SEVA, which depends on flash-attn (Linux only).

> If you only need models other than SEVA, you can skip WSL and run directly on Windows using `setup.ps1`. See the note at the bottom.

---

## Prerequisites

- **Windows 10/11** with administrator access
- **NVIDIA GPU** with drivers that support CUDA 12.8+
- **Git** installed and on PATH
- ~30 GB free disk space (vendor repos + model weights + Python environment)

---

## Step 1: Install WSL2 with Ubuntu

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart your computer if prompted. After reboot, Ubuntu will launch and ask you to create a username/password.

Verify WSL2 is working and can see your GPU:

```powershell
wsl -d Ubuntu-24.04 -- nvidia-smi
```

You should see your GPU listed. If not, update your NVIDIA drivers.

---

## Step 2: Install uv (Python package manager)

Open WSL (run `wsl -d Ubuntu-24.04` from PowerShell, or launch Ubuntu from Start Menu):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

---

## Step 3: Install CUDA Toolkit

The NVIDIA driver passes through from Windows, but the CUDA development tools (nvcc) need to be installed inside WSL. This is needed if the flash-attn prebuilt wheel is unavailable and it falls back to compiling from source.

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq
sudo apt-get install -y cuda-toolkit-12-8
```

Add to your shell profile (`~/.bashrc`):

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version
```

---

## Step 4: Clone the repository

Clone to the **Linux filesystem** (not `/mnt/c/...`). The Linux filesystem is significantly faster for Python and git operations.

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/hauha46/image-explorer.git
cd image-explorer
```

---

## Step 5: Run the setup script

The `setup.sh` script handles everything automatically:

```bash
bash setup.sh
```

This takes 10-20 minutes and does the following:

1. Clones 8 vendor repositories (DUSt3R, DepthPro, ViewCrafter, VIVID, PanoDreamer, SV3D, SEVA)
2. Downloads the DepthPro checkpoint (~1.5 GB)
3. Installs all Python dependencies via `uv sync`
4. Installs flash-attn (tries prebuilt wheel first, falls back to source compilation)
5. Installs additional SEVA dependencies (open-clip-torch, etc.)
6. Downloads the DUSt3R checkpoint for ViewCrafter

---

## Step 6: HuggingFace authentication

Some models (SVD, SEVA) are gated and require a HuggingFace account.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token
3. Accept the license for gated models:
   - [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
   - [stabilityai/stable-virtual-camera](https://huggingface.co/stabilityai/stable-virtual-camera)
4. Log in:

```bash
uv run huggingface-cli login
# Paste your token when prompted
```

---

## Step 7: Start the server

```bash
cd ~/projects/image-explorer/backend
uv run --project ~/projects/image-explorer uvicorn app:app --host 0.0.0.0 --port 9876
```

Open your **Windows browser** and go to:

```
http://localhost:9876/app
```

WSL2 automatically forwards ports, so `localhost` works from Windows.

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
| SEVA | Stable Virtual Camera | Requires flash-attn (Linux/WSL only). Gated HF model. |

---

## Troubleshooting

**"Warning, cannot find cuda-compiled version of RoPE2D"**
This is harmless. DUSt3R falls back to a slower PyTorch implementation. Output quality is identical.

**Server won't start / module not found errors**
Make sure you run from the `backend/` directory:
```bash
cd ~/projects/image-explorer/backend
uv run --project ~/projects/image-explorer uvicorn app:app --host 0.0.0.0 --port 9876
```

**flash-attn compilation fails (OOM)**
The source compilation is very memory-intensive. The setup script tries a prebuilt wheel first. If that fails and source compilation also fails, increase WSL memory in `%USERPROFILE%\.wslconfig`:
```ini
[wsl2]
memory=16GB
swap=8GB
```
Then restart WSL: `wsl --shutdown`

**Port 9876 already in use**
Stop any existing server or pick a different port:
```bash
uv run --project ~/projects/image-explorer uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Windows-Only Mode (without WSL)

If you don't need SEVA and want to run directly on Windows:

```powershell
cd image-explorer
.\setup.ps1
cd backend
uv run uvicorn app:app --host 0.0.0.0 --port 9876
```

All models except SEVA will work. SEVA will appear in the GUI dropdown but will fail at inference time.
