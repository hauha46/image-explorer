# setup.ps1 — One-time setup for image-explorer (Windows PowerShell)
# Run from the repo root:  .\setup.ps1

$ErrorActionPreference = "Stop"
$VENDOR = "backend\vendor"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  image-explorer setup (Windows)            " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ── Prerequisites check ──────────────────────────────────────────────
Write-Host "`n[prereq] Checking for required tools..." -ForegroundColor Yellow

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: git is not installed or not on PATH." -ForegroundColor Red; exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: uv is not installed. Install from https://docs.astral.sh/uv/" -ForegroundColor Red; exit 1
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host "  GPU detected:" -NoNewline; & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
} else {
    Write-Host "  WARNING: nvidia-smi not found. An NVIDIA GPU with CUDA 12.8+ drivers is required." -ForegroundColor Yellow
}

# ── Step 1: Clone vendor repositories ────────────────────────────────
Write-Host "`n[1/5] Cloning vendor repositories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $VENDOR | Out-Null

$repos = @(
    @{ Name = "dust3r";                  Url = "https://github.com/naver/dust3r.git";                          Recursive = $false; Submodules = $true  },
    @{ Name = "ml-depth-pro";            Url = "https://github.com/apple/ml-depth-pro.git";                    Recursive = $false; Submodules = $false },
    @{ Name = "DepthPro";                Url = "https://github.com/apple/ml-depth-pro.git";                    Recursive = $false; Submodules = $false },
    @{ Name = "ViewCrafter";             Url = "https://github.com/Drexubery/ViewCrafter.git";                 Recursive = $true;  Submodules = $false },
    @{ Name = "ml-vivid";                Url = "https://github.com/apple/ml-vivid.git";                        Recursive = $false; Submodules = $false },
    @{ Name = "PanoDreamer";             Url = "https://github.com/avinashpaliwal/PanoDreamer.git";            Recursive = $false; Submodules = $false },
    @{ Name = "sv3d-diffusers";          Url = "https://github.com/chenguolin/sv3d-diffusers.git";             Recursive = $false; Submodules = $false },
    @{ Name = "stable-virtual-camera";   Url = "https://github.com/Stability-AI/stable-virtual-camera.git";   Recursive = $true;  Submodules = $false }
)

foreach ($repo in $repos) {
    $dest = Join-Path $VENDOR $repo.Name
    if (Test-Path $dest) {
        Write-Host "  $($repo.Name) — already exists, skipping."
        continue
    }
    Write-Host "  Cloning $($repo.Name)..."
    if ($repo.Recursive) {
        git clone --recursive $repo.Url $dest
    } else {
        git clone $repo.Url $dest
    }
    if ($repo.Submodules) {
        Push-Location $dest
        git submodule update --init --recursive
        Pop-Location
    }
}

# ── Step 2: Download model checkpoints ───────────────────────────────
Write-Host "`n[2/5] Downloading model checkpoints..." -ForegroundColor Cyan

# DepthPro checkpoint (~1.5 GB)
$depthProDir = Join-Path $VENDOR "DepthPro\checkpoints"
$depthProCkpt = Join-Path $depthProDir "depth_pro.pt"
if (-not (Test-Path $depthProCkpt)) {
    Write-Host "  Downloading DepthPro checkpoint (~1.5 GB) — this may take a few minutes..."
    New-Item -ItemType Directory -Force -Path $depthProDir | Out-Null
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" -OutFile $depthProCkpt
    $ProgressPreference = 'Continue'
    $sizeMB = [math]::Round((Get-Item $depthProCkpt).Length / 1MB, 1)
    Write-Host "  DepthPro checkpoint downloaded (${sizeMB} MB)."
} else {
    Write-Host "  DepthPro checkpoint already exists, skipping."
}

# ── Step 3: Install Python dependencies ──────────────────────────────
Write-Host "`n[3/5] Installing Python dependencies with uv..." -ForegroundColor Cyan
uv sync

# ── Step 4: Download DUSt3R checkpoint for ViewCrafter ───────────────
Write-Host "`n[4/5] Downloading DUSt3R checkpoint for ViewCrafter..." -ForegroundColor Cyan
$vcCkptDir = Join-Path $VENDOR "ViewCrafter\checkpoints"
$dust3rCkpt = Join-Path $vcCkptDir "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if (-not (Test-Path $dust3rCkpt)) {
    New-Item -ItemType Directory -Force -Path $vcCkptDir | Out-Null
    uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt', 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', local_dir=r'$vcCkptDir')"
    Write-Host "  DUSt3R checkpoint downloaded."
} else {
    Write-Host "  DUSt3R ViewCrafter checkpoint already exists, skipping."
}

# ── Step 5: HuggingFace login reminder ───────────────────────────────
Write-Host "`n[5/5] HuggingFace authentication" -ForegroundColor Cyan
Write-Host "  Some models (SVD, SEVA) are gated and require a HuggingFace account."
Write-Host "  If you haven't already, run:" -ForegroundColor Yellow
Write-Host "    uv run huggingface-cli login" -ForegroundColor Yellow

# ── Done ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup complete!                           " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the server:" -ForegroundColor Yellow
Write-Host "  uv run uvicorn backend.app:app --host 0.0.0.0 --port 9876"
Write-Host ""
Write-Host "Then open:  http://localhost:9876/app"
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  - Other model weights (SVD, SV3D, Zero123++, VIVID, ViewCrafter) auto-download on first use"
Write-Host "  - SEVA requires Linux/WSL (flash-attn is not available on native Windows)"
Write-Host "  - First run of each model will be slow due to weight downloads"
Write-Host ""
