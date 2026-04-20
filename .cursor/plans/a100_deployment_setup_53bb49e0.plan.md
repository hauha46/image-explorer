---
name: A100 deployment setup
overview: Add an A100/RunPod deployment guide, a reproducible setup script, and minor requirements hygiene so the existing (already portable) pipeline runs cleanly on rented A100 hardware with the right flags. No code behavior changes are required; the existing SDPA fallback list and `SEVA_AUTOCAST_DTYPE` already work correctly on Ampere.
todos:
  - id: doc-runpod
    content: Write docs/SETUP_RUNPOD_A100.md with template, volume, bootstrap, HF login, smoke test, full sweep, and cost/tmux notes
    status: pending
  - id: script-bootstrap
    content: "Write scripts/setup_runpod.sh: venv + deps + HF_HOME export (idempotent)"
    status: pending
  - id: script-sweep
    content: Write scripts/run_sweep_a100.sh wrapper encapsulating the recommended --num-steps 25 --dtype bf16 invocation
    status: pending
  - id: reqs-hygiene
    content: Add kornia, imageio-ffmpeg, roma, gradio, flash-attn to backend/requirements.txt with a short CUDA compatibility comment
    status: pending
isProject: false
---

# A100 deployment setup

## Recommended GPU rental site

**RunPod Secure Cloud** — [runpod.io](https://runpod.io/)

- Tier-3/4 data centers, dedicated hardware, SLA-backed (vs. RunPod *Community* which is peer-hosted).
- A100 40 GB: $1.19/hr, A100 80 GB: $1.79/hr (on-demand).
- Persistent Network Volumes (~$0.07/GB/month) so the HuggingFace cache survives pod restarts - critical since SEVA weights are ~8 GB gated and you don't want to re-download them.
- Official PyTorch templates with CUDA preinstalled; your existing `backend/requirements.txt` installs cleanly on top.
- No Docker image surgery required; SSH + VS Code remote work out of the box.

Honorable mentions: Lambda Labs (simpler UI but availability is spotty), Paperspace (reliable, pricier), Modal (serverless, requires rewriting the sweep as a Modal app). Skip Vast.ai for anything you care about; it's a peer-to-peer marketplace, not a data center.

## Why no code changes are strictly required

The pipeline is already portable to A100. The A100 (sm_80, Ampere) is *more* conservative than your 5070 Ti (sm_120, Blackwell) - everything working locally works there, and the paths that were silently falling back to MATH on Blackwell will hit FLASH cleanly on Ampere. Concretely:

- `_SDPA_BACKENDS = [FLASH, EFFICIENT, CUDNN, MATH]` in [backend/vendor/stable-virtual-camera/seva/modules/transformer.py](backend/vendor/stable-virtual-camera/seva/modules/transformer.py) - torch picks FLASH on A100 automatically.
- `SEVA_AUTOCAST_DTYPE` env var in [backend/vendor/stable-virtual-camera/seva/eval.py](backend/vendor/stable-virtual-camera/seva/eval.py) - `bf16` works great on A100 (unlike on your 5070 Ti where it regressed to 40 s/it).
- `SevaSynthesizer(num_steps=..., dtype=...)` already exposed in [backend/synthesizers/seva_synthesizer.py](backend/synthesizers/seva_synthesizer.py) and plumbed through [backend/experiments/clip_recond_sweep.py](backend/experiments/clip_recond_sweep.py).

So "reconfigure for A100" is really "document the right flags + deployment procedure".

## What I will add

### 1. `docs/SETUP_RUNPOD_A100.md` (new)

Step-by-step:

- Pick the template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or the latest `2.5.x` equivalent). A100 is happy with CUDA 12.1/12.4; you do not need cu128.
- Create a **50 GB Network Volume** mounted at `/workspace` and keep the pod's system disk for caches only.
- One-time setup inside the pod (clone repo, venv, install):
  - `git clone` the project into `/workspace/image-explorer`
  - `uv venv /workspace/.venv && source /workspace/.venv/bin/activate`
  - `uv pip install -r backend/requirements.txt`
  - `uv pip install kornia imageio-ffmpeg roma gradio` (SEVA + DUSt3R transitives that your existing notes show are needed)
  - `uv pip install flash-attn --no-build-isolation` (prebuilt wheel available for A100 + torch 2.4/2.5)
  - `huggingface-cli login` to pull gated SEVA weights
- Environment: `export HF_HOME=/workspace/.hf` (volume-persistent cache) and `export TORCH_CUDNN_V8_API_ENABLED=1`
- Sanity SDPA microbench command (confirms FLASH engages, copied from your earlier WSL test)
- First smoke run: `--limit 2` single-prompt, single-lambda to validate end-to-end in ~10 minutes
- Full sweep command **tuned for A100** (see section 3 below)
- How to detach (`tmux`) so the pod can run unattended overnight
- Cost tracker note: stop the pod when `Sweep finished:` is printed, otherwise you keep paying

### 2. `scripts/setup_runpod.sh` (new)

Idempotent bootstrap script that does everything in section 1 except `huggingface-cli login` (that's interactive). Runs on a fresh RunPod pod and leaves it ready to launch the sweep. Uses `HF_HOME=/workspace/.hf` and writes a small `scripts/run_sweep_a100.sh` wrapper with the recommended flags so you can re-run without re-remembering the CLI.

### 3. Recommended A100 sweep invocation (in the doc + wrapper script)

```bash
python -m backend.experiments.clip_recond_sweep \
    --inputs backend/images/standard_benchmark.jpg \
    --input-labels living_room \
    --experiment-root backend/experiments/clip_recond \
    --prompt-set default \
    --lambdas 0.0 0.05 0.1 0.2 0.3 \
    --num-views 10 \
    --num-steps 25 \
    --dtype bf16 \
    --skip-consistency \
    --skip-existing
```

Rationale per flag:
- `--num-steps 25`: standard diffusion halving; ~2x speedup vs. default 50 with near-identical output. Free win on any GPU.
- `--dtype bf16`: on A100 this engages FLASH with bf16 Tensor Cores and removes per-op fp32->fp16 casts. Expected 2-3x further speedup (opposite of what you saw on 5070 Ti, where it regressed).
- `--skip-consistency`: DUSt3R self-consistency adds ~1-2 min/run; compute it offline later from saved views by re-running the metrics module.
- `--skip-existing`: crash-safe resumability against `summary/all_runs.csv`.

Expected A100 40 GB wall time: ~3-4 min per run, ~5 hours for the full 90-cell sweep, **~$6 total**.

### 4. Tiny `backend/requirements.txt` comment update

Replace the line
```
# PyTorch (CUDA recommended)
```
with a short note that any CUDA 12.1+ PyTorch 2.4+ wheel works on A100 (no cu128 requirement), and call out the extra packages (`kornia`, `imageio-ffmpeg`, `roma`, `gradio`, `flash-attn`) that are currently implicit transitives pulled in ad-hoc. Moving them into `requirements.txt` means the RunPod bootstrap is a single `pip install -r`.

## What I will NOT change

- Default values of `--dtype`, `--num-steps`, SDPA backend list, or the `SEVA_AUTOCAST_DTYPE` default. Those defaults are already safe on every machine; bf16 is opt-in through a flag because it's not universally a win (your Blackwell case proves that).
- Any model code in `backend/vendor/`. The existing patches are already A100-compatible.
- The master CSV schema or run layout - the A100 run produces CSV rows identical in structure to your local runs, so cross-hardware comparisons stay clean.

## Deliverables

- [docs/SETUP_RUNPOD_A100.md](docs/SETUP_RUNPOD_A100.md) (new)
- [scripts/setup_runpod.sh](scripts/setup_runpod.sh) (new)
- [scripts/run_sweep_a100.sh](scripts/run_sweep_a100.sh) (new, tiny wrapper)
- [backend/requirements.txt](backend/requirements.txt) (comment + add 5 transitive deps)