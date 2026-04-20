# RunPod A100 quickstart

Run the 6 "main" pipeline cells + the 6 CLIP re-conditioning experiment
cells on a rented A100 80GB and download everything for the report.

All heavy lifting is already scripted under `scripts/`; this doc is just the
step-by-step list of commands you type.  Expected total A100 time: ~1-1.5
hours, ~$2-3 at current RunPod Secure Cloud rates.

---

## 1. Launch the pod

On [runpod.io](https://runpod.io):

1. Pods -> **+ Deploy**
2. GPU: **1x A100 80GB PCIe**, Cloud type: **Secure Cloud**
3. Template: **RunPod PyTorch 2.4** (any 2.x with CUDA 12.1+)
4. Container disk: **20 GB**
5. **Volume: 50 GB at `/workspace`** - critical, the HuggingFace cache lives
   here so SEVA's ~8 GB gated weights survive pod restarts
6. **Deploy On-Demand**, wait ~60 s for `Running`
7. Click **Connect -> Start Web Terminal -> Connect**

---

## 2. Transfer the repo with `runpodctl`

On the pod:

```bash
cd /workspace
runpodctl receive
# leaves you at:  Enter the one-time code:
```

On your Windows PowerShell (assuming you've tarred the repo locally as
`image-explorer.tar.gz` in Downloads, and `runpodctl.exe` is in Downloads too):

```powershell
cd C:\Users\bcliu\Downloads
.\runpodctl.exe send image-explorer.tar.gz
# copy the code it prints, e.g. 8219-snowball-golden-hotel
```

Paste the code into the waiting pod terminal.  Wait for the progress bar.

Extract and verify:

```bash
cd /workspace
tar -xzf image-explorer.tar.gz
ls image-explorer/backend/images/    # standard_benchmark.jpg, techbro_room.jpg
ls image-explorer/scripts/           # setup_runpod.sh, run_all.sh, etc.
ls image-explorer/backend/vendor/    # dust3r, stable-virtual-camera, ViewCrafter, ...
```

If any of those are empty, the Windows tar missed them - rebuild the tarball
with the right excludes and resend.

---

## 3. Bootstrap the environment

```bash
cd /workspace/image-explorer
bash scripts/setup_runpod.sh
```

This (idempotent) script:

- exports `HF_HOME=/workspace/.hf` + persists it to `~/.bashrc`
- installs `uv`
- creates `/workspace/.venv` (python 3.11)
- installs torch 2.4.1 (cu121), `backend/requirements.txt`, the transitive
  deps SEVA/DUSt3R/ViewCrafter need, and tries a prebuilt flash-attn wheel
  (non-fatal if it fails - SDPA fallback covers us)
- ends with an **A100 SDPA sanity check** - look for a line like:

  ```
  FLASH     OK
  ```

  If you see `FLASH     FAIL` instead, you did not actually get an A100 and
  should stop before burning money.  `nvidia-smi` should show `A100`.

Total bootstrap time: ~5-8 min on a fresh pod.

---

## 4. HuggingFace login (one-time, interactive)

Accept the license for [stabilityai/stable-virtual-camera](https://huggingface.co/stabilityai/stable-virtual-camera)
in your browser first (click *Access repository*), then:

```bash
source /workspace/.venv/bin/activate
huggingface-cli login
# paste a HF token with read access; input is hidden
```

The token is cached under `/workspace/.hf/token` so the pod survives restarts.

---

## 5. Kick off the full sweep (detached)

```bash
tmux new -s sweep
cd /workspace/image-explorer
source /workspace/.venv/bin/activate
bash scripts/run_all.sh
```

Detach with `Ctrl+b  d`.  The script keeps running.  What it does:

1. Starts `uvicorn backend.app:app` on 127.0.0.1:9876 in the background,
   waits for `/` to answer.
2. Runs `scripts/run_main.sh` - 6 pipeline cells
   (2 SEVA + 4 ViewCrafter), teed to `main.log`.
3. Runs `scripts/run_experiment.sh` - 6 CLIP cells (3 SEVA + 3 ViewCrafter,
   `clip_lambda=0.25` for SEVA), teed to `sweep.log`.
4. Packages everything into `~/a100_results.tar.gz` (excluding the bulky
   `views/_seva_work/` SEVA staging directories).
5. Stops the uvicorn it started.

Soft-fail is built in: any single failing cell is logged as `FAILED
status=... sid=...` and the sweep continues; a failing sub-script is caught
at the `run_all.sh` level so the other one still runs.

Reattach any time with `tmux attach -t sweep`, or monitor from a separate
terminal:

```bash
tail -f /workspace/image-explorer/main.log        # while main cells are running
tail -f /workspace/image-explorer/sweep.log       # during experiment cells
cat    /workspace/image-explorer/backend/uploads/sessions_index.jsonl   # compact per-run rows
```

---

## 6. Download results

When `run_all.sh` prints

```
Packaged results -> /root/a100_results.tar.gz (XXX MB / XXXXXX bytes)
```

transfer it back:

```bash
# on the pod
cd ~
runpodctl send a100_results.tar.gz
# copy the code it prints
```

```powershell
# on Windows
cd C:\Users\bcliu\Downloads
.\runpodctl.exe receive <code>
tar -xzf a100_results.tar.gz
```

Inside the extracted `a100_results/` you'll find:

- `MANIFEST.md` - plain-English description of every file (read this first)
- `main.log`, `sweep.log`, `run_all.log`, `server.log`
- `main_session_map.tsv`, `exp_session_map.tsv` - human label -> session_id
- `backend/uploads/<session_id>/` - one folder per run, each with
  `run_report.txt` (human), `run_info.json` (machine), `input.jpg`,
  `depth.png`, `depth_metric.npy`, `views/view_*.png`, `views/trajectory.json`,
  `scene.glb`, `scene_mesh.glb`, `scene.json`
- `backend/uploads/sessions_index.jsonl` - one JSON row per run,
  pandas-loadable

---

## 7. Stop paying

Critical.  A100 80GB is ~$1.79/hr even idle.

On runpod.io -> your pod -> **Stop** preserves the volume (so a future run
skips `scripts/setup_runpod.sh`).  **Terminate** deletes the volume too.

If you stop and relaunch later, `/workspace/.venv` + `/workspace/.hf` both
persist, so you go straight to `bash scripts/run_all.sh`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FLASH FAIL` at end of Phase 3 | Not on an A100.  `nvidia-smi` should show `A100`. |
| `GatedRepoError` on first SEVA cell | You haven't accepted SEVA's license in a browser.  Visit the repo page, click *Access repository*, re-run. |
| ViewCrafter first run very slow | Downloading ~10 GB of checkpoints to `backend/vendor/ViewCrafter/checkpoints/`.  Subsequent runs are fast. |
| One cell logged `FAILED`, others OK | Working as intended - soft-fail.  Check that session's `run_report.txt` / `run_info.json` for the stack trace. |
| Tarball is much bigger than expected | `views/_seva_work/` exclude might have slipped.  `du -sh backend/uploads/*/views/_seva_work/` to confirm. |
| `run_all.sh` hangs with no output | `tmux attach -t sweep`; usually mid-diffusion and fine. |
