---
name: Save Run Artifacts Log
overview: Enhance the pipeline to write a comprehensive `run_report.txt` after each run, documenting all inputs, intermediate outputs, final outputs, timing breakdowns, and model parameters -- making it easy to share results and know exactly what each file is.
todos:
  - id: collect-timings
    content: Add timings dict and artifacts dict, collect per-step data in run_scene_pipeline
    status: completed
  - id: write-report
    content: Write run_report.txt with full manifest, timings, and run info at end of pipeline
    status: completed
  - id: remove-old-txt
    content: Remove the early model_name.txt write since run_report.txt supersedes it
    status: completed
isProject: false
---

# Save Run Artifacts and Report

## Current State

Each pipeline run saves files into `uploads/<session_id>/` but there is no comprehensive manifest or report. The existing `<model_name>.txt` (written at the start of `run_scene_pipeline` in [backend/app.py](backend/app.py) lines 177-183) only records session_id, model name, prompt, input filename, and timestamp.

## Changes

All changes are in a single file: [backend/app.py](backend/app.py), inside the `run_scene_pipeline` function (line 169 onwards).

### 1. Collect per-step timing and file lists

The pipeline already computes `t0`/`time.time()` for each step but only logs the timings. We will collect them into a dict:

```python
timings = {}
artifacts = {}
```

After each step, record both the elapsed time and the list of files produced:
- **Depth Estimation**: record `depth.png`, `depth_metric.npy`, `depth.npy` + timing
- **Novel View Synthesis**: record all `view_*.png` files + timing
- **3D Reconstruction**: record `scene.glb` + timing
- **Scene Composition**: record `scene.json` + timing

### 2. Write `run_report.txt` at the end of the pipeline

After the "Done" status update, write a human-readable report to `uploads/<session_id>/run_report.txt` with:

- **Run Info**: session_id, model, prompt, timestamp, input image dimensions
- **Artifacts by Stage**:
  - Stage 1 (DepthPro): lists `depth.png`, `depth_metric.npy`, estimated FOV
  - Stage 2 (NVS / before DUSt3R): lists all `views/view_*.png` files with count
  - Stage 3 (DUSt3R / after): lists `scene.glb` with file size
  - Stage 4 (Composition): lists `scene.json`
- **Timing Breakdown**: each step's duration + total pipeline time
- **File Manifest**: flat list of every file in the session directory with relative paths and sizes

### 3. Replace the existing minimal `<model_name>.txt`

Remove the early write of `<model_name>.txt` (lines 177-183) since `run_report.txt` supersedes it with strictly more information.

## What the output looks like

```
========================================
 RUN REPORT — session abc12345
========================================
Model:      seva
Prompt:     (none)
Input:      input.jpg (1920x1080)
Timestamp:  2026-04-18T14:32:01

── Stage 1: Depth Estimation (DepthPro) ──
  Time: 3.42s
  FOV:  68.5 degrees
  Files:
    depth.png           (245 KB)
    depth_metric.npy    (8.3 MB)
    depth.npy           (8.3 MB)

── Stage 2: Novel View Synthesis (seva) ──
  Time: 48.12s
  Views generated: 10
  Files:
    views/view_000.png  (312 KB)
    views/view_001.png  (298 KB)
    ...

── Stage 3: 3D Reconstruction (DUSt3R) ──
  Time: 22.07s
  Files:
    scene.glb           (14.2 MB)

── Stage 4: Scene Composition ──
  Time: 0.01s
  Files:
    scene.json          (0.4 KB)

── Summary ──
  Total pipeline time: 73.62s
  Total files: 16
  Total size:  31.4 MB
```

## Files Changed

- **[backend/app.py](backend/app.py)**: `run_scene_pipeline` function only
