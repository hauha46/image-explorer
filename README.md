GitHub link: https://github.com/hauha46/image-explorer

# For Graders:

For inference, we use SVC, found here: https://huggingface.co/stabilityai/stable-virtual-camera.

This is a gated model, meaning you will need to have an active huggingface token in order to access it. While other models don't require this, 
we chose to use SVC as it gives the best results.

The pipeline will setup without a huggingface token, but it will not be able to download or run SEVA, which will cause the pipeline to fail.

# Requirements

To run the demo_pipeline.py file, which is one inference run, you will need at least 12 GB of VRAM. This will produce results but they will not be as nice
as what we have in our report and presentation because we used an A100 for those.


# Setup (Also for Graders)
First, run this command to install the necessary vendor repositories, model checkpoints, and Python dependencies:

```bash
bash setup.sh
```

This will download 9 models that were used in making our pipeline. During execution of our pipeline, only 3 are used; however, the other models were used in testing.

Second, ensure your Python packages are synced:

```bash
uv sync
```

Third, run the demo pipeline to see the end-to-end inference results (image → SEVA novel views → DUSt3R reconstruction → BPA meshing):

```bash
uv run python demo_pipeline.py
```

The default input image is `benchmark_image_1.jpg`. To use the default image:
```bash
uv run python demo_pipeline.py
```

To use a different image:

```bash
uv run python demo_pipeline.py --image path/to/your_image.jpg
```

Output will be saved to `final_demo_outputs/` at the repo root:
- `final_demo_outputs/views/_seva_work/samples-rgb/` — generated novel views
- `final_demo_outputs/reconstruction/scene.glb` — DUSt3R point cloud
- `final_demo_outputs/reconstruction/scene_mesh.glb` — BPA mesh

If you want to see the frontend (note: the frontend is not connected to the pipeline due to breaking code changes):

```bash
bash run.sh
```


## Top-Level File Tree

```
image-explorer/
│
├── README.md                          # Setup & grader instructions
├── setup.sh                           # One-shot setup (vendors, weights, deps)
├── demo_pipeline.py                   # ⭐ MAIN INFERENCE SCRIPT
│                                      #    Single image → SEVA views → DUSt3R → mesh
│
├── backend/                           # All backend / ML code
│   ├── app.py                         # FastAPI server (loads models, exposes endpoints)
│   ├── scene_processor.py             # Core pipeline orchestrator
│   │                                    # (depth → NVS → reconstruction → JSON export)
│   ├── depth_pro_estimator.py         # DepthPro depth / FOV estimation
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── configs/                       # YAML / JSON presets for camera orbits
│   │   ├── default.yaml
│   │   ├── indoor_orbit.yaml
│   │   └── seva_orbit_preset_for_dust3r.json
│   │
│   ├── novel_view_synthesis/          # ⭐ NVS MODELS
│   │   ├── base.py                    # Abstract base class for all synthesizers
│   │   ├── seva_synthesizer.py        # SEVA (Stable Virtual Camera) — primary model
│   │   ├── seva_synthesizer_4070ti.py # SEVA variant tuned for 12 GB VRAM
│   │   ├── sv3d_synthesizer.py        # SV3D (Stable Video 3D)
│   │   ├── svd_synthesizer.py         # SVD (Stable Video Diffusion)
│   │   ├── vivid_synthesizer.py       # VIVID
│   │   ├── panodreamer_synthesizer.py # PanoDreamer
│   │   ├── viewcrafter_synthesizer.py # ViewCrafter
│   │   └── zero123pp_synthesizer.py   # Zero123++
│   │
│   ├── reconstruction/                # ⭐ 3D RECONSTRUCTION
│   │   ├── dust3r_reconstructor.py    # DUSt3R point-cloud reconstruction
│   │   ├── vggt_reconstructor.py      # VGGT reconstruction
│   │   ├── vggt_depthpro_reconstructor.py
│   │   ├── mesh_generator.py          # Ball-Pivoting Algorithm (BPA) meshing
│   │   └── run_reconstruction.py      # CLI entry point for reconstruction
│   │
│   ├── experiments/                   # Research / ablation code
│   │   ├── clip_recond_sweep.py       # CLIP text re-conditioning sweep
│   │   ├── metrics.py                 # PSNR, SSIM, LPIPS, CLIP score helpers
│   │   ├── prompts.py                 # Prompt bank for experiments
│   │   ├── make_report_artifacts.py   # Generate contact sheets / grids
│   │   └── clip_recond/               # Saved sweep runs + per-run READMEs
│   │       ├── summary/all_runs.csv
│   │       └── 2026.../               # Individual run folders
│   │
│   └── vendor/                        # Third-party repositories (git submodules / clones)
│       ├── dust3r/                    # DUSt3R official repo
│       ├── stable-virtual-camera/     # SEVA official repo
│       ├── DepthPro/                  # DepthPro depth estimator
│       ├── vggt/                      # VGGT
│       ├── PanoDreamer/
│       ├── ViewCrafter/
│       ├── sv3d-diffusers/
│       └── ml-depth-pro, ml-vivid     # Additional depth / vivid vendors
│
├── frontend/                          # Minimal web UI (NOT connected to pipeline)
│   ├── index.html
│   └── src/
│       ├── main.js
│       └── style.css
│
├── final_demo_outputs/                # Generated by demo_pipeline.py
│   ├── views/
│   │   └── _seva_work/samples-rgb/    # Novel view frames
│   └── reconstruction/
│       ├── scene.glb                  # Point cloud
│       └── scene_mesh.glb             # BPA mesh
│
└── benchmark_image_1.jpg              # Default input for demo_pipeline.py
```

---

## Where to Look for What

| What you want to grade | Where to find it |
|---|---|
| **End-to-end inference script** | `demo_pipeline.py` |
| **Core pipeline logic** (depth → NVS → reconstruction) | `backend/scene_processor.py` |
| **Novel View Synthesis (SEVA)** | `backend/novel_view_synthesis/seva_synthesizer_4070ti.py` |
| **3D Reconstruction (DUSt3R)** | `backend/reconstruction/dust3r_reconstructor.py` |
| **Mesh generation (BPA)** | `backend/reconstruction/mesh_generator.py` |
| **Depth / FOV estimation** | `backend/depth_pro_estimator.py` |
| **CLIP re-conditioning experiments** | `backend/experiments/clip_recond_sweep.py` |
| **Experiment metrics & prompts** | `backend/experiments/metrics.py`, `backend/experiments/prompts.py` |
| **API / server entry point** | `backend/app.py` |
| **Vendor repos (external code)** | `backend/vendor/*` |

---
