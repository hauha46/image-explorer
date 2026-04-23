# Project Deltas — What We Built Beyond the Baseline

This document summarizes the original contributions ("deltas") of this project on top of the baseline models and pipelines.

---

## Delta 1: Training-Free CLIP Text Re-Conditioning for SEVA

**Baseline:** SEVA (Stable Virtual Camera) loads OpenCLIP ViT-H-14 but only calls `encode_image`. The upstream `/process` API accepted a `prompt` parameter that was logged and silently discarded.

**Our Contribution:** We implemented **Flavor A directional CLIP guidance** (StyleCLIP / StyleGAN-NADA recipe) inside SEVA's conditioning pipeline:

```
Δ = encode_text(target) − encode_text(neutral)
c̃ = c_img + λ · ‖c_img‖ · (Δ / ‖Δ‖)
```

This turns the unused text head into an inference-time steering mechanism for hallucinated regions. We then ran a full sweep across λ ∈ {0.0, 0.05, 0.1, 0.2, 0.3} and multiple prompt axes (sunset, rain, cyberpunk, etc.), measuring CLIPScore, LPIPS drift, and 3D self-consistency (DUSt3R Chamfer proxy) to empirically find the breaking point.

**Files:** `backend/vendor/stable-virtual-camera/seva/modules/conditioner.py`, `backend/synthesizers/seva_synthesizer.py`, `backend/app.py`, `backend/experiments/clip_recond_sweep.py`

---

## Delta 2: Systematic Meshing Comparison + Plane-Based Indoor Reconstruction

**Baseline:** The standard pipeline everywhere (including the original DUSt3R demo) runs Poisson Surface Reconstruction and calls it done. For indoor scenes, this creates the classic "balloon back" — Poisson tries to seal the room like a watertight blob.

**Our Contribution:** We built a unified `mesh_generator.py` with **four algorithms** (Poisson, BPA, Alpha Shapes, Plane-based) and ran them head-to-head on the same point clouds. Our empirical findings:

- **Poisson** fails on open indoor scenes (invents geometry, balloons)
- **BPA** is maximally rigid but too hole-y on sparse areas
- **Alpha Shapes** are a good middle ground for general data
- **Plane-based** (RANSAC plane detection → 2D convex hull triangulation → BPA for non-planar clutter) is the right default for rooms

We didn't just implement plane-based meshing — we tested all four and argued for it as the correct default for this pipeline.

**Files:** `backend/mesh_generator.py`

---

## Delta 3: Empirical Reconstructor Selection Rule

**Baseline:** DUSt3R and VGGT are used as drop-in alternatives in the literature, with no clear guidance on *when* to use which.

**Our Contribution:** We built a unified reconstructor interface (`run_reconstruction.py`, YAML configs) and ran DUSt3R, VGGT, and VGGT+DepthPro fusion on the same data. Our finding:

- **DUSt3R wins on synthetic orbit views** because its `global_aligner` corrects frame-to-frame inconsistencies
- **VGGT loses on synthetic data** because it's feed-forward with no cross-view reconciliation; dense but noisy clouds mesh poorly
- We saved the hardcoded SEVA orbit preset as a **configurable artifact** (`backend/configs/seva_orbit_preset_for_dust3r.json`) and showed it *helps* SEVA data but *breaks* non-orbit data (VEO images)

We turned "which reconstructor should I use?" from a guess into a data-driven choice.

**Files:** `backend/run_reconstruction.py`, `backend/dust3r_reconstructor.py`, `backend/vggt_reconstructor.py`, `backend/vggt_depthpro_reconstructor.py`, `backend/configs/`

---

## One-Sentence Summary

> We extend a single-image-to-3D pipeline with (1) a training-free CLIP directional-guidance mechanism for text control of SEVA's multi-view generation, (2) a systematic comparison of four meshing algorithms leading to a plane-based reconstruction method tailored for indoor scenes, and (3) an empirical analysis of DUSt3R vs. VGGT that establishes when each reconstructor is appropriate.
