#!/usr/bin/env python3
"""
End-to-end demo: single image → SEVA novel views → DUSt3R reconstruction + meshing.

Usage:
    uv run python demo_pipeline.py --image benchmark_image_1.jpg
    uv run python demo_pipeline.py --image benchmark_image_1.jpg --num-views 8
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure backend/ is on the path
REPO_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from novel_view_synthesis.seva_synthesizer_4070ti import SevaSynthesizer
from reconstruction.dust3r_reconstructor import Dust3rReconstructor
from reconstruction.mesh_generator import point_cloud_to_mesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Output folder at repo root for easy grader access
FINAL_OUTPUT_DIR = REPO_ROOT / "final_demo_outputs"


def main():
    parser = argparse.ArgumentParser(description="Image → SEVA views → DUSt3R → Mesh")
    parser.add_argument("--image", default="benchmark_image_1.jpg", help="Path to input image (default: benchmark_image_1.jpg)")
    parser.add_argument("--num-views", type=int, default=8, help="Number of SEVA views to generate")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    views_dir = FINAL_OUTPUT_DIR / "views"
    recon_dir = FINAL_OUTPUT_DIR / "reconstruction"

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # Create (or recreate) output folders
    if FINAL_OUTPUT_DIR.exists():
        shutil.rmtree(FINAL_OUTPUT_DIR)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Novel View Synthesis (SEVA)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: Generating novel views with SEVA")
    logger.info("=" * 60)

    synth = SevaSynthesizer()
    synth.load_model(device=args.device)

    t0 = time.time()
    view_paths = synth.generate_views(
        image_path=str(image_path),
        output_dir=str(FINAL_OUTPUT_DIR),
        num_views=args.num_views,
    )
    nvs_time = time.time() - t0
    logger.info(f"Generated {len(view_paths)} views in {nvs_time:.1f}s")
    logger.info(f"Views saved to: {views_dir}")

    # SEVA writes raw frames under _seva_work/samples-rgb/; use those
    # directly for reconstruction rather than the copied view_*.png subset.
    raw_samples_dir = views_dir / "_seva_work" / "samples-rgb"
    if not raw_samples_dir.exists():
        logger.error(f"SEVA raw frames not found at {raw_samples_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 2: 3D Reconstruction (DUSt3R) - no config file, direct call
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Reconstructing 3D point cloud with DUSt3R")
    logger.info("=" * 60)

    recon = Dust3rReconstructor(device=args.device)

    t0 = time.time()
    scene_glb = recon.reconstruct(
        images_dir=str(raw_samples_dir),
        output_dir=str(recon_dir),
        as_pointcloud=True,
    )
    recon_time = time.time() - t0
    logger.info(f"Point cloud saved to: {scene_glb} ({recon_time:.1f}s)")

    # ------------------------------------------------------------------
    # Stage 3: Meshing (mandatory)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: Meshing with BPA algorithm")
    logger.info("=" * 60)

    mesh_glb = recon_dir / "scene_mesh.glb"
    t0 = time.time()
    point_cloud_to_mesh(
        glb_path=str(scene_glb),
        output_path=str(mesh_glb),
        algo="bpa",
        outlier_std=1.2,
        outlier_nb=30,
        normal_knn=30,
        bpa_radius_mult=2.0,
        bpa_radii_levels=3,
    )
    mesh_time = time.time() - t0
    logger.info(f"Mesh saved to: {mesh_glb} ({mesh_time:.1f}s)")

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Total time:  {total_time:.1f}s")
    logger.info(f"  Views:       {views_dir}")
    logger.info(f"  Point cloud: {scene_glb}")
    logger.info(f"  Mesh:        {mesh_glb}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
