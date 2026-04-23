#!/usr/bin/env python3
"""One-shot reconstruction + meshing from a folder of images.

Run from the repo root (so ``backend/`` is on sys.path) as:

    uv run python backend/run_reconstruction.py --config backend/configs/default.yaml
    uv run python backend/run_reconstruction.py --config backend/configs/indoor_orbit.yaml
    uv run python backend/run_reconstruction.py --config default.yaml --method dust3r

What it does
------------
1. Loads a YAML config (or uses built-in defaults).
2. Reconstructs the images into a point-cloud ``.glb``.
3. Meshes that point cloud with the configured algorithm.
4. Writes both outputs into ``backend/outputs/<method>/<algo>/`` (or custom dir).

Command-line flags always override YAML values.
"""

import argparse
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

# Allow running as: uv run python backend/reconstruction/run_reconstruction.py from repo root.
THIS_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = THIS_DIR.parent.resolve()
REPO_ROOT = BACKEND_DIR.parent.resolve()

for p in (THIS_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = BACKEND_DIR / "duster_images"


def _load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_absolute():
        # Try multiple bases so the user can pass either
        #   --config configs/default.yaml          (from repo root)
        #   --config backend/configs/default.yaml  (also from repo root)
        for base in (Path.cwd(), REPO_ROOT, BACKEND_DIR):
            candidate = base / p
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        logger.warning(f"Config file not found: {p}; using CLI defaults only.")
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def _flatten_yaml(cfg: dict) -> dict:
    """Flatten nested YAML into argparse-friendly flat dict.

    e.g. {reconstructor: {method: dust3r}} -> {method: dust3r}
         {meshing: {algo: planes}} -> {algo: planes}
    """
    flat = {}
    # top-level keys
    flat["images_dir"] = cfg.get("images_dir", str(DEFAULT_IMAGES_DIR))
    flat["output_dir"] = cfg.get("output_dir")

    # reconstructor block
    rc = cfg.get("reconstructor", {})
    flat["method"] = rc.get("method", "vggt")
    flat["device"] = rc.get("device")

    # per-method reconstructor params
    for m in ("dust3r", "vggt", "vggt_depthpro"):
        flat[m] = rc.get(m, {})

    # meshing block
    mg = cfg.get("meshing", {})
    flat["skip_mesh"] = not mg.get("enabled", True)
    flat["algo"] = mg.get("algo", "poisson")
    flat["outlier_nb"] = mg.get("outlier_nb", 30)
    flat["outlier_std"] = mg.get("outlier_std", 1.5)
    flat["normal_knn"] = mg.get("normal_knn", 30)

    # per-algo meshing params
    flat["poisson"] = mg.get("poisson", {})
    flat["bpa"] = mg.get("bpa", {})
    flat["alpha"] = mg.get("alpha", {})
    flat["planes"] = mg.get("planes", {})

    return flat


def _ensure_images(images_dir: Path) -> Path:
    if not images_dir.exists():
        logger.error(f"Input images not found: {images_dir}")
        sys.exit(1)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    n = sum(len(list(images_dir.glob(e))) for e in exts)
    logger.info(f"Found {n} images in {images_dir}")
    return images_dir


def _clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        logger.info(f"Removing previous output: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run_vggt(images_dir: Path, device: str | None, out_dir: Path, cfg: dict) -> Path:
    from vggt_reconstructor import VGGTReconstructor

    logger.info("=== VGGT Reconstruction ===")
    recon = VGGTReconstructor(device=device)
    vggt_cfg = cfg.get("vggt", {})
    scene_glb = recon.reconstruct(
        str(images_dir),
        str(out_dir),
        conf_threshold_pct=vggt_cfg.get("conf_threshold_pct", 50.0),
        target_size=vggt_cfg.get("target_size", 574),
        depth_chunk_size=vggt_cfg.get("depth_chunk_size", 2),
        max_images=vggt_cfg.get("max_images"),
    )
    return Path(scene_glb)


def run_vggt_depthpro(images_dir: Path, device: str | None, out_dir: Path, cfg: dict) -> Path:
    from vggt_depthpro_reconstructor import VGGTDepthProReconstructor

    logger.info("=== VGGT + DepthPro Fusion ===")
    recon = VGGTDepthProReconstructor(device=device)
    vdp_cfg = cfg.get("vggt_depthpro", {})
    scene_glb = recon.reconstruct(
        str(images_dir),
        str(out_dir),
        conf_threshold_pct=vdp_cfg.get("conf_threshold_pct", 10.0),
        vggt_target_size=vdp_cfg.get("vggt_target_size", 574),
        depthpro_size=vdp_cfg.get("depthpro_size", 1500),
        depth_chunk_size=vdp_cfg.get("depth_chunk_size", 2),
        outlier_pct=vdp_cfg.get("outlier_pct", 5.0),
        max_images=vdp_cfg.get("max_images"),
    )
    return Path(scene_glb)


def run_dust3r(images_dir: Path, device: str | None, out_dir: Path, cfg: dict) -> Path:
    from reconstruction.dust3r_reconstructor import Dust3rReconstructor

    logger.info("=== DUSt3R Reconstruction ===")
    recon = Dust3rReconstructor(device=device)
    scene_glb = recon.reconstruct(
        str(images_dir),
        str(out_dir),
        as_pointcloud=cfg.get("as_pointcloud", True),
    )
    return Path(scene_glb)


def run_mesh(scene_glb: Path, mesh_glb: Path, cfg: dict) -> None:
    from reconstruction.mesh_generator import point_cloud_to_mesh

    algo = cfg["algo"]
    logger.info(f"=== Meshing ({algo}) ===")

    kwargs = {
        "glb_path": str(scene_glb),
        "output_path": str(mesh_glb),
        "algo": algo,
        "outlier_nb": cfg.get("outlier_nb", 30),
        "outlier_std": cfg.get("outlier_std", 1.5),
        "normal_knn": cfg.get("normal_knn", 30),
    }

    if algo == "poisson":
        p = cfg.get("poisson", {})
        kwargs.update(
            depth=p.get("depth", 10),
            scale=p.get("scale", 1.0),
            linear_fit=p.get("linear_fit", True),
            point_weight=p.get("point_weight", 15.0),
            density_quantile=p.get("density_quantile", 0.4),
            smooth_iters=p.get("smooth_iters", 0),
        )
    elif algo == "bpa":
        p = cfg.get("bpa", {})
        kwargs.update(
            bpa_radius_mult=p.get("radius_mult", 2.0),
            bpa_radii_levels=p.get("radii_levels", 3),
        )
    elif algo == "alpha":
        p = cfg.get("alpha", {})
        kwargs.update(alpha_mult=p.get("mult", 2.0))
    elif algo == "planes":
        p = cfg.get("planes", {})
        kwargs.update(
            plane_distance_threshold=p.get("distance_threshold", 0.02),
            plane_ransac_n=p.get("ransac_n", 3),
            plane_num_iterations=p.get("num_iterations", 1000),
            plane_min_plane_ratio=p.get("min_plane_ratio", 0.05),
            plane_max_planes=p.get("max_planes", 10),
        )
    else:
        raise ValueError(f"Unknown meshing algo: {algo}")

    point_cloud_to_mesh(**kwargs)


def main():
    # ------------------------------------------------------------------
    # 1. Pre-parse to grab --config
    # ------------------------------------------------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to YAML config file.")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    yaml_cfg = _load_yaml(pre_args.config)
    flat_defaults = _flatten_yaml(yaml_cfg)

    # ------------------------------------------------------------------
    # 2. Main parser with YAML as defaults
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="One-shot reconstruct + mesh from a folder of images. "
                    "YAML config values are overridden by explicit CLI flags.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--images-dir",
        default=flat_defaults.get("images_dir", str(DEFAULT_IMAGES_DIR)),
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--method",
        default=flat_defaults.get("method", "vggt"),
        choices=["vggt", "vggt_depthpro", "dust3r"],
        help="Which reconstructor to use.",
    )
    parser.add_argument(
        "--device",
        default=flat_defaults.get("device"),
        help="Force device: cuda / mps / cpu (auto-detect if omitted).",
    )
    parser.add_argument(
        "--output",
        default=flat_defaults.get("output_dir"),
        help="Custom output directory. Default: backend/outputs/<method>/<algo>/",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        default=flat_defaults.get("skip_mesh", False),
        help="Only run reconstruction, skip mesh generation.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="VGGT input resolution in pixels (must be multiple of 14). "
             "Lower = less VRAM. 518 = trained res, ~8 GB. 574 = ~12 GB+. "
             "Overrides the YAML config value.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of input images (uniform subsample). "
             "Useful to stay under 12 GB VRAM. Overrides YAML.",
    )
    args = parser.parse_args(remaining_argv)

    # ------------------------------------------------------------------
    # 3. Re-assemble the full nested config (CLI overrides YAML)
    # ------------------------------------------------------------------
    cfg = {
        "images_dir": args.images_dir,
        "output_dir": args.output,
        "method": args.method,
        "device": args.device,
        "skip_mesh": args.skip_mesh,
        "algo": flat_defaults.get("algo", "poisson"),
        "outlier_nb": flat_defaults.get("outlier_nb", 30),
        "outlier_std": flat_defaults.get("outlier_std", 1.5),
        "normal_knn": flat_defaults.get("normal_knn", 30),
        # per-method reconstructor blocks
        "dust3r": flat_defaults.get("dust3r", {}),
        "vggt": flat_defaults.get("vggt", {}),
        "vggt_depthpro": flat_defaults.get("vggt_depthpro", {}),
        # per-algo meshing blocks
        "poisson": flat_defaults.get("poisson", {}),
        "bpa": flat_defaults.get("bpa", {}),
        "alpha": flat_defaults.get("alpha", {}),
        "planes": flat_defaults.get("planes", {}),
    }

    # Override meshing algo if the user somehow passed it (not exposed as CLI
    # flag yet, but could be added easily).
    cfg["algo"] = flat_defaults.get("algo", "poisson")

    # CLI overrides for VGGT / VGGT+DepthPro VRAM tuning
    if args.target_size is not None:
        cfg.setdefault("vggt", {})["target_size"] = args.target_size
        cfg.setdefault("vggt_depthpro", {})["vggt_target_size"] = args.target_size
    if args.max_images is not None:
        cfg.setdefault("vggt", {})["max_images"] = args.max_images
        cfg.setdefault("vggt_depthpro", {})["max_images"] = args.max_images

    # ------------------------------------------------------------------
    # 4. Paths
    # ------------------------------------------------------------------
    images_dir = Path(args.images_dir).resolve()
    _ensure_images(images_dir)
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = THIS_DIR / "outputs" / args.method / cfg["algo"]
    _clean_output_dir(out_dir)

    scene_glb = out_dir / "scene.glb"
    mesh_glb = out_dir / "scene_mesh.glb"

    # ------------------------------------------------------------------
    # 5. Stage 1: Reconstruction
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        if args.method == "vggt":
            run_vggt(images_dir, args.device, out_dir, cfg["vggt"])
        elif args.method == "vggt_depthpro":
            run_vggt_depthpro(images_dir, args.device, out_dir, cfg["vggt_depthpro"])
        elif args.method == "dust3r":
            run_dust3r(images_dir, args.device, out_dir, cfg["dust3r"])
        else:
            raise ValueError(f"Unhandled method: {args.method}")
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    recon_time = time.time() - t0
    logger.info(f"Reconstruction done in {recon_time:.1f}s -> {scene_glb}")

    if args.skip_mesh:
        logger.info("--skip-mesh set, done.")
        return

    # ------------------------------------------------------------------
    # 6. Stage 2: Meshing
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        run_mesh(scene_glb, mesh_glb, cfg)
    except Exception as e:
        logger.error(f"Meshing failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    mesh_time = time.time() - t0
    logger.info(f"Meshing done in {mesh_time:.1f}s -> {mesh_glb}")
    logger.info(
        f"\nTotal: {recon_time + mesh_time:.1f}s | "
        f"Point cloud: {scene_glb} | Mesh: {mesh_glb}"
    )


if __name__ == "__main__":
    main()
