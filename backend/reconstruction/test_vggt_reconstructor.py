"""Smoke test for VGGTReconstructor.

Run from the repo root (so `backend/` is on sys.path) as:

    python -m backend.test_vggt_reconstructor

or directly:

    cd backend && python test_vggt_reconstructor.py

By default it reconstructs from `backend/duster_images/` (the 10 orbit views
already checked into the repo) and writes `scene.glb` to
`backend/test_vggt_output/`.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# Allow running as a bare script from backend/ without -m.
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from vggt_reconstructor import VGGTReconstructor  # noqa: E402

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        default=str(THIS_DIR / "duster_images"),
        help="Folder of input images (default: backend/duster_images)",
    )
    parser.add_argument(
        "--output_dir",
        default=str(THIS_DIR / "test_vggt_output"),
        help="Where to write scene.glb (default: backend/test_vggt_output)",
    )
    parser.add_argument(
        "--conf_threshold_pct", type=float, default=50.0,
        help="Drop lowest N%% of per-pixel confidences (default: 50).",
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Cap # of input views (uniform subsample) to save VRAM.",
    )
    parser.add_argument(
        "--target_size", type=int, default=574,
        help="VGGT input resolution (pixels, multiple of 14). "
             "Default 574 matches 576x576 SEVA orbit views.",
    )
    parser.add_argument(
        "--depth_chunk_size", type=int, default=2,
        help="Frames per DPT head pass (default 2). Lower = less VRAM.",
    )
    parser.add_argument("--device", default=None, help="Force device (cuda/mps/cpu).")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: images_dir does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing VGGTReconstructor (device={args.device or 'auto'})...")
    t0 = time.time()
    try:
        recon = VGGTReconstructor(device=args.device)
    except Exception as e:
        print(f"ERROR while loading VGGT: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    print(f"Model loaded in {time.time() - t0:.1f}s on device={recon.device}.")

    print(f"Reconstructing from {images_dir} -> {args.output_dir}")
    t0 = time.time()
    try:
        outfile = recon.reconstruct(
            str(images_dir),
            args.output_dir,
            conf_threshold_pct=args.conf_threshold_pct,
            max_images=args.max_images,
            target_size=args.target_size,
            depth_chunk_size=args.depth_chunk_size,
        )
    except Exception as e:
        print(f"ERROR during reconstruction: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"Output GLB: {outfile}")


if __name__ == "__main__":
    main()
