"""
Novel View Synthesis via Stable Video Diffusion (img2vid-xt).

SVD generates a 25-frame video clip from a single conditioning image.
We extract N evenly-spaced frames and save them as individual PNGs
that DUSt3R can consume for 3D reconstruction.

Note: SVD has no explicit camera-pose control — the generated motion
is implicit.  It works well as a baseline and is the easiest backend
to run (weights download automatically from HuggingFace).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

HF_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
SVD_NUM_FRAMES = 25
SVD_DECODE_CHUNK = 4


class SVDSynthesizer(BaseSynthesizer):
    name = "svd"

    def __init__(self):
        self.pipe = None
        self.device = "cpu"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.pipe is not None:
            logger.info("SVD pipeline already loaded — skipping.")
            return

        from diffusers import StableVideoDiffusionPipeline

        logger.info(f"Loading SVD pipeline from {HF_MODEL_ID} …")
        self.device = device
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.unet.enable_forward_chunking()
        logger.info("SVD pipeline loaded (CPU-offload + forward-chunking enabled).")

    # ------------------------------------------------------------------
    def generate_views(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 8,
        depth_map: Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> list[str]:
        if self.pipe is None:
            raise RuntimeError("SVD model not loaded. Call load_model() first.")

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        # SVD expects 1024x576 (16:9) conditioning image
        img_resized = img.resize((1024, 576), Image.LANCZOS)

        logger.info(f"Running SVD inference ({SVD_NUM_FRAMES} frames) …")
        with torch.inference_mode():
            result = self.pipe(
                img_resized,
                num_frames=SVD_NUM_FRAMES,
                decode_chunk_size=SVD_DECODE_CHUNK,
                generator=torch.Generator(device="cpu").manual_seed(42),
            )
        frames: list[Image.Image] = result.frames[0]

        # Pick num_views evenly-spaced frame indices (always include first & last)
        if num_views >= SVD_NUM_FRAMES:
            indices = list(range(SVD_NUM_FRAMES))
        else:
            indices = [
                round(i * (SVD_NUM_FRAMES - 1) / (num_views - 1))
                for i in range(num_views)
            ]

        saved: list[str] = []
        for idx in indices:
            out_path = views_dir / f"view_{idx:03d}.png"
            frames[idx].save(str(out_path))
            saved.append(str(out_path))
            logger.info(f"  saved frame {idx} → {out_path.name}")

        logger.info(f"SVD: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logger.info("SVD pipeline unloaded.")
