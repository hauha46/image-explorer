"""
Novel View Synthesis via Zero123++ (v1.2).

Zero123++ generates 6 multi-view-consistent images from a single input
in one forward pass.  The output is a 960x640 image containing a 3x2
grid of 320x320 views at fixed camera poses:

  Azimuths (relative): 30, 90, 150, 210, 270, 330 degrees
  Elevations (absolute): 20, -10, 20, -10, 20, -10 degrees
  FOV: 30 degrees

These views are split into individual PNGs for DUSt3R consumption.
Weights auto-download from HuggingFace (~5 GB) on first use.
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

HF_MODEL_ID = "sudo-ai/zero123plus-v1.2"
HF_CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

GRID_COLS = 3
GRID_ROWS = 2
TILE_W = 320
TILE_H = 320
NUM_VIEWS = GRID_COLS * GRID_ROWS  # always 6


class Zero123PPSynthesizer(BaseSynthesizer):
    name = "zero123pp"

    def __init__(self):
        self.pipe = None
        self.device = "cpu"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.pipe is not None:
            logger.info("Zero123++ pipeline already loaded — skipping.")
            return

        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        logger.info(f"Loading Zero123++ pipeline from {HF_MODEL_ID} …")
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            HF_MODEL_ID,
            custom_pipeline=HF_CUSTOM_PIPELINE,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.to(device)
        logger.info("Zero123++ pipeline loaded.")

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
            raise RuntimeError(
                "Zero123++ model not loaded. Call load_model() first."
            )

        if prompt:
            logger.info(
                "Zero123++ is image-conditioned only — ignoring prompt."
            )

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        img_cond = img.resize((TILE_W, TILE_H), Image.LANCZOS)

        logger.info("Running Zero123++ inference (6 views) …")
        with torch.inference_mode():
            result = self.pipe(
                img_cond,
                num_inference_steps=75,
                guidance_scale=4.0,
            )
        grid_image: Image.Image = result.images[0]

        saved: list[str] = []
        idx = 0
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                left = col * TILE_W
                upper = row * TILE_H
                tile = grid_image.crop(
                    (left, upper, left + TILE_W, upper + TILE_H)
                )
                out_path = views_dir / f"view_{idx:03d}.png"
                tile.save(str(out_path))
                saved.append(str(out_path))
                logger.info(f"  saved view {idx} → {out_path.name}")
                idx += 1

        logger.info(f"Zero123++: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logger.info("Zero123++ pipeline unloaded.")
