"""
Novel View Synthesis via Stable Video 3D (SV3D).

SV3D generates a 21-frame orbital video around an object from a single
conditioning image.  Unlike plain SVD, SV3D has explicit camera control
(azimuth + elevation) and produces 3D-aware views at 576x576.

We use the community diffusers wrapper from chenguolin/sv3d-diffusers
for the custom pipeline classes and converted HuggingFace weights.

Requires:
  - ``backend/vendor/sv3d-diffusers/`` cloned from
    https://github.com/chenguolin/sv3d-diffusers
  - rembg (already a project dependency) for background removal
  - Weights auto-download from HuggingFace (~5 GB) on first use
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "sv3d-diffusers"
HF_MODEL_ID = "chenguolin/sv3d-diffusers"

SV3D_NUM_FRAMES = 21
SV3D_RES = 576
DEFAULT_ELEVATION = 10.0


def _ensure_vendor_on_path() -> None:
    sv3d_root = str(VENDOR_DIR)
    if sv3d_root not in sys.path:
        sys.path.insert(0, sv3d_root)


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Remove background with rembg and paste onto white, resized to 576x576."""
    import rembg

    rgba = rembg.remove(image.convert("RGB"))
    out = Image.new("RGB", rgba.size, (255, 255, 255))
    out.paste(rgba, mask=rgba.split()[3])
    return out.resize((SV3D_RES, SV3D_RES), Image.LANCZOS)


def _build_camera_params(
    elevation: float = DEFAULT_ELEVATION,
    num_frames: int = SV3D_NUM_FRAMES,
) -> tuple[list[float], list[float]]:
    """Compute polar and azimuth angles matching SV3D's reference infer.py."""
    elevations_deg = [elevation] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]

    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [
        np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg
    ]
    azimuths_rad[:-1].sort()

    return polars_rad, azimuths_rad


class SV3DSynthesizer(BaseSynthesizer):
    name = "sv3d"

    def __init__(self):
        self.pipe = None
        self.device = "cpu"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.pipe is not None:
            logger.info("SV3D pipeline already loaded — skipping.")
            return

        _ensure_vendor_on_path()
        from diffusers_sv3d import (
            SV3DUNetSpatioTemporalConditionModel,
            StableVideo3DDiffusionPipeline,
        )
        from diffusers import AutoencoderKL, EulerDiscreteScheduler
        from transformers import (
            CLIPImageProcessor,
            CLIPVisionModelWithProjection,
        )

        self.device = device
        logger.info(f"Loading SV3D pipeline from {HF_MODEL_ID} …")

        unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(
            HF_MODEL_ID, subfolder="unet"
        )
        vae = AutoencoderKL.from_pretrained(HF_MODEL_ID, subfolder="vae")
        scheduler = EulerDiscreteScheduler.from_pretrained(
            HF_MODEL_ID, subfolder="scheduler"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            HF_MODEL_ID, subfolder="image_encoder"
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            HF_MODEL_ID, subfolder="feature_extractor"
        )

        self.pipe = StableVideo3DDiffusionPipeline(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to(device)
        logger.info("SV3D pipeline loaded.")

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
                "SV3D model not loaded. Call load_model() first."
            )

        if prompt:
            logger.info("SV3D is image-conditioned only — ignoring prompt.")

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")

        logger.info("Preprocessing image (background removal + resize) …")
        input_image = _preprocess_image(img)

        polars_rad, azimuths_rad = _build_camera_params()

        logger.info(
            f"Running SV3D inference ({SV3D_NUM_FRAMES} orbital frames) …"
        )
        with torch.no_grad():
            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=True
            ):
                frames = self.pipe(
                    input_image,
                    height=SV3D_RES,
                    width=SV3D_RES,
                    num_frames=SV3D_NUM_FRAMES,
                    decode_chunk_size=8,
                    polars_rad=polars_rad,
                    azimuths_rad=azimuths_rad,
                    generator=torch.manual_seed(42),
                ).frames[0]

        if num_views >= SV3D_NUM_FRAMES:
            indices = list(range(SV3D_NUM_FRAMES))
        else:
            indices = [
                round(i * (SV3D_NUM_FRAMES - 1) / (num_views - 1))
                for i in range(num_views)
            ]

        saved: list[str] = []
        for idx in indices:
            out_path = views_dir / f"view_{idx:03d}.png"
            frames[idx].save(str(out_path))
            saved.append(str(out_path))
            logger.info(f"  saved frame {idx} → {out_path.name}")

        logger.info(f"SV3D: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logger.info("SV3D pipeline unloaded.")
