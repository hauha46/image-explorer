"""
Novel View Synthesis via PanoDreamer (360° Cylindrical Panorama).

PanoDreamer generates a coherent 360° panorama from a single image using
iterative inpainting with MultiConDiffusion in cylindrical space.  We then
extract N evenly-spaced perspective crops and hand them to DUSt3R for 3D
reconstruction — fitting the standard BaseSynthesizer interface.

Requires:
  - ``backend/vendor/PanoDreamer/`` cloned from
    https://github.com/avinashpaliwal/PanoDreamer
  - kornia (geometric transforms)
  - Stable Diffusion 2 Inpainting weights auto-download from HuggingFace
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "PanoDreamer"

PANO_HEIGHT = 512
PANO_WIDTH = 3912
INTERNAL_FOV_DEG = 44.701948991275390
DEFAULT_STEPS = 50
DEFAULT_ITERATIONS = 15
DEFAULT_GUIDANCE = 7.5
DEFAULT_NEGATIVE = (
    "caption, subtitle, text, blur, lowres, bad anatomy, "
    "bad hands, cropped, worst quality, watermark"
)


def _ensure_vendor_on_path() -> None:
    """Put the PanoDreamer vendor root on sys.path for bare imports."""
    pd_root = str(VENDOR_DIR)
    if pd_root not in sys.path:
        sys.path.insert(0, pd_root)


def _fov2focal(fov_radians: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov_radians / 2))


def _extract_perspective_views(
    panorama: Image.Image,
    num_views: int,
    device: torch.device,
) -> list[Image.Image]:
    """
    Extract evenly-spaced perspective crops from a cylindrical panorama.

    For each target azimuth, a 512-pixel-wide strip is taken from the
    panorama (with wraparound), then projected from cylindrical to
    perspective coordinates using PanoDreamer's ``cyl_proj`` transform.
    """
    _ensure_vendor_on_path()
    from multicondiffusion_panorama import cyl_proj  # type: ignore[import-not-found]

    pano_np = np.array(panorama.convert("RGB"))
    H, W = pano_np.shape[:2]

    focal = _fov2focal(math.radians(INTERNAL_FOV_DEG), PANO_HEIGHT)
    strip_w = PANO_HEIGHT  # perspective view width in cylindrical space (512)

    views: list[Image.Image] = []
    for i in range(num_views):
        center_x = int(i * W / num_views)
        half = strip_w // 2

        cols = np.arange(center_x - half, center_x + half) % W
        strip = pano_np[:, cols, :]  # H x strip_w x 3

        strip_tensor = (
            T.ToTensor()(strip).unsqueeze(0).to(device).float()
        )

        with torch.no_grad():
            persp = cyl_proj(strip_tensor, focal)

        persp_img = T.ToPILImage()(persp[0].cpu().clamp(0, 1))
        views.append(persp_img)

    return views


class PanoDreamerSynthesizer(BaseSynthesizer):
    name = "panodreamer"

    def __init__(self):
        self.model = None
        self.device = "cpu"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.model is not None:
            logger.info("PanoDreamer model already loaded — skipping.")
            return

        _ensure_vendor_on_path()
        from multicondiffusion_panorama import (  # type: ignore[import-not-found]
            CylindricalPanorama,
            seed_everything,
        )

        self.device = device
        seed_everything(42)
        logger.info("Loading PanoDreamer CylindricalPanorama model …")
        self.model = CylindricalPanorama(device=torch.device(device))
        logger.info("PanoDreamer model loaded.")

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
        if self.model is None:
            raise RuntimeError(
                "PanoDreamer model not loaded. Call load_model() first."
            )

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        input_image = Image.open(image_path).convert("RGB")
        scene_name = Path(image_path).stem

        text_prompt = prompt or "A detailed photo of a scene"

        logger.info(
            f"Running PanoDreamer panorama generation "
            f"(iterations={DEFAULT_ITERATIONS}, steps={DEFAULT_STEPS}) …"
        )

        pano_save_dir = str(Path(output_dir) / "panodreamer_intermediates")
        panorama = self.model.image_to_cylindrical_panorama(
            scene=scene_name,
            input_image=input_image,
            prompt=text_prompt,
            negative_prompt=DEFAULT_NEGATIVE,
            height=PANO_HEIGHT,
            width=PANO_WIDTH,
            num_inference_steps=DEFAULT_STEPS,
            guidance_scale=DEFAULT_GUIDANCE,
            num_iterations=DEFAULT_ITERATIONS,
            save_dir=pano_save_dir,
        )

        pano_path = views_dir / "panorama_360.png"
        panorama.save(str(pano_path))
        logger.info(f"360° panorama saved to {pano_path}")

        logger.info(
            f"Extracting {num_views} perspective views from panorama …"
        )
        perspective_views = _extract_perspective_views(
            panorama, num_views, torch.device(self.device)
        )

        saved: list[str] = []
        for idx, view in enumerate(perspective_views):
            out_path = views_dir / f"view_{idx:03d}.png"
            view.save(str(out_path))
            saved.append(str(out_path))
            logger.info(f"  saved view {idx} → {out_path.name}")

        logger.info(f"PanoDreamer: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("PanoDreamer model unloaded.")
