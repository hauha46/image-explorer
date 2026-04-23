"""
Novel View Synthesis via VIVID (Apple).

VIVID performs end-to-end NVS in pixel space using diffusion, conditioned on
the source image and a relative camera transformation (geometry label).

Requires:
  - ``backend/vendor/ml-vivid/`` cloned from https://github.com/apple/ml-vivid
  - Pretrained weights auto-download from Apple CDN on first use.
  - ``torch.distributed`` initialised (the wrapper handles single-GPU init).
  - DepthAnythingV2 installed for depth-conditioned variant (optional).

The wrapper programmatically generates target camera poses (a horizontal arc)
so it can work from a single input image without a RealEstate10K dataset.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "ml-vivid"

MODEL_URLS = {
    "base": "https://ml-site.cdn-apple.com/models/vivid/vivid-base.pkl",
    "uncond": "https://ml-site.cdn-apple.com/models/vivid/vivid-uncond.pkl",
    "sr": "https://ml-site.cdn-apple.com/models/vivid/vivid-sr.pkl",
}

GUIDANCE = 1.5


# ── Camera pose helpers ──────────────────────────────────────────────
def _rotation_y(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around the Y axis."""
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _make_K(focal_px: float, img_w: int, img_h: int) -> torch.Tensor:
    """Build a 3x3 intrinsic matrix from focal length and image size."""
    K = torch.zeros(3, 3)
    K[0, 0] = focal_px           # fx
    K[1, 1] = focal_px           # fy  (square pixels)
    K[0, 2] = img_w / 2.0        # cx
    K[1, 2] = img_h / 2.0        # cy
    K[2, 2] = 1.0
    return K


def _make_relative_geometry(
    angle_deg: float,
    translation_scale: float = 0.15,
    focal_px: float = 500.0,
    img_w: int = 64,
    img_h: int = 64,
) -> torch.Tensor:
    """
    Build a 20-float geometry label using VIVID's compose_geometry().

    The model expects: [tgt2src_extrinsics(12), src_intrinsics(4), tgt_intrinsics(4)]
    normalised via precomputed MEAN/STD statistics from training/utils.py.

    tgt2src is the relative transform that maps target camera coords to source
    camera coords (i.e. src_pose^{-1} @ tgt_pose, or just R|t when src=identity).
    """
    vivid_root = str(VENDOR_DIR)
    if vivid_root not in sys.path:
        sys.path.insert(0, vivid_root)
    from training.utils import compose_geometry

    R = torch.from_numpy(_rotation_y(angle_deg))   # 3x3
    t = torch.tensor(
        [math.sin(math.radians(angle_deg)) * translation_scale, 0.0, 0.0]
    ).reshape(3, 1)
    tgt2src = torch.cat([R, t], dim=1).unsqueeze(0)  # [1, 3, 4]

    K = _make_K(focal_px, img_w, img_h).unsqueeze(0)  # [1, 3, 3]

    geom = compose_geometry(tgt2src, K, K, imsize=img_w)  # [1, 20]
    return geom.squeeze(0)  # [20]


def _generate_pose_arc(
    num_views: int,
    max_angle: float = 30.0,
    focal_px: float = 500.0,
    img_w: int = 64,
    img_h: int = 64,
) -> list[torch.Tensor]:
    """Generate *num_views* geometry labels spanning [-max_angle, +max_angle]."""
    if num_views == 1:
        return [_make_relative_geometry(0.0, focal_px=focal_px, img_w=img_w, img_h=img_h)]
    angles = np.linspace(-max_angle, max_angle, num_views)
    return [
        _make_relative_geometry(float(a), focal_px=focal_px, img_w=img_w, img_h=img_h)
        for a in angles
    ]


# ── Synthesizer ──────────────────────────────────────────────────────
class VIVIDSynthesizer(BaseSynthesizer):
    name = "vivid"

    def __init__(self):
        self.net = None
        self.gnet = None
        self.sr_model = None
        self.encoder = None
        self.device = "cuda"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.net is not None:
            logger.info("VIVID models already loaded — skipping.")
            return

        vivid_root = str(VENDOR_DIR)
        if vivid_root not in sys.path:
            sys.path.insert(0, vivid_root)

        # VIVID uses torch.distributed internally; init for single-GPU
        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        import dnnlib  # noqa: from ml-vivid

        self.device = device
        logger.info("Loading VIVID base model …")
        with dnnlib.util.open_url(MODEL_URLS["base"]) as f:
            data = pickle.load(f)
        self.net = data[("ema" if "ema" in data else "net")].to(device)
        self.encoder = data.get("encoder", None)
        if self.encoder is None:
            from training.encoders import StandardRGBEncoder
            self.encoder = StandardRGBEncoder()
        self.encoder.init(device)
        del data

        logger.info("Loading VIVID unconditional (guidance) model …")
        with dnnlib.util.open_url(MODEL_URLS["uncond"]) as f:
            data = pickle.load(f)
        self.gnet = data[("ema" if "ema" in data else "net")].to(device)
        del data

        logger.info("Loading VIVID super-resolution model …")
        with dnnlib.util.open_url(MODEL_URLS["sr"]) as f:
            data = pickle.load(f)
        self.sr_model = data[("ema" if "ema" in data else "net")].to(device)
        del data

        logger.info("VIVID models loaded.")

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
        if self.net is None:
            raise RuntimeError("VIVID models not loaded. Call load_model() first.")

        vivid_root = str(VENDOR_DIR)
        if vivid_root not in sys.path:
            sys.path.insert(0, vivid_root)

        from generate_images import edm_sampler, StackedRandomGenerator

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        res = self.net.img_resolution  # typically 64 (base) or 256 (SR)
        img_tensor = torch.from_numpy(
            np.array(img.resize((res, res), Image.LANCZOS))
        ).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        # encoder expects [0, 255] range
        src_latent = self.encoder.encode_latents(img_tensor)

        focal_px = res / 2.0  # neutral default: 90-degree FOV
        if fov_deg is not None and fov_deg > 0:
            focal_px = (res / 2.0) / math.tan(math.radians(fov_deg / 2.0))

        poses = _generate_pose_arc(num_views, max_angle=25.0, focal_px=focal_px, img_w=res, img_h=res)

        saved: list[str] = []
        logger.info(f"Running VIVID inference for {num_views} target views …")

        for i, geom in enumerate(poses):
            labels = geom.unsqueeze(0).to(self.device)
            rnd = StackedRandomGenerator(self.device, [42 + i])
            noise = rnd.randn(
                [1, self.net.img_channels, res, res], device=self.device
            )

            with torch.no_grad():
                latents = edm_sampler(
                    net=self.net,
                    src=src_latent,
                    noise=noise,
                    labels=labels,
                    gnet=self.gnet,
                    guidance=GUIDANCE,
                    randn_like=rnd.randn_like,
                )
                result = self.encoder.decode(latents)

            frame = result[0].permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            out_path = views_dir / f"view_{i:03d}.png"
            Image.fromarray(frame).save(str(out_path))
            saved.append(str(out_path))
            logger.info(f"  view {i} (angle {poses[i]})")

        logger.info(f"VIVID: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        for attr in ("net", "gnet", "sr_model"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self.encoder = None
        torch.cuda.empty_cache()
        logger.info("VIVID models unloaded.")
