"""
Novel View Synthesis via Stable Virtual Camera (SEVA).

SEVA is a 1.3B-parameter generalist diffusion model for Novel View Synthesis
that generates 3D-consistent novel views given input views and target cameras.
We use the ``img2trajvid_s-prob`` task for single-image orbit generation.

Requires:
  - ``backend/vendor/stable-virtual-camera/`` cloned (with --recursive) from
    https://github.com/Stability-AI/stable-virtual-camera
  - HuggingFace authentication (``huggingface-cli login``) for gated weights
  - python >= 3.10, torch >= 2.6.0
  - flash-attn (Linux / WSL only -- not supported on native Windows)
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "stable-virtual-camera"
HF_MODEL_ID = "stabilityai/stable-virtual-camera"
SEVA_RES = 576
SEVA_DEFAULT_FRAMES = 21


def _ensure_vendor_on_path() -> None:
    seva_root = str(VENDOR_DIR)
    if seva_root not in sys.path:
        sys.path.insert(0, seva_root)


class SevaSynthesizer(BaseSynthesizer):
    name = "seva"

    def __init__(self):
        self.model = None
        self.ae = None
        self.conditioner = None
        self.denoiser = None
        self.device = "cpu"
        self._compile = False

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.model is not None:
            logger.info("SEVA already loaded — skipping.")
            return

        _ensure_vendor_on_path()

        from seva.model import SGMWrapper
        from seva.modules.autoencoder import AutoEncoder
        from seva.modules.conditioner import CLIPConditioner
        from seva.sampling import DiscreteDenoiser
        from seva.utils import load_model as seva_load_model
        from seva.eval import IS_TORCH_NIGHTLY

        self.device = device
        self._compile = IS_TORCH_NIGHTLY

        if self._compile:
            os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

        logger.info(f"Loading SEVA model (v1.1) from {HF_MODEL_ID} …")
        self.model = SGMWrapper(
            seva_load_model(
                model_version=1.1,
                pretrained_model_name_or_path=HF_MODEL_ID,
                weight_name="model.safetensors",
                device="cpu",
                verbose=True,
            ).eval()
        ).to(device)

        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.conditioner = CLIPConditioner().to(device)
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)

        if self._compile:
            self.model = torch.compile(self.model, dynamic=False)
            self.conditioner = torch.compile(self.conditioner, dynamic=False)
            self.ae = torch.compile(self.ae, dynamic=False)

        logger.info("SEVA model loaded.")

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
            raise RuntimeError("SEVA model not loaded. Call load_model() first.")

        if prompt:
            logger.info("SEVA is image-conditioned only — ignoring prompt.")

        _ensure_vendor_on_path()

        from seva.eval import infer_prior_stats, run_one_scene
        from seva.geometry import get_preset_pose_fov, get_default_intrinsics

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        num_targets = max(num_views - 1, SEVA_DEFAULT_FRAMES - 1)
        T = num_targets + 1

        version_dict = {
            "H": SEVA_RES,
            "W": SEVA_RES,
            "T": T,
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": "interp",
                "video_save_fps": 30.0,
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": 1,
                "cfg": 4.0,
                "camera_scale": 2.0,
                "num_steps": 50,
                "cfg_min": 1.2,
                "encoding_t": 1,
                "decoding_t": 1,
                "traj_prior": "orbit",
            },
        }

        options = version_dict["options"]

        num_inputs = 1
        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()
        all_imgs_path = [image_path] + [None] * num_targets

        c2ws, fovs = get_preset_pose_fov(
            option="orbit",
            num_frames=num_targets + 1,
            start_w2c=torch.eye(4),
            look_at=torch.Tensor([0, 0, 10]),
        )

        with Image.open(image_path) as img:
            W_img, H_img = img.size
        aspect_ratio = W_img / H_img
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)
        Ks[:, :2] *= (
            torch.tensor([W_img, H_img]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )
        Ks = Ks.numpy()

        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

        save_path = str(views_dir / "_seva_work")

        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        camera_cond = {
            "c2w": torch.tensor(c2ws[:, :3]).float().clone(),
            "K": torch.tensor(Ks).float().clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }

        logger.info(
            f"Running SEVA inference (img2trajvid_s-prob, orbit, "
            f"{num_targets + 1} frames) …"
        )
        video_path_generator = run_one_scene(
            "img2trajvid_s-prob",
            version_dict,
            model=self.model,
            ae=self.ae,
            conditioner=self.conditioner,
            denoiser=self.denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=save_path,
            use_traj_prior=True,
            traj_prior_Ks=torch.tensor(anchor_Ks).float(),
            traj_prior_c2ws=torch.tensor(anchor_c2ws[:, :3]).float(),
            seed=23,
        )
        for _ in video_path_generator:
            pass

        raw_frames = sorted(glob.glob(os.path.join(save_path, "samples-rgb", "*.png")))
        if not raw_frames:
            raw_frames = sorted(glob.glob(os.path.join(save_path, "*.png")))

        if num_views >= len(raw_frames):
            indices = list(range(len(raw_frames)))
        else:
            indices = [
                round(i * (len(raw_frames) - 1) / (num_views - 1))
                for i in range(num_views)
            ]

        saved: list[str] = []
        for out_idx, src_idx in enumerate(indices):
            out_path = views_dir / f"view_{out_idx:03d}.png"
            img = Image.open(raw_frames[src_idx])
            img.save(str(out_path))
            saved.append(str(out_path))

        logger.info(f"SEVA: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        for attr in ("model", "ae", "conditioner", "denoiser"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        torch.cuda.empty_cache()
        logger.info("SEVA pipeline unloaded.")
