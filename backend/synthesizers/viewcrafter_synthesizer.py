"""
Novel View Synthesis via ViewCrafter.

ViewCrafter uses point-cloud-guided video diffusion to synthesize novel views
with **explicit camera trajectory control** (d_phi, d_theta, d_r).  It runs
DUSt3R internally to build a point cloud from the input, renders the cloud
along the requested trajectory, and then refines the renders with a diffusion
model.

Requires:
  - ``backend/vendor/ViewCrafter/`` cloned from
    https://github.com/Drexubery/ViewCrafter
  - ``backend/vendor/ViewCrafter/checkpoints/model.ckpt``
    (auto-downloaded from HuggingFace ``Drexubery/ViewCrafter_25``)
  - ``backend/vendor/ViewCrafter/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth``
  - pytorch3d installed
"""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "ViewCrafter"
CHECKPOINT_DIR = VENDOR_DIR / "checkpoints"


def _ensure_checkpoint() -> str:
    """Download the ViewCrafter diffusion checkpoint if missing."""
    ckpt = CHECKPOINT_DIR / "model.ckpt"
    if ckpt.exists():
        return str(ckpt)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ViewCrafter checkpoint from HuggingFace …")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="Drexubery/ViewCrafter_25",
        filename="model.ckpt",
        local_dir=str(CHECKPOINT_DIR),
        force_download=True,
    )
    return str(ckpt)


def _build_opts(
    image_path: str,
    save_dir: str,
    device: str,
    d_phi: float = 30.0,
    d_theta: float = 0.0,
    d_r: float = -0.2,
    elevation: float = 5.0,
    center_scale: float = 1.0,
    ddim_steps: int = 50,
    video_length: int = 25,
    prompt: str = "Rotating view of a scene",
) -> types.SimpleNamespace:
    """Build a namespace that mimics ViewCrafter's argparse opts."""
    ckpt_path = _ensure_checkpoint()
    dust3r_path = str(CHECKPOINT_DIR / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")

    return types.SimpleNamespace(
        image_dir=image_path,
        out_dir=save_dir,
        save_dir=save_dir,
        device=device,
        exp_name=None,
        # renderer
        mode="single_view_target",
        traj_txt=None,
        elevation=elevation,
        center_scale=center_scale,
        d_theta=[d_theta],
        d_phi=[d_phi],
        d_r=[d_r],
        d_x=[0.0],
        d_y=[0.0],
        mask_image=False,
        mask_pc=True,
        reduce_pc=False,
        bg_trd=0.0,
        dpt_trd=1.0,
        # diffusion
        ckpt_path=ckpt_path,
        config=str(VENDOR_DIR / "configs" / "inference_pvd_1024.yaml"),
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        bs=1,
        height=576,
        width=1024,
        frame_stride=10,
        unconditional_guidance_scale=7.5,
        seed=123,
        video_length=video_length,
        negative_prompt=False,
        text_input=True,
        prompt=prompt,
        multiple_cond_cfg=False,
        cfg_img=None,
        timestep_spacing="uniform_trailing",
        guidance_rescale=0.7,
        perframe_ae=True,
        n_samples=1,
        # dust3r
        model_path=dust3r_path,
        batch_size=1,
        schedule="linear",
        niter=300,
        lr=0.01,
        min_conf_thr=3.0,
    )


class ViewCrafterSynthesizer(BaseSynthesizer):
    name = "viewcrafter"

    def __init__(self):
        self.model = None
        self.device = "cuda"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.model is not None:
            logger.info("ViewCrafter already loaded — skipping.")
            return

        vc_root = str(VENDOR_DIR)
        if vc_root not in sys.path:
            sys.path.insert(0, vc_root)

        dust3r_root = str(VENDOR_DIR / "extern" / "dust3r")
        if dust3r_root not in sys.path:
            sys.path.insert(0, dust3r_root)

        self.device = device
        logger.info("ViewCrafter model will be instantiated on first generate_views() call.")

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
        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        vc_root = str(VENDOR_DIR)
        if vc_root not in sys.path:
            sys.path.insert(0, vc_root)
        dust3r_root = str(VENDOR_DIR / "extern" / "dust3r")
        if dust3r_root not in sys.path:
            sys.path.insert(0, dust3r_root)

        # ViewCrafter bundles its own DUSt3R (extern/dust3r) whose API differs
        # from the project's vendor/dust3r.  Temporarily evict the project's
        # cached dust3r modules so Python resolves the bundled version instead.
        stashed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "dust3r" or k.startswith("dust3r.")}
        try:
            from viewcrafter import ViewCrafter
        finally:
            # Restore the project's dust3r so Dust3rReconstructor keeps working.
            # ViewCrafter's own references are already bound to the correct objects.
            sys.modules.update(stashed)

        opts = _build_opts(
            image_path=image_path,
            save_dir=str(views_dir),
            device=self.device,
            d_phi=30.0,
            d_theta=0.0,
            d_r=-0.2,
            video_length=num_views,
            prompt=prompt or "Rotating view of a scene",
        )

        logger.info(f"Running ViewCrafter (single_view_target, {num_views} frames) …")
        vc = ViewCrafter(opts, gradio=False)
        diffusion_result = vc.nvs_single_view()
        # diffusion_result: Tensor [num_frames, H, W, 3] in [-1, 1]

        frames = ((diffusion_result + 1.0) / 2.0).clamp(0, 1)
        frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)

        saved: list[str] = []
        for i, frame in enumerate(frames_np):
            out_path = views_dir / f"view_{i:03d}.png"
            Image.fromarray(frame).save(str(out_path))
            saved.append(str(out_path))

        logger.info(f"ViewCrafter: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
        logger.info("ViewCrafter unloaded.")
