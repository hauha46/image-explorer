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

import json
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
    """Download the ViewCrafter 512 diffusion checkpoint if missing."""
    ckpt = CHECKPOINT_DIR / "model_512.ckpt"
    if ckpt.exists():
        return str(ckpt)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ViewCrafter 512 checkpoint from HuggingFace …")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="Drexubery/ViewCrafter_25_512",
        filename="model.ckpt",
        local_dir=str(CHECKPOINT_DIR),
        local_dir_use_symlinks=False,
    )
    # Rename to model_512.ckpt to distinguish from 1024 version
    import shutil
    shutil.move(str(CHECKPOINT_DIR / "model.ckpt"), str(ckpt))
    return str(ckpt)


def _build_opts(
    image_path: str,
    save_dir: str,
    device: str,
    d_phi: float = 20.0,
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
        bg_trd=0.2,
        dpt_trd=0.7,
        # diffusion
        ckpt_path=ckpt_path,
        config=str(VENDOR_DIR / "configs" / "inference_pvd_512.yaml"),
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        bs=1,
        height=320,
        width=512,
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
        # Populated by generate_views() so callers (e.g. backend/app.py) can
        # log the exact configuration used for a given session.
        self.last_run_params: Optional[dict] = None

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
            video_length=num_views,
            prompt=prompt or "Rotating view of a scene",
        )

        # Stash the effective config so backend/app.py can log it in
        # run_report.txt / run_info.json / sessions_index.jsonl.  Mirrors the
        # approach SEVA takes with dtype_name / num_steps.
        self.last_run_params = {
            "backend": "viewcrafter",
            "mode": opts.mode,
            "video_length": int(opts.video_length),
            "prompt": opts.prompt,
            "trajectory": {
                "d_phi": list(opts.d_phi),
                "d_theta": list(opts.d_theta),
                "d_r": list(opts.d_r),
                "d_x": list(opts.d_x),
                "d_y": list(opts.d_y),
                "elevation": float(opts.elevation),
                "center_scale": float(opts.center_scale),
            },
            "diffusion": {
                "ddim_steps": int(opts.ddim_steps),
                "ddim_eta": float(opts.ddim_eta),
                "cfg": float(opts.unconditional_guidance_scale),
                "guidance_rescale": float(opts.guidance_rescale),
                "frame_stride": int(opts.frame_stride),
                "seed": int(opts.seed),
                "height": int(opts.height),
                "width": int(opts.width),
                "timestep_spacing": opts.timestep_spacing,
                "perframe_ae": bool(opts.perframe_ae),
                "n_samples": int(opts.n_samples),
            },
            "point_cloud": {
                "bg_trd": float(opts.bg_trd),
                "dpt_trd": float(opts.dpt_trd),
                "mask_pc": bool(opts.mask_pc),
                "reduce_pc": bool(opts.reduce_pc),
                "mask_image": bool(opts.mask_image),
            },
            "checkpoints": {
                "diffusion_ckpt": opts.ckpt_path,
                "dust3r_ckpt": opts.model_path,
                "config": opts.config,
            },
            "dust3r": {
                "batch_size": int(opts.batch_size),
                "schedule": opts.schedule,
                "niter": int(opts.niter),
                "lr": float(opts.lr),
                "min_conf_thr": float(opts.min_conf_thr),
            },
        }

        # Dump parametric trajectory descriptor for parity with SEVA's
        # per-frame c2ws/Ks trajectory.json (ViewCrafter's public API does
        # not expose the computed per-frame poses).
        try:
            (views_dir / "trajectory.json").write_text(
                json.dumps(
                    {
                        "backend": "viewcrafter",
                        "mode": opts.mode,
                        "video_length": int(opts.video_length),
                        "d_phi": list(opts.d_phi),
                        "d_theta": list(opts.d_theta),
                        "d_r": list(opts.d_r),
                        "d_x": list(opts.d_x),
                        "d_y": list(opts.d_y),
                        "elevation": float(opts.elevation),
                        "center_scale": float(opts.center_scale),
                        "height": int(opts.height),
                        "width": int(opts.width),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ViewCrafter: failed to write views/trajectory.json: %s", exc)

        logger.info(f"Running ViewCrafter (single_view_target, {num_views} frames) …")
        vc = ViewCrafter(opts, gradio=False)
        diffusion_result = vc.nvs_single_view()
        # diffusion_result: Tensor [num_frames, H, W, 3] in [-1, 1]

        frames = ((diffusion_result + 1.0) / 2.0).clamp(0, 1)
        frames_np = (frames.cpu().float().numpy() * 255).astype(np.uint8)

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
