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
import json
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


DEFAULT_NEUTRAL_PROMPT = "a photo of a living room interior"
DEFAULT_CLIP_LAMBDA = 0.2
DEFAULT_NUM_STEPS = 50

# Allowed --dtype aliases -> (torch dtype, SEVA_AUTOCAST_DTYPE env string)
_DTYPE_ALIASES: dict[str, tuple[torch.dtype, str]] = {
    "fp32": (torch.float32, "float16"),  # autocast still fp16 by default
    "float32": (torch.float32, "float16"),
    "fp16": (torch.float16, "float16"),
    "float16": (torch.float16, "float16"),
    "half": (torch.float16, "float16"),
    "bf16": (torch.bfloat16, "bfloat16"),
    "bfloat16": (torch.bfloat16, "bfloat16"),
}


class SevaSynthesizer(BaseSynthesizer):
    name = "seva"

    def __init__(
        self,
        neutral_prompt: str = DEFAULT_NEUTRAL_PROMPT,
        clip_lambda: float = DEFAULT_CLIP_LAMBDA,
        num_steps: int = DEFAULT_NUM_STEPS,
        dtype: str = "fp32",
    ):
        self.model = None
        self.ae = None
        self.conditioner = None
        self.denoiser = None
        self.device = "cpu"
        self._compile = False
        self.neutral_prompt = neutral_prompt
        self.clip_lambda = clip_lambda
        self.num_steps = int(num_steps)
        if dtype not in _DTYPE_ALIASES:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Allowed: {sorted(_DTYPE_ALIASES)}"
            )
        self.dtype_name = dtype
        self.torch_dtype, self._autocast_env = _DTYPE_ALIASES[dtype]

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

        # Tell seva.eval which autocast dtype to use (defaults to fp16).
        os.environ["SEVA_AUTOCAST_DTYPE"] = self._autocast_env

        logger.info(
            f"Loading SEVA model (v1.1) from {HF_MODEL_ID} "
            f"(weights dtype={self.dtype_name}, autocast={self._autocast_env}) …"
        )
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

        # Cast the big denoiser backbone to the requested dtype.  The AE and
        # CLIP conditioner are small and numerically sensitive, so we keep
        # them in fp32 and rely on autocast for those ops.
        if self.torch_dtype != torch.float32:
            logger.info(
                "Casting SEVA backbone weights to %s (saves per-op cast "
                "overhead and keeps FLASH kernels engaged).",
                self.torch_dtype,
            )
            self.model = self.model.to(self.torch_dtype)

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
        neutral_prompt: Optional[str] = None,
        clip_lambda: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> list[str]:
        """
        Generate novel views with optional CLIP-text re-conditioning.

        When ``prompt`` is provided, a unit-norm direction
        ``encode_text(prompt) - encode_text(neutral_prompt)`` is added to the
        pooled CLIP image embedding before it enters SEVA's cross-attention:

            z_tilde = z_img + clip_lambda * ||z_img|| * delta

        ``neutral_prompt`` and ``clip_lambda`` default to the synthesizer's
        instance-level defaults (``self.neutral_prompt`` / ``self.clip_lambda``).
        """
        if self.model is None:
            raise RuntimeError("SEVA model not loaded. Call load_model() first.")

        eff_neutral = neutral_prompt if neutral_prompt is not None else self.neutral_prompt
        eff_lambda = float(clip_lambda) if clip_lambda is not None else float(self.clip_lambda)

        clip_direction_active = False
        if prompt:
            try:
                delta = self.conditioner.encode_text_direction(prompt, eff_neutral)
                self.conditioner.set_direction(delta, eff_lambda)
                clip_direction_active = True
                logger.info(
                    "SEVA CLIP text-reconditioning ON | prompt=%r neutral=%r lambda=%.3f",
                    prompt,
                    eff_neutral,
                    eff_lambda,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to set CLIP text direction (%s); falling back to image-only.",
                    exc,
                )
                clip_direction_active = False

        _ensure_vendor_on_path()

        eff_num_steps = int(num_steps) if num_steps is not None else int(self.num_steps)

        try:
            return self._run_generation(
                image_path=image_path,
                output_dir=output_dir,
                num_views=num_views,
                num_steps=eff_num_steps,
            )
        finally:
            if clip_direction_active and self.conditioner is not None:
                self.conditioner.set_direction(None, 0.0)

    def _run_generation(
        self,
        image_path: str,
        output_dir: str,
        num_views: int,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> list[str]:
        from seva.eval import (
            infer_prior_stats,
            run_one_scene,
        )
        from seva.geometry import (
            get_preset_pose_fov,
            get_default_intrinsics,
        )

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
                "num_steps": int(num_steps),
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

        # Persist the full camera trajectory so downstream tooling (report
        # figures, offline metrics, alternate viewers) can reproduce the orbit
        # without re-running SEVA.  c2ws is (T, 4, 4), Ks is (T, 3, 3).
        try:
            c2ws_np = c2ws.cpu().numpy() if hasattr(c2ws, "cpu") else np.asarray(c2ws)
            fovs_np = fovs.cpu().numpy() if hasattr(fovs, "cpu") else np.asarray(fovs)
            trajectory = {
                "backend": "seva",
                "task": "img2trajvid_s-prob",
                "traj_prior": "orbit",
                "fov_deg_input": float(fov_deg) if fov_deg is not None else None,
                "aspect_ratio": float(aspect_ratio),
                "num_targets": int(num_targets),
                "num_frames": int(T),
                "num_anchors": int(num_anchors),
                "anchor_indices": [float(i) for i in anchor_indices],
                "fovs_rad": fovs_np.tolist(),
                "c2ws": c2ws_np.tolist(),
                "Ks": Ks.tolist(),
            }
            (views_dir / "trajectory.json").write_text(
                json.dumps(trajectory, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SEVA: failed to write views/trajectory.json: %s", exc)

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
            f"{num_targets + 1} frames, num_steps={num_steps}, "
            f"dtype={self.dtype_name}) …"
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

        # SEVA writes PNGs to different subdirectories depending on the
        # scene mode: historically ``samples-rgb/*.png`` right under
        # ``save_path`` but newer releases use ``first-pass/samples-rgb/*.png``.
        # Search both shapes (and any deeper ``samples-rgb`` folder) so future
        # layout changes don't silently produce empty ``views/``.
        search_roots = [
            os.path.join(save_path, "samples-rgb", "*.png"),
            os.path.join(save_path, "**", "samples-rgb", "*.png"),
            os.path.join(save_path, "*.png"),
        ]
        raw_frames: list[str] = []
        for pattern in search_roots:
            raw_frames = sorted(glob.glob(pattern, recursive=True))
            if raw_frames:
                break

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
