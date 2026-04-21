"""
Novel View Synthesis via Stable Video 3D (SV3D_p).

SV3D generates a 21-frame video clip representing a 360-degree panning 
shot around a single object.

Requires:
  - backend/vendor1/generative-models/ (Stability AI SGM repo)
  - backend/vendor1/generative-models/checkpoints/sv3d_p.safetensors
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat
from rembg import remove

from .base import BaseSynthesizer
from torchvision.transforms import ToTensor


logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor1" / "generative-models"
CHECKPOINT_PATH = VENDOR_DIR / "checkpoints" / "sv3d_p.safetensors"
CONFIG_PATH = VENDOR_DIR / "scripts" / "sampling" / "configs" / "sv3d_p.yaml"

class SV3DSynthesizer(BaseSynthesizer):
    name = "sv3d"

    def __init__(self):
        self.model = None
        self.device = "cuda"

    # ------------------------------------------------------------------
    def load_model(self, device: str = "cuda") -> None:
        if self.model is not None:
            logger.info("SV3D model already loaded — skipping.")
            return

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Missing SV3D checkpoint at {CHECKPOINT_PATH}")

        gm_root = str(VENDOR_DIR)
        if gm_root not in sys.path:
            sys.path.insert(0, gm_root)

        from omegaconf import OmegaConf
        from sgm.util import instantiate_from_config

        logger.info("Loading SV3D_p configuration …")
        config = OmegaConf.load(str(CONFIG_PATH))
        # Override checkpoint path to absolute
        config.model.params.ckpt_path = str(CHECKPOINT_PATH)
        
        # Set openclip device
        if device == "cuda":
            config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device

        config.model.params.sampler_config.params.verbose = False
        config.model.params.sampler_config.params.num_steps = 20
        config.model.params.sampler_config.params.guider_config.params.num_frames = 21

        logger.info(f"Loading SV3D_p model from {CHECKPOINT_PATH} onto {device} …")
        self.device = device
        
        # Load model using SGM, instantiate on CPU and cast to fp16 to save ~4GB of VRAM
        self.model = instantiate_from_config(config.model).eval()
        self.model.to(torch.float16)

        logger.info("SV3D_p model loaded successfully in FP16.")

    # ------------------------------------------------------------------
    def _get_batch(self, keys, value_dict, N, T, device):
        batch = {}
        batch_uc = {}
        for key in keys:
            if key == "fps_id" or key == "motion_bucket_id":
                batch[key] = torch.tensor([value_dict[key]]).to(device).repeat(int(math.prod(N)))
            elif key == "cond_aug":
                batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N))
            elif key in ["cond_frames", "cond_frames_without_noise"]:
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            elif key in ["polars_rad", "azimuths_rad"]:
                batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def generate_views(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 21,
        depth_map: Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> list[str]:
        if self.model is None:
            raise RuntimeError("SV3D model not loaded. Call load_model() first.")

        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing input image for SV3D …")
        image = Image.open(image_path).convert("RGB")

        # Bypass background removal since we are trying to process a scene/room.
        # Resize and center-crop to exactly 576x576 as required by SV3D.
        w, h = image.size
        scale = 576.0 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        left = (new_w - 576) // 2
        top = (new_h - 576) // 2
        image = image.crop((left, top, left + 576, top + 576))
        
        input_image_np = np.array(image)
        
        img_tensor = ToTensor()(input_image_np)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device, dtype=torch.float16)

        num_frames = 21
        arc_degrees = 15.0 # Set to 15 instead of 360
        elevations_deg = [10.0] * num_frames
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        
        # Start at 0 (the input image) and pan by arc_degrees
        azimuths_deg = np.linspace(0, arc_degrees, num_frames)
        # SV3D expects radians relative to the conditioning view (which is now frame 0)
        azimuths_rad = [np.deg2rad(a % 360) for a in azimuths_deg]

        value_dict = {
            "cond_frames_without_noise": img_tensor,
            "motion_bucket_id": 127,
            "fps_id": 6,
            "cond_aug": 1e-5,
            "cond_frames": img_tensor + 1e-5 * torch.randn_like(img_tensor),
            "polars_rad": polars_rad,
            "azimuths_rad": azimuths_rad
        }

        logger.info(f"Running SV3D inference ({num_frames} frames) …")
        
        # Aggressive VRAM offloading: start with main memory-hogging components on CPU
        self.model.model.to("cpu")
        self.model.first_stage_model.to("cpu")
        torch.cuda.empty_cache()

        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16):
            conditioner = self.model.conditioner
            conditioner.to(self.device)
            
            keys = list(set([x.input_key for x in conditioner.embedders]))
            batch, batch_uc = self._get_batch(keys, value_dict, [1, num_frames], num_frames, self.device)
            
            c, uc = conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )

            # Move conditioner back to CPU to free up space for the UNet
            conditioner.to("cpu")
            torch.cuda.empty_cache()

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn((num_frames, 4, 576 // 8, 576 // 8), device=self.device, dtype=torch.float16)

            additional_model_inputs = {
                "image_only_indicator": torch.zeros(2, num_frames, device=self.device, dtype=torch.float16),
                "num_video_frames": batch["num_video_frames"]
            }

            def denoiser(input, sigma, c):
                return self.model.denoiser(self.model.model, input, sigma, c, **additional_model_inputs)

            # Move UNet to GPU for the heavy sampling loop
            self.model.model.to(self.device)
            
            samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)
            
            # Move UNet back to CPU
            self.model.model.to("cpu")
            torch.cuda.empty_cache()
            
            # Move VAE to GPU for the final decode, casting back to FP32 to avoid overflow garbage
            self.model.first_stage_model.to(self.device, dtype=torch.float32)

            self.model.en_and_decode_n_samples_a_time = 1
            with torch.autocast(self.device, enabled=False):
                samples_x = self.model.decode_first_stage(samples_z.to(dtype=torch.float32))
            
            # Move VAE back to CPU and drop down to fp16 again to save ram
            self.model.first_stage_model.to("cpu", dtype=torch.float16)
            torch.cuda.empty_cache()

            samples_x[:1] = value_dict["cond_frames_without_noise"]
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        # Convert to numpy
        vid = ((rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8))
        frames = [Image.fromarray(frame) for frame in vid]

        if num_views >= num_frames:
            indices = list(range(num_frames))
        else:
            indices = [round(i * (num_frames - 1) / (num_views - 1)) for i in range(num_views)]

        saved = []
        for idx in indices:
            out_path = views_dir / f"view_{idx:03d}.png"
            frames[idx].save(str(out_path))
            saved.append(str(out_path))
            logger.info(f"  saved frame {idx} → {out_path.name}")

        logger.info(f"SV3D: saved {len(saved)} views to {views_dir}")
        return sorted(saved)

    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            logger.info("SV3D model unloaded.")
