"""
Panorama Generator — wraps PanoDreamer's CylindricalPanorama module.

Takes a single input image + text prompt and produces a 360° cylindrical panorama
using Stable Diffusion 2 inpainting with perspective-cylindrical projection.
"""

import os
import sys
import logging
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Add PanoDreamer to path
PANODREAMER_DIR = os.path.join(os.path.dirname(__file__), 'vendor', 'PanoDreamer')
if PANODREAMER_DIR not in sys.path:
    sys.path.insert(0, PANODREAMER_DIR)


class PanoramaGenerator:
    """Generates 360° cylindrical panoramas from a single image using PanoDreamer."""

    def __init__(self, device=None):
        self.device = device or self._get_device()
        self.model = None
        logger.info(f"PanoramaGenerator initialized (device={self.device})")

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _ensure_model(self):
        """Lazy-load the SD2 inpainting model on first use."""
        if self.model is not None:
            return

        logger.info("Loading PanoDreamer (SD2 inpainting model)...")
        from multicondiffusion_panorama import CylindricalPanorama
        self.model = CylindricalPanorama(device=self.device)
        logger.info("PanoDreamer loaded successfully")

    def generate(
        self,
        input_image: Image.Image,
        prompt: str,
        output_dir: str,
        session_id: str = "scene",
        height: int = 512,
        width: int = 3072,
        num_iterations: int = 10,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """
        Generate a 360° cylindrical panorama from an input image.

        Args:
            input_image: Input PIL Image
            prompt: Text description of the scene
            output_dir: Directory to save intermediate results
            session_id: Session identifier for file names
            height: Panorama height (default 512)
            width: Panorama width (default 3072, ~360° at the given FOV)
            num_iterations: Refinement iterations (fewer = faster, more = better)
            num_inference_steps: Denoising steps per iteration
            guidance_scale: Classifier-free guidance strength

        Returns:
            PIL Image of the 360° cylindrical panorama
        """
        self._ensure_model()

        input_image = input_image.convert("RGB")

        negative_prompt = (
            "caption, subtitle, text, blur, lowres, bad anatomy, "
            "bad hands, cropped, worst quality, watermark, distorted"
        )

        logger.info(
            f"Generating 360° panorama: {width}x{height}, "
            f"{num_iterations} iterations × {num_inference_steps} steps"
        )
        logger.info(f"Prompt: {prompt}")

        # Pick autocast device type
        if self.device.type == 'cuda':
            autocast_device = 'cuda'
        else:
            # MPS and CPU don't support torch.autocast the same way;
            # we'll monkey-patch within the generation call
            autocast_device = None

        # For MPS/CPU: patch torch.autocast('cuda') to be a no-op
        if autocast_device is None:
            import contextlib
            _orig_autocast = torch.autocast

            @contextlib.contextmanager
            def _noop_autocast(*args, **kwargs):
                yield

            torch.autocast = _noop_autocast

        try:
            panorama = self.model.image_to_cylindrical_panorama(
                scene=session_id,
                input_image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_iterations=num_iterations,
                save_dir=output_dir,
                debug=False,
            )
        finally:
            # Restore original autocast
            if autocast_device is None:
                torch.autocast = _orig_autocast

        # Save final panorama
        output_path = os.path.join(output_dir, "panorama_360.jpg")
        panorama.save(output_path, quality=95)
        logger.info(f"Panorama saved to {output_path}")

        return panorama
