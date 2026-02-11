"""
Inpainting module for filling occluded regions behind foreground objects.
Uses LaMa for fast inpainting or Stable Diffusion for quality.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class OcclusionInpainter:
    """
    Fills in occluded regions (areas hidden behind foreground objects).
    """
    
    def __init__(self, method: str = "lama"):
        self.method = method
        self.model = None
        self.device = "cuda" if import_torch_has_cuda() else ("mps" if import_torch_has_mps() else "cpu")
        logger.info(f"Initializing inpainter with method: {method} on {self.device}")
    
    def _load_model(self):
        """Lazy load inpainting model."""
        if self.model is not None:
            return
        
        if self.method == "lama":
            self._load_lama()
        else:
            self._load_diffusion()
    
    def _load_lama(self):
        """Load LaMa inpainting model with MPS/CPU fix."""
        try:
            # simple_lama_inpainting uses TorchScript model saved with CUDA tensors
            # We need to monkeypatch torch.jit.load to force map_location=cpu
            # or try to load it safely.
            
            import torch
            from simple_lama_inpainting import SimpleLama
            
            # The issue is specifically in SimpleLama's __init__ where it does:
            # self.model = torch.jit.load(model_path)
            # This fails if model was saved on CUDA and we are on CPU/MPS
            
            # Since we can't easily patch the library without being messy, 
            # let's try to fix it by monkeypatching momentarily or just implementing our own SimpleLama wrapper
            # that loads correctly.
            
            logger.info(f"Loading LaMa model...")
            
            # OPTION 1: Monkeypatch torch.jit.load
            original_load = torch.jit.load
            
            def safe_load(f, map_location=None, _extra_files=None):
                # Force map_location to cpu if not specified, to avoid CUDA errors on non-CUDA devices
                if map_location is None:
                    map_location = torch.device('cpu')
                return original_load(f, map_location=map_location, _extra_files=_extra_files)
            
            torch.jit.load = safe_load
            
            try:
                self.model = SimpleLama()
                # LaMa is fast enough on CPU, and moving to MPS causes issues 
                # because SimpleLama wrapper doesn't move inputs to device automatically.
                # So we keep it on CPU.
                
                logger.info("Loaded LaMa inpainting model (CPU)")
            finally:
                # Restore original
                torch.jit.load = original_load
                
        except ImportError:
            logger.warning("LaMa not available, falling back to OpenCV inpainting")
            self.model = "opencv"
        except Exception as e:
            logger.error(f"Failed to load LaMa: {e}")
            self.model = "opencv"
    
    def _load_diffusion(self):
        """Load Stable Diffusion inpainting."""
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            
            # Use device for diffusion as it's heavy
            self.model = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

            
            if self.device == "cuda":
                self.model.enable_attention_slicing()
                
            logger.info("Loaded Stable Diffusion inpainting")
            
        except Exception as e:
            logger.warning(f"Diffusion inpainting failed: {e}")
            self.model = "opencv"
    
    def fill_occlusions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: Optional[str] = None
    ) -> np.ndarray:
        """Fill occluded regions in image."""
        self._load_model()
        
        # Ensure mask is binary uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255
        
        if self.model == "opencv":
            return self._inpaint_opencv(image, mask)
        elif self.method == "lama":
            return self._inpaint_lama(image, mask)
        else:
            return self._inpaint_diffusion(image, mask, prompt)
    
    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback OpenCV inpainting."""
        import cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = cv2.inpaint(image_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMa inpainting."""
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        
        # SimpleLama expects PIL images
        result = self.model(img_pil, mask_pil)
        return np.array(result)
    
    def _inpaint_diffusion(self, image: np.ndarray, mask: np.ndarray, prompt: Optional[str]) -> np.ndarray:
        import torch
        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        
        orig_size = img_pil.size
        img_pil = img_pil.resize((512, 512), Image.LANCZOS)
        mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
        
        if prompt is None:
            prompt = "seamless background, photorealistic, natural continuation"
        
        with torch.no_grad():
            result = self.model(
                prompt=prompt,
                negative_prompt="artifacts, seams, discontinuity",
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=25,
                guidance_scale=7.5,
            ).images[0]
        
        result = result.resize(orig_size, Image.LANCZOS)
        return np.array(result)


def import_torch_has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False

def import_torch_has_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except:
        return False
