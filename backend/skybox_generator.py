"""
Skybox Generator - Creates 360° environment skyboxes matching scene context.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SkyboxGenerator:
    """
    Generates 360° HDRI-style skybox that matches the input scene's
    lighting, time of day, and atmosphere.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        
        # Predefined skybox templates for fallback
        self.templates = {
            "day_clear": {"top": (135, 206, 235), "horizon": (255, 255, 255), "gradient": 0.7},
            "day_cloudy": {"top": (169, 169, 169), "horizon": (220, 220, 220), "gradient": 0.5},
            "sunset": {"top": (70, 130, 180), "horizon": (255, 140, 0), "gradient": 0.6},
            "night": {"top": (25, 25, 60), "horizon": (50, 50, 80), "gradient": 0.3},
            "overcast": {"top": (128, 128, 140), "horizon": (180, 180, 190), "gradient": 0.4},
        }
    
    def _load_pipeline(self):
        """Lazy load diffusion pipeline for sky generation."""
        if self.pipe is not None:
            return
        
        try:
            from diffusers import StableDiffusionPipeline
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                
            logger.info("Loaded diffusion pipeline for skybox generation")
            
        except Exception as e:
            logger.warning(f"Failed to load diffusion pipeline: {e}")
            self.pipe = "fallback"
    
    def generate(
        self,
        image_path: str,
        output_dir: str,
        use_ai: bool = True
    ) -> str:
        """
        Generate a skybox that matches the input scene.
        
        Args:
            image_path: Path to input scene image
            output_dir: Directory to save skybox
            use_ai: Whether to use AI generation (slower but better)
            
        Returns:
            Path to generated skybox image
        """
        output_path = Path(output_dir)
        image = Image.open(image_path).convert("RGB")
        
        # Analyze scene
        scene_info = self._analyze_scene(image)
        
        if use_ai:
            self._load_pipeline()
            if self.pipe != "fallback":
                skybox = self._generate_ai_skybox(scene_info)
            else:
                skybox = self._generate_gradient_skybox(scene_info)
        else:
            skybox = self._generate_gradient_skybox(scene_info)
        
        # Save skybox
        skybox_path = output_path / "skybox.jpg"
        skybox.save(skybox_path, quality=95)
        
        # Also create cubemap faces for Three.js
        self._create_cubemap_faces(skybox, output_path)
        
        logger.info(f"Generated skybox: {skybox_path}")
        return str(skybox_path)
    
    def _analyze_scene(self, image: Image.Image) -> dict:
        """Analyze image to determine sky style."""
        arr = np.array(image)
        
        # Sample from top portion (likely sky area)
        top_portion = arr[:arr.shape[0] // 4]
        avg_color = top_portion.mean(axis=(0, 1))
        brightness = top_portion.mean()
        
        # Determine time of day / atmosphere
        is_blue = avg_color[2] > avg_color[0] * 1.2
        is_warm = avg_color[0] > avg_color[2] * 1.1
        is_bright = brightness > 150
        is_dark = brightness < 80
        
        if is_dark:
            style = "night"
            prompt = "night sky, stars, dark blue, peaceful night atmosphere"
        elif is_warm and is_bright:
            style = "sunset"
            prompt = "golden hour sky, warm sunset colors, orange and pink clouds"
        elif is_blue and is_bright:
            style = "day_clear"
            prompt = "clear blue sky, bright daylight, white fluffy clouds"
        elif brightness < 130:
            style = "overcast"
            prompt = "overcast sky, gray clouds, diffused light"
        else:
            style = "day_cloudy"
            prompt = "partly cloudy sky, soft clouds, natural daylight"
        
        return {
            "style": style,
            "prompt": prompt,
            "avg_color": avg_color.tolist(),
            "brightness": float(brightness),
            "top_color": tuple(int(c) for c in avg_color),
        }
    
    def _generate_ai_skybox(self, scene_info: dict) -> Image.Image:
        """Generate skybox using diffusion model."""
        prompt = f"360 degree equirectangular panorama skybox, {scene_info['prompt']}, seamless, photorealistic"
        
        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt="ground, buildings, objects, seams, distortion",
                width=1024,
                height=512,
                num_inference_steps=25,
                guidance_scale=7.5,
            ).images[0]
            
            # Scale up for better quality
            result = result.resize((2048, 1024), Image.LANCZOS)
            return result
            
        except Exception as e:
            logger.warning(f"AI skybox generation failed: {e}")
            return self._generate_gradient_skybox(scene_info)
    
    def _generate_gradient_skybox(self, scene_info: dict) -> Image.Image:
        """Generate procedural gradient skybox."""
        width, height = 2048, 1024
        
        style = scene_info.get("style", "day_clear")
        template = self.templates.get(style, self.templates["day_clear"])
        
        top_color = np.array(template["top"])
        horizon_color = np.array(template["horizon"])
        gradient_power = template["gradient"]
        
        # Create gradient
        skybox = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            # Normalized y position (0 at top, 1 at horizon, back to 0 at bottom)
            if y < height // 2:
                t = 1 - (y / (height // 2))  # 1 at top, 0 at middle
            else:
                t = (y - height // 2) / (height // 2)  # 0 at middle, 1 at bottom
            
            t = t ** gradient_power  # Apply curve
            
            # Interpolate color
            color = top_color * t + horizon_color * (1 - t)
            skybox[y, :] = color.astype(np.uint8)
        
        # Add subtle noise for texture
        noise = np.random.randint(-5, 6, skybox.shape, dtype=np.int16)
        skybox = np.clip(skybox.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(skybox)
    
    def _create_cubemap_faces(self, equirect: Image.Image, output_dir: Path):
        """
        Convert equirectangular to cubemap faces for Three.js CubeTextureLoader.
        Creates: px, nx, py, ny, pz, nz (positive/negative x, y, z)
        """
        # For simplicity, we'll save the equirect and let Three.js handle it
        # Three.js can use equirectangular directly with EquirectangularReflectionMapping
        
        # But also create a simple cubemap approximation
        w, h = equirect.size
        face_size = h // 2  # Each face is half the height
        
        faces = {}
        arr = np.array(equirect)
        
        # Simplified face extraction (not geometrically correct, but fast)
        # For proper conversion, would need proper spherical mapping
        
        # Front (pz) - center of panorama
        front_start = w // 4
        faces["pz"] = arr[:, front_start:front_start + face_size]
        
        # Right (px)
        right_start = w // 2
        faces["px"] = arr[:, right_start:right_start + face_size]
        
        # Back (nz)
        back_start = 3 * w // 4
        faces["nz"] = arr[:, back_start:min(back_start + face_size, w)]
        
        # Left (nx)
        faces["nx"] = arr[:, :face_size]
        
        # Top (py) and bottom (ny) - approximate from top/bottom of panorama
        top_strip = arr[:h // 4, :]
        faces["py"] = np.resize(np.mean(top_strip, axis=0, keepdims=True), (face_size, face_size, 3))
        
        bottom_strip = arr[3 * h // 4:, :]
        faces["ny"] = np.resize(np.mean(bottom_strip, axis=0, keepdims=True), (face_size, face_size, 3))
        
        # Save faces
        for name, face in faces.items():
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_img = Image.fromarray(face.astype(np.uint8))
                face_img = face_img.resize((512, 512), Image.LANCZOS)
                face_img.save(output_dir / f"skybox_{name}.jpg", quality=90)


def create_gradient_sky(output_dir: str, style: str = "day_clear") -> str:
    """Quick helper to create a simple gradient skybox."""
    gen = SkyboxGenerator()
    
    # Create dummy scene info
    scene_info = {"style": style}
    skybox = gen._generate_gradient_skybox(scene_info)
    
    output_path = Path(output_dir)
    skybox_path = output_path / "skybox.jpg"
    skybox.save(skybox_path, quality=95)
    
    return str(skybox_path)
