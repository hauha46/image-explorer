"""
Enhanced Depth Estimation using Depth-Anything-V2.
ZoeDepth repository is archived and has compatibility issues with PyTorch 2.x.
Using Depth-Anything-V2 Large as primary, with fallback to Small for faster inference.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Depth estimation using Depth-Anything-V2.
    
    The ZoeDepth repository (isl-org/ZoeDepth) was archived in May 2025 and has
    compatibility issues with PyTorch 2.x and Python 3.10+.
    
    Depth-Anything-V2 provides excellent relative depth and is actively maintained.
    For metric depth, we estimate scale based on typical indoor/outdoor assumptions.
    """
    
    def __init__(self, model_size: str = "large"):
        """
        Initialize depth estimator.
        
        Args:
            model_size: "small", "base", or "large"
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.model = None
        self.pipe = None
        
        logger.info(f"Initializing Depth-Anything-V2 ({model_size}) on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the depth estimation model."""
        from transformers import pipeline
        
        # Model mapping
        models = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        model_id = models.get(self.model_size, models["large"])
        
        try:
            self.pipe = pipeline(
                "depth-estimation",
                model=model_id,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Loaded {model_id}")
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
            # Fallback to small model
            self.pipe = pipeline(
                "depth-estimation",
                model=models["small"],
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Fell back to Depth-Anything-V2-Small")
    
    def process(self, image_path: str, output_dir: str) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Generate depth map from image.
        
        Returns:
            depth_arr: Depth array (0-255, 0=near, 255=far)
            size: Original image (width, height)
        """
        image = Image.open(image_path).convert("RGB")
        output_path = Path(output_dir)
        
        # Run depth estimation
        result = self.pipe(image)
        depth_pil = result["depth"]
        
        # Convert to numpy array
        depth_arr = np.array(depth_pil)
        
        # Ensure proper range (0-255)
        if depth_arr.max() <= 1.0:
            depth_arr = (depth_arr * 255).astype(np.uint8)
        
        # Estimate metric depth (rough approximation)
        # Assume typical scene depth range of 0.5m to 10m
        depth_meters = self._estimate_metric_depth(depth_arr)
        
        # Save outputs
        depth_pil.save(output_path / "depth.png")
        np.save(output_path / "depth.npy", depth_arr)
        np.save(output_path / "depth_meters.npy", depth_meters)
        
        logger.info(f"Generated depth map: {depth_arr.shape}")
        return depth_arr, image.size
    
    def _estimate_metric_depth(
        self,
        depth_arr: np.ndarray,
        near: float = 0.5,
        far: float = 10.0
    ) -> np.ndarray:
        """
        Estimate metric depth from relative depth.
        
        This is an approximation. For true metric depth, a model like
        Metric3D or ZoeDepth (with older PyTorch) would be needed.
        """
        # Normalize to 0-1
        depth_norm = depth_arr.astype(float) / 255.0
        
        # Invert if needed (0 = near, 1 = far for most uses)
        # Depth-Anything outputs: bright = far, dark = near
        
        # Linear mapping to metric range
        depth_meters = near + depth_norm * (far - near)
        
        return depth_meters
    
    def get_metric_depth(self, output_dir: str) -> np.ndarray | None:
        """Load metric depth if available."""
        metric_path = Path(output_dir) / "depth_meters.npy"
        if metric_path.exists():
            return np.load(metric_path)
        return None


class DepthAnythingV2:
    """
    Alternative direct implementation using the native Depth-Anything-V2 repo.
    Use this if you want more control over the model.
    """
    
    def __init__(self, encoder: str = "vitl"):
        """
        Args:
            encoder: "vits" (small), "vitb" (base), or "vitl" (large)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        self.model = None
        
    def _load_model(self):
        """Load model from HuggingFace."""
        if self.model is not None:
            return
            
        try:
            from transformers import AutoModelForDepthEstimation, AutoImageProcessor
            
            model_map = {
                "vits": "depth-anything/Depth-Anything-V2-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
                "vitl": "depth-anything/Depth-Anything-V2-Large-hf"
            }
            
            model_id = model_map.get(self.encoder, model_map["vitl"])
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded Depth-Anything-V2 {self.encoder}")
            
        except Exception as e:
            logger.error(f"Failed to load Depth-Anything-V2: {e}")
            raise
    
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Predict depth for an image.
        
        Returns:
            Depth map as numpy array (H, W), values 0-1 (0=near, 1=far)
        """
        self._load_model()
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False
        )
        
        depth = prediction.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth