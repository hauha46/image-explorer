"""
Scene Segmentation using SAM (Segment Anything Model).
Separates image into depth-based layers for multi-plane rendering.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DepthLayer:
    """Represents a single depth layer with its mask and properties."""
    index: int
    name: str
    mask: np.ndarray  # Binary mask (H, W)
    depth_min: float
    depth_max: float
    depth_avg: float
    area_ratio: float  # Fraction of image this layer covers


class SceneSegmenter:
    """
    Segments an image into depth-ordered layers for multi-plane imaging (MPI).
    
    Uses:
    1. SAM for object-aware segmentation
    2. Depth map to cluster segments into layers
    3. Produces clean alpha masks for each depth layer
    """
    
    def __init__(self, sam_checkpoint: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.mask_generator = None
        self.sam_checkpoint = sam_checkpoint
        
        logger.info(f"Initializing scene segmenter on {self.device}")
    
    def _load_sam(self):
        """Lazy load SAM model."""
        if self.sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            
            # Download SAM checkpoint if needed
            if self.sam_checkpoint is None:
                self.sam_checkpoint = self._download_sam_checkpoint()
            
            self.sam = sam_model_registry["vit_b"](checkpoint=self.sam_checkpoint)
            self.sam.to(self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=16,  # Lower for speed
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=1000,  # Filter tiny segments
            )
            
            logger.info("Loaded SAM model")
            
        except Exception as e:
            logger.warning(f"SAM loading failed: {e}")
            logger.info("Will use depth-only segmentation")
            self.sam = "fallback"
    
    def _download_sam_checkpoint(self) -> str:
        """Download SAM checkpoint from HuggingFace."""
        from huggingface_hub import hf_hub_download
        
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam-vit-base",
            filename="sam_vit_b_01ec64.pth",
        )
        return checkpoint_path
    
    def segment_layers(
        self,
        image_path: str,
        depth_path: str,
        output_dir: str,
        num_layers: int = 4
    ) -> List[DepthLayer]:
        """
        Segment image into depth layers.
        
        Args:
            image_path: Path to input image
            depth_path: Path to depth map (.npy or .png)
            output_dir: Directory to save layer masks
            num_layers: Number of depth layers to create
            
        Returns:
            List of DepthLayer objects, ordered far to near
        """
        self._load_sam()
        
        output_path = Path(output_dir)
        image = Image.open(image_path).convert("RGB")
        image_arr = np.array(image)
        
        # Load depth
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path).convert("L"))
        
        # Normalize depth to 0-1
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Create layers
        if self.sam != "fallback" and self.sam is not None:
            layers = self._segment_with_sam(image_arr, depth_norm, num_layers)
        else:
            layers = self._segment_depth_only(depth_norm, num_layers)
        
        # Save layer masks and images
        layers_data = []
        for layer in layers:
            # Save mask
            mask_path = output_path / f"layer_{layer.index}_mask.png"
            mask_img = Image.fromarray((layer.mask * 255).astype(np.uint8))
            mask_img.save(mask_path)
            
            # Save layer image with alpha
            layer_rgba = np.zeros((image_arr.shape[0], image_arr.shape[1], 4), dtype=np.uint8)
            layer_rgba[:, :, :3] = image_arr
            layer_rgba[:, :, 3] = (layer.mask * 255).astype(np.uint8)
            
            layer_path = output_path / f"layer_{layer.index}.png"
            Image.fromarray(layer_rgba).save(layer_path)
            
            layers_data.append({
                "index": layer.index,
                "name": layer.name,
                "mask_path": str(mask_path),
                "image_path": str(layer_path),
                "depth_min": float(layer.depth_min),
                "depth_max": float(layer.depth_max),
                "depth_avg": float(layer.depth_avg),
                "area_ratio": float(layer.area_ratio)
            })
        
        # Save layers metadata
        import json
        with open(output_path / "layers.json", "w") as f:
            json.dump(layers_data, f, indent=2)
        
        logger.info(f"Created {len(layers)} depth layers")
        return layers
    
    def _segment_with_sam(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        num_layers: int
    ) -> List[DepthLayer]:
        """Segment using SAM + depth clustering."""
        # Get SAM masks
        masks = self.mask_generator.generate(image)
        
        # Calculate average depth for each mask
        mask_depths = []
        for mask_data in masks:
            mask = mask_data["segmentation"]
            avg_depth = depth[mask].mean()
            mask_depths.append((mask, avg_depth, mask_data["area"]))
        
        # Sort by depth
        mask_depths.sort(key=lambda x: x[1])
        
        # Cluster into layers
        depth_thresholds = np.linspace(0, 1, num_layers + 1)
        layers = []
        
        for i in range(num_layers):
            layer_mask = np.zeros(depth.shape, dtype=bool)
            d_min, d_max = depth_thresholds[i], depth_thresholds[i + 1]
            
            # Add all masks that fall in this depth range
            for mask, avg_d, _ in mask_depths:
                if d_min <= avg_d < d_max:
                    layer_mask |= mask
            
            # Also add depth-based pixels not covered by SAM
            depth_in_range = (depth >= d_min) & (depth < d_max)
            layer_mask |= depth_in_range
            
            layer = DepthLayer(
                index=i,
                name=self._get_layer_name(i, num_layers),
                mask=layer_mask.astype(float),
                depth_min=d_min,
                depth_max=d_max,
                depth_avg=(d_min + d_max) / 2,
                area_ratio=layer_mask.sum() / layer_mask.size
            )
            layers.append(layer)
        
        return layers
    
    def _segment_depth_only(
        self,
        depth: np.ndarray,
        num_layers: int
    ) -> List[DepthLayer]:
        """Fallback: segment purely by depth thresholds."""
        layers = []
        depth_thresholds = np.linspace(0, 1, num_layers + 1)
        
        for i in range(num_layers):
            d_min, d_max = depth_thresholds[i], depth_thresholds[i + 1]
            
            # Create mask for this depth range
            mask = (depth >= d_min) & (depth < d_max)
            
            # Apply slight blur to reduce hard edges
            from scipy.ndimage import gaussian_filter
            mask_soft = gaussian_filter(mask.astype(float), sigma=2)
            mask_soft = np.clip(mask_soft * 1.5, 0, 1)  # Sharpen a bit
            
            layer = DepthLayer(
                index=i,
                name=self._get_layer_name(i, num_layers),
                mask=mask_soft,
                depth_min=d_min,
                depth_max=d_max,
                depth_avg=(d_min + d_max) / 2,
                area_ratio=mask.sum() / mask.size
            )
            layers.append(layer)
        
        return layers
    
    def _get_layer_name(self, index: int, total: int) -> str:
        """Get descriptive name for layer."""
        if total == 4:
            names = ["background", "far", "mid", "foreground"]
        elif total == 3:
            names = ["background", "midground", "foreground"]
        else:
            return f"layer_{index}"
        
        return names[min(index, len(names) - 1)]


def quick_segment(image_path: str, depth_path: str, output_dir: str, num_layers: int = 3) -> List[dict]:
    """Quick helper to segment without SAM (faster)."""
    output_path = Path(output_dir)
    image = Image.open(image_path).convert("RGB")
    image_arr = np.array(image)
    
    # Load depth
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    else:
        depth = np.array(Image.open(depth_path).convert("L"))
    
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    from scipy.ndimage import gaussian_filter
    
    layers_data = []
    thresholds = np.linspace(0, 1, num_layers + 1)
    
    layer_names = ["background", "midground", "foreground"] if num_layers == 3 else \
                  ["background", "far", "mid", "foreground"] if num_layers == 4 else \
                  [f"layer_{i}" for i in range(num_layers)]
    
    for i in range(num_layers):
        d_min, d_max = thresholds[i], thresholds[i + 1]
        mask = (depth_norm >= d_min) & (depth_norm < d_max)
        
        # Smooth mask edges
        mask_soft = gaussian_filter(mask.astype(float), sigma=3)
        mask_soft = np.clip(mask_soft * 1.2, 0, 1)
        
        # Save layer with alpha
        layer_rgba = np.zeros((image_arr.shape[0], image_arr.shape[1], 4), dtype=np.uint8)
        layer_rgba[:, :, :3] = image_arr
        layer_rgba[:, :, 3] = (mask_soft * 255).astype(np.uint8)
        
        layer_path = output_path / f"layer_{i}.png"
        Image.fromarray(layer_rgba).save(layer_path)
        
        layers_data.append({
            "index": i,
            "name": layer_names[i],
            "image_path": str(layer_path),
            "depth_avg": float((d_min + d_max) / 2)
        })
    
    return layers_data
