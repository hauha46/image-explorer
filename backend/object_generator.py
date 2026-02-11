"""
3D Object Generator using TripoSR.

Specializes in generating high-quality 3D meshes from single images 
using the TripoSR model.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import json
import gc

import sys

logger = logging.getLogger(__name__)


class Object3DGenerator:
    """
    Generates 3D meshes from segmented objects using TripoSR.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
    def _load_model(self):
        """Lazy load TripoSR model."""
        if self.model is not None:
            return
        
        try:
            # Add vendor path to system path
            vendor_path = Path(__file__).parent / "vendor" / "TripoSR"
            if vendor_path.exists() and str(vendor_path) not in sys.path:
                sys.path.append(str(vendor_path))
                
            logger.info(f"Loading TripoSR on {self.device}...")
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground
            
            # Try local model directory first (avoids slow HF downloads)
            local_model_dir = Path(__file__).parent / "models" / "triposr"
            if (local_model_dir / "model.ckpt").exists() and (local_model_dir / "config.yaml").exists():
                logger.info(f"Loading TripoSR from local: {local_model_dir}")
                model_source = str(local_model_dir)
            else:
                logger.info("Loading TripoSR from HuggingFace Hub...")
                model_source = "stabilityai/TripoSR"
            
            self.model = TSR.from_pretrained(
                model_source,
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self.model.renderer.set_chunk_size(8192)
            self.model.to(self.device)
            self.remove_bg_fn = remove_background
            self.resize_fg_fn = resize_foreground
            
            logger.info("TripoSR loaded successfully")
            
        except ImportError as e:
            logger.error(f"TripoSR dependencies missing: {e}. Ensure vendor/TripoSR is present and pip install -r requirements.txt")
            self.model = "missing"
        except Exception as e:
            logger.error(f"Failed to load TripoSR: {e}")
            self.model = "error"

    def generate_from_segment(
        self,
        image: Image.Image,  # Full image
        mask: np.ndarray,    # Mask for object
        output_path: str,
        object_name: str
    ) -> Optional[Dict]:
        """Generate 3D mesh from segment."""
        self._load_model()
        
        if isinstance(self.model, str):
            logger.warning(f"Skipping 3D gen for {object_name}: TripoSR not available ({self.model})")
            return None

        output_dir = Path(output_path)
        
        # 1. Preprocess: Extract and crop object
        # Convert mask to alpha
        img_arr = np.array(image)
        rgba = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = img_arr
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        # Crop to bounding box
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        
        if not rows.any() or not cols.any():
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Pad slightly
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(mask.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(mask.shape[1], cmax + pad)
        
        cropped_rgba = rgba[rmin:rmax, cmin:cmax]
        cropped_pil = Image.fromarray(cropped_rgba)
        
        # 2. TripoSR expects RGB (3 channels) — composite RGBA over white background
        rgb_bg = Image.new("RGB", cropped_pil.size, (255, 255, 255))
        rgb_bg.paste(cropped_pil, mask=cropped_pil.split()[3])  # paste using alpha as mask
        cropped_pil = rgb_bg
        
        try:
            # Run generation
            with torch.no_grad():
                scene_codes = self.model([cropped_pil], device=self.device)
            
            meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True)
            mesh = meshes[0]
            
            # Save
            mesh_path = output_dir / f"{object_name}.glb"
            mesh.export(str(mesh_path))
            
            # Get centroid for placement metadata
            center_y = (rmin + rmax) / 2 / mask.shape[0]
            center_x = (cmin + cmax) / 2 / mask.shape[1]
            
            return {
                "name": object_name,
                "mesh_path": str(mesh_path),
                "position_uv": [center_x, center_y],
                "type": "model"
            }
            
        except Exception as e:
            logger.error(f"Generation failed for {object_name}: {e}")
            return None
            
    def generate_scene_objects(self, image_path, segments, output_dir, max_objects=5):
        """Batch generate objects."""
        image = Image.open(image_path).convert("RGB")
        
        results = []
        for i, seg in enumerate(segments[:max_objects]):
            if 'mask' not in seg and 'mask_path' in seg:
                # Load mask if not in memory
                mask_img = Image.open(seg['mask_path']).convert("L")
                seg['mask'] = np.array(mask_img) / 255.0
                
            if 'mask' in seg:
                res = self.generate_from_segment(
                    image, 
                    seg['mask'], 
                    output_dir, 
                    seg.get('name', f"obj_{i}")
                )
                if res:
                    results.append(res)
                    
        # Cleanup VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return results


class SceneObjectDetector:
    """Wrapper for Object Detection (YOLO/GroundingDINO)."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = None
        
    def detect_objects(self, image_path, output_dir) -> List[Dict]:
        """Detect objects and return list of dicts with 'mask_path', 'box', 'label'."""
        if self.detector is None:
            self._load_yolo()
            
        try:
            results = self.detector(image_path)
            # Parse YOLO results
            parsed = []
            img = Image.open(image_path)
            w, h = img.size
            
            for r in results:
                for match in r.boxes:
                    box = match.xyxy[0].cpu().numpy() # x1, y1, x2, y2
                    conf = float(match.conf[0])
                    cls = int(match.cls[0])
                    label = r.names[cls]
                    
                    if conf < 0.3: continue
                    
                    # Create mask from bbox (simple rectangle for now, 
                    # ideally use SAM here for segmentation!)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    x1, y1, x2, y2 = map(int, box)
                    mask[y1:y2, x1:x2] = 255
                    
                    mask_path = Path(output_dir) / f"{label}_{len(parsed)}.png"
                    Image.fromarray(mask).save(mask_path)
                    
                    parsed.append({
                        "name": f"{label}_{len(parsed)}",
                        "label": label,
                        "score": conf,
                        "box": [x1, y1, x2, y2],
                        "mask_path": str(mask_path)
                    })
            return parsed
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8n.pt")
        except ImportError:
            logger.error("YOLO not installed. pip install ultralytics")
