"""
Scene Processor - The Core Pipeline

Orchestrates the decomposition of a 2D image into 3D scene elements:
1. Depth Estimation (for placement)
2. Object Detection & Segmentation (for extraction)
3. Background Inpainting (for scene completeness)
4. 3D Object Generation (for geometry)
5. Scene Composition (exporting JSON)
"""
import asyncio
import logging
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Core modules
from depth_estimator import DepthEstimator
from segmentation import SceneSegmenter
from inpainting import OcclusionInpainter
from object_generator import Object3DGenerator, SceneObjectDetector

logger = logging.getLogger(__name__)

class SceneProcessor:
    def __init__(self, depth_estimator: DepthEstimator):
        self.depth_estimator = depth_estimator
        self.segmenter = SceneSegmenter()
        self.detector = SceneObjectDetector()
        self.inpainter = OcclusionInpainter(method="lama")
        self.object_generator = Object3DGenerator()

    async def estimate_depth(self, image_path: str, output_dir: str):
        """Step 1: Get depth map for scene layout."""
        # This is CPU/GPU intensive, might want to run in thread pool if blocking
        depth_arr, size = self.depth_estimator.process(image_path, output_dir)
        return depth_arr

    async def detect_and_segment(self, image_path: str, output_dir: str):
        """Step 2: Find objects and create masks."""
        # Use YOLO/GroundingDINO to find objects
        objects = self.detector.detect_objects(image_path, output_dir)
        
        # Optional: Refine masks with SAM if needed (already done in detector for now)
        logger.info(f"Detected {len(objects)} objects")
        return objects

    async def inpaint_background(self, image_path: str, objects: list, output_dir: str):
        """Step 3: Create a clean background by removing objects."""
        image = Image.open(image_path).convert("RGB")
        img_arr = np.array(image)
        
        # Create combined mask of all objects
        combined_mask = np.zeros(img_arr.shape[:2], dtype=np.uint8)
        for obj in objects:
            if obj.get("mask_path"):
                mask = np.array(Image.open(obj["mask_path"]).convert("L"))
                combined_mask = np.maximum(combined_mask, mask)
        
        # Inpaint the background
        clean_bg = self.inpainter.fill_occlusions(img_arr, combined_mask)
        
        # Save background
        bg_path = Path(output_dir) / "background_clean.jpg"
        Image.fromarray(clean_bg).save(bg_path, quality=95)
        
        # Also save the mask for debugging
        Image.fromarray(combined_mask).save(Path(output_dir) / "background_mask.png")
        
        return str(bg_path)

    async def extract_object_layers(self, image_path: str, objects: list, output_dir: str):
        """Step 4: Extract RGBA layers and depth patches for objects."""
        image = Image.open(image_path).convert("RGB")
        img_arr = np.array(image)
        
        # Load full depth map
        depth_path = Path(output_dir) / "depth.npy"
        full_depth = np.load(depth_path)
        
        extracted_layers = []
        
        for i, obj in enumerate(objects):
            if obj['score'] < 0.3: continue
            
            # Load mask
            if 'mask_path' in obj:
                mask = np.array(Image.open(obj['mask_path']).convert("L"))
            else:
                continue
                
            # Get bounding box
            x1, y1, x2, y2 = map(int, obj['box'])
            
            # Crop Image (RGBA)
            # Create RGBA with mask applied
            mask_bool = mask > 127
            rgba = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = img_arr
            rgba[..., 3] = mask
            
            obj_crop = rgba[y1:y2, x1:x2]
            
            # Crop Depth
            depth_crop = full_depth[y1:y2, x1:x2]
            
            # Pre-calculate ID and stats
            layer_id = f"layer_{i}"
            d_min, d_max = depth_crop.min(), depth_crop.max()
            
            # Try to generate 3D mesh if high confidence
            is_3d_model = False
            mesh_data = None
            
            if obj['score'] > 0.5:
                try:
                    # Lazy load generator only if needed
                    if not self.object_generator.model:
                        self.object_generator._load_model()
                        
                    if self.object_generator.model != "missing" and self.object_generator.model != "error":
                        # We need the full mask for TripoSR
                        # Re-load mask from path to be safe
                        full_mask = np.array(Image.open(obj['mask_path']).convert("L")) / 255.0
                        
                        mesh_res = self.object_generator.generate_from_segment(
                            image, 
                            full_mask, 
                            output_dir, 
                            f"{layer_id}_mesh"
                        )
                        
                        if mesh_res:
                            is_3d_model = True
                            mesh_data = mesh_res
                except Exception as e:
                    logger.warning(f"3D generation failed for {layer_id}: {e}")
            
            if is_3d_model and mesh_data:
                # Add as 3D Model
                extracted_layers.append({
                    "id": layer_id,
                    "label": obj.get('label', 'object'),
                    "type": "model",
                    "url": f"/uploads/{Path(output_dir).name}/{Path(mesh_data['mesh_path']).name}",
                    "bbox": [x1, y1, x2, y2],
                    "center_uv": [(x1+x2)/2/img_arr.shape[1], (y1+y2)/2/img_arr.shape[0]],
                    "depth_range": [float(d_min), float(d_max)]
                })
            else:
                # Faceback to 2.5D Layer
                img_filename = f"{layer_id}.png"
                depth_filename = f"{layer_id}_depth.npy"
                
                Image.fromarray(obj_crop).save(Path(output_dir) / img_filename)
                np.save(Path(output_dir) / depth_filename, depth_crop)
                
                # Create depth PNG for frontend visualization/loading
                # Normalize to 0-255 for PNG
                if d_max > d_min:
                    depth_norm = ((depth_crop - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_crop, dtype=np.uint8)
                Image.fromarray(depth_norm).save(Path(output_dir) / f"{layer_id}_depth.png")
                
                extracted_layers.append({
                    "id": layer_id,
                    "label": obj.get('label', 'object'),
                    "type": "layer",
                    "image_path": img_filename,
                    "depth_path": f"{layer_id}_depth.png",
                    "bbox": [x1, y1, x2, y2],
                    "center_uv": [(x1+x2)/2/img_arr.shape[1], (y1+y2)/2/img_arr.shape[0]],
                    "depth_range": [float(d_min), float(d_max)]
                })
            
        return extracted_layers

    def compose_scene(self, session_id: str, session_dir: str, bg_path: str, layers: list):
        """Step 5: Create final scene JSON with layered depth info."""
        import math
        
        # Camera parameters
        FOV_DEG = 75.0
        NEAR_DEPTH = 1.0
        FAR_DEPTH = 10.0
        
        # Load depth map stats for global scaling
        depth_path = Path(session_dir) / "depth.npy"
        full_depth = np.load(depth_path)
        global_min, global_max = float(full_depth.min()), float(full_depth.max())
        
        # Get image dimensions
        img = Image.open(Path(session_dir) / Path(bg_path).name)
        img_w, img_h = img.size
        
        scene_layers = []
        for layer in layers:
            # Calculate world position from center UV & depth
            u, v = layer["center_uv"]
            
            # Average depth of the object
            # Map global depth value to scene Z
            # (Note: Depth-Anything is relative disparity, usually inverse depth)
            # We used simple min-max normalization in the crop. 
            # Ideally we want the object's depth relative to the scene.
            # layer["depth_range"] are raw values from the full depth map.
            
            obj_d_min, obj_d_max = layer["depth_range"]
            avg_raw_depth = (obj_d_min + obj_d_max) / 2
            
            # Normalize global depth to 0-1
            if global_max > global_min:
                d_norm = (avg_raw_depth - global_min) / (global_max - global_min)
            else:
                d_norm = 0.5
            
            # Depth-Anything is High=Near (Disparity-like)
            # So Normalize 1.0 = Near, 0.0 = Far
            # We want z_dist: High value = Far, Low value = Near
            # So we invert d_norm: (1 - 1) -> 0 (Near), (1 - 0) -> 1 (Far)
            d_norm = 1.0 - d_norm
                
            # Map to scene distance
            z_dist = NEAR_DEPTH + d_norm * (FAR_DEPTH - NEAR_DEPTH)
            
            scene_obj = {
                "id": layer["id"],
                "type": layer["type"],  # 'layer' or 'model'
                "position_uv": [u, v],
                "depth_val": d_norm, # 0-1 relative to scene
                "bbox_uv": [
                    layer["bbox"][0]/img_w, layer["bbox"][1]/img_h,
                    layer["bbox"][2]/img_w, layer["bbox"][3]/img_h
                ]
            }
            
            if layer["type"] == "model":
                scene_obj["url"] = layer["url"] # Already formatted
                scene_obj["scale_factor"] = 1.0 # Global scale for GLB
            else:
                scene_obj["url"] = f"/uploads/{session_id}/{layer['image_path']}"
                scene_obj["depth_url"] = f"/uploads/{session_id}/{layer['depth_path']}"
            
            scene_layers.append(scene_obj)

        scene_data = {
            "background": {
                "url": f"/uploads/{session_id}/{Path(bg_path).name}",
                "depth_url": f"/uploads/{session_id}/depth.png",
                "type": "background_layer",
            },
            "objects": scene_layers,
            "camera": {
                "fov": FOV_DEG,
                "near": NEAR_DEPTH,
                "far": FAR_DEPTH,
            },
            "metadata": {
                "image_size": [img_w, img_h],
                "depth_range": [global_min, global_max]
            }
        }
        
        with open(Path(session_dir) / "scene.json", "w") as f:
            json.dump(scene_data, f, indent=2)
            
        return scene_data
