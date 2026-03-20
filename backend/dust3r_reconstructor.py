import os
import sys
import glob
import argparse
from pathlib import Path
import logging

# Add vendor/Dust3r to Python Path so we can import its modules
VENDOR_DIR = Path(__file__).parent / "vendor/dust3r"
sys.path.append(str(VENDOR_DIR))

import torch
import numpy as np
from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# For exporting
import importlib.util
demo_path = VENDOR_DIR / "dust3r/demo.py"
spec = importlib.util.spec_from_file_location("dust3r_demo", str(demo_path))
dust3r_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dust3r_demo)
get_3D_model_from_scene = dust3r_demo.get_3D_model_from_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dust3rReconstructor:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Determine checkpoint path
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        logger.info(f"Loading DUSt3R model {model_name} onto {self.device}...")
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.device).eval()
        logger.info("DUSt3R model loaded successfully.")

    def reconstruct(self, images_dir: str, output_dir: str, as_pointcloud=True):
        """
        Takes a folder of images, runs Dust3r, and exports a 3D model (GLB pointcloud).
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather all images
        img_pths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.webp", "*.WEBP"]:
            img_pths.extend(glob.glob(os.path.join(images_dir, ext)))
            
        img_pths = sorted(img_pths)
        
        if len(img_pths) == 0:
            raise ValueError(f"No images found in {images_dir}")
        elif len(img_pths) == 1:
            logger.warning("Only 1 image found. Dust3r requires at least 2 for stereoscopic depth. Duplicating image for monocular mode.")
        
        logger.info(f"Found {len(img_pths)} images for reconstruction.")

        # Load and resize images to exactly 512 as expected by the ViTLarge_512 model
        imgs = load_images(img_pths, size=512)
        
        # Handle single image case (monocular depth)
        if len(imgs) == 1:
            import copy
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1

        # Make pairs (complete scene graph means every image is compared to every other image)
        logger.info("Generating image pairs...")
        pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
        
        # Run Inference
        logger.info("Running DUSt3R Inference (this may take a moment)...")
        output = inference(pairs, self.model, self.device, batch_size=1)
        
        # Global Alignment (Optimizing the 3D Point Cloud)
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        logger.info(f"Starting Global Alignment Mode: {mode.name}")
        scene = global_aligner(output, device=self.device, mode=mode)
        
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            logger.info("Computing global alignment (Optimization steps)...")
            # 300 iterations is a good default for quality vs speed
            loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

        # Export to GLB
        logger.info("Exporting 3D Scene to GLB...")
        outfile = get_3D_model_from_scene(
            outdir=str(out_dir), 
            scene=scene, 
            min_conf_thr=3.0, 
            as_pointcloud=as_pointcloud,
            mask_sky=False,
            clean_depth=True,
            transparent_cams=False,
            cam_size=0.05,
            silent=False
        )
        
        logger.info(f"Reconstruction Complete! Saved to: {outfile}")
        return outfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder containing input images")
    parser.add_argument("--output_dir", required=True, help="Folder to save the output scene.glb")
    args = parser.parse_args()
    
    try:
        reconstructor = Dust3rReconstructor()
        reconstructor.reconstruct(args.images_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
