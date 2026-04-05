import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import depth_pro
from depth_pro import create_model_and_transforms, load_rgb

class DepthProEstimator:
    # eason cuda execute
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.transform = None
        
        # Calculate absolute path to checkpoint
        # backend/depth_pro_estimator.py -> backend/vendor/DepthPro/checkpoints/depth_pro.pt
        base_dir = Path(__file__).parent
        checkpoint_path = base_dir / "vendor/DepthPro/checkpoints/depth_pro.pt"
        
        if not checkpoint_path.exists():
            print(f"DepthPro Checkpoint NOT FOUND at: {checkpoint_path}")
            # Try root relative path just in case
            root_checkpoint = Path.cwd().parent / "backend/vendor/DepthPro/checkpoints/depth_pro.pt"
            if root_checkpoint.exists():
                checkpoint_path = root_checkpoint
            else:
                raise FileNotFoundError(f"DepthPro checkpoint not found at {checkpoint_path}")
        
        # Check file size (should be > 1.5GB)
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        if size_mb < 1000:
            print(f"WARNING: DepthPro checkpoint seems too small ({size_mb:.1f} MB). Download might be incomplete.")
            raise ValueError("Checkpoint file incomplete. Please wait for download to finish.")

        print(f"Loading DepthPro model from {checkpoint_path} ({size_mb:.1f} MB) on {self.device}...")
        
        # We need to pass the custom checkpoint URI in the config
        # Import config class
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        
        config = DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = str(checkpoint_path)
        
        try:
            self.model, self.transform = create_model_and_transforms(
                config=config,
                device=self.device,
                precision=torch.float16 if self.device == 'cuda' else torch.float32 
            )
            self.model.eval()
            print("DepthPro model loaded successfully.")
        except Exception as e:
            print(f"Failed to load DepthPro model: {e}")
            raise e

    def set_device(self, device: str) -> None:
        """Move weights to *device* (e.g. offload to cpu before ViewCrafter diffusion)."""
        if self.model is None:
            return
        self.device = device
        self.model.to(device)

    def process(self, image_path, output_dir):
        """
        Estimate metric depth and FOV.
        Returns:
            depth_map (numpy array): Normalized depth map (0-255) for visualization/masking
            size (tuple): (width, height)
            fov (float): Estimated Field of View in degrees
            metric_depth (numpy array): Metric depth in meters
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load image using DepthPro utility (handles EXIF if available)
        image, _, f_px = load_rgb(image_path)
        
        # Transform for model
        input_tensor = self.transform(image)
        
        # Inference
        with torch.no_grad():
            prediction = self.model.infer(input_tensor, f_px=f_px)
        
        metric_depth = prediction["depth"].cpu().numpy() # In meters
        focallength_px = prediction["focallength_px"]
        
        if isinstance(focallength_px, torch.Tensor):
            focallength_px = focallength_px.item()
            
        # Get original image size
        original_image = Image.open(image_path)
        W, H = original_image.size
        
        # Calculate FOV
        fov_rad = 2 * np.arctan(0.5 * W / focallength_px)
        fov_deg = np.degrees(fov_rad)
        
        print(f"DepthPro: Estimated Focal Length: {focallength_px:.2f} px, FOV: {fov_deg:.2f} degrees")
        
        # Save metric depth as .npy for downstream processing
        np.save(output_path / "depth_metric.npy", metric_depth)
        
        # Normalize depth for visualization (0-255)
        d_min = metric_depth.min()
        d_max = metric_depth.max()
        depth_normalized = (metric_depth - d_min) / (d_max - d_min + 1e-6)
        
        # Invert depth to match system convention (White=Near, Black=Far)
        # DepthPro is Low=Near (Black=Near). DepthAnything is High=Near (White=Near).
        depth_inverted = 1.0 - depth_normalized
        
        depth_uint8 = (depth_inverted * 255).astype(np.uint8)
        
        # Save outputs
        Image.fromarray(depth_uint8).save(output_path / "depth.png")
        np.save(output_path / "depth.npy", depth_inverted) # Legacy format (Inverted)
        
        return depth_normalized, (W, H), fov_deg
