"""
Scene Processor - The Core Pipeline (NVS + Dust3r V3)

Orchestrates the decomposition of a 2D image into a true 3D scene:
1. Depth Estimation (DepthPro)
2. Novel View Synthesis (SVD / ViewCrafter / VIVID)
3. 3D Reconstruction (Dust3r)
4. Scene Composition (exporting JSON)
"""
import logging
import json
import shutil
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class SceneProcessor:
    def __init__(self, depth_estimator=None, duster_model=None, synthesizer=None):
        self.depth_estimator = depth_estimator
        self.duster_model = duster_model
        self.synthesizer = synthesizer
        self.scene_metadata = {}

    async def estimate_depth(self, image_path: str, output_dir: str):
        """Step 1: Get depth map for scene layout and estimated FOV."""
        depth_arr, size, fov_deg = self.depth_estimator.process(image_path, output_dir)
        self.scene_metadata["fov"] = fov_deg
        logger.info(f"DepthPro estimated FOV: {fov_deg:.2f} degrees")
        return depth_arr

    async def generate_novel_views(self, image_path: str, output_dir: str,
                                   depth_map=None, num_views: int = 8,
                                   prompt: str = None,
                                   clip_lambda: float | None = None,
                                   neutral_prompt: str | None = None):
        """Step 2: Generate multi-view images using the configured NVS backend.

        ``clip_lambda`` and ``neutral_prompt`` are forwarded to synthesizers
        that support CLIP text re-conditioning (currently only SEVA).  They are
        silently ignored by synthesizers whose ``generate_views`` signature
        does not accept them.
        """
        views_dir = Path(output_dir) / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        if self.synthesizer is None:
            logger.warning("No synthesizer configured — skipping NVS.")
            shutil.copy2(image_path, str(views_dir / "view_000.png"))
            return [str(views_dir / "view_000.png")]

        fov = self.scene_metadata.get("fov")
        synth_kwargs = {
            "image_path": image_path,
            "output_dir": output_dir,
            "num_views": num_views,
            "depth_map": depth_map,
            "fov_deg": fov,
            "prompt": prompt,
        }
        import inspect
        sig = inspect.signature(self.synthesizer.generate_views)
        if "clip_lambda" in sig.parameters and clip_lambda is not None:
            synth_kwargs["clip_lambda"] = clip_lambda
        if "neutral_prompt" in sig.parameters and neutral_prompt is not None:
            synth_kwargs["neutral_prompt"] = neutral_prompt

        view_paths = self.synthesizer.generate_views(**synth_kwargs)

        # Always include the original image so DUSt3R has an anchor view
        original_in_views = views_dir / "input_original.png"
        if not original_in_views.exists():
            shutil.copy2(image_path, str(original_in_views))
            view_paths.append(str(original_in_views))

        logger.info(f"NVS produced {len(view_paths)} views in {views_dir}")
        return sorted(view_paths)

    async def reconstruct_3d(self, views_dir: str, output_dir: str):
        """Step 3: Generate Point Cloud from views using Dust3r."""
        logger.info(f"Starting Dust3r reconstruction from {views_dir}")
        if self.duster_model is None:
            raise ValueError(
                "Dust3r reconstructor model was not initialized or "
                "was not passed in to SceneProcessor"
            )

        outfile = self.duster_model.reconstruct(
            images_dir=views_dir,
            output_dir=output_dir,
        )

        logger.info(f"Dust3r reconstruction complete. Output located at {outfile}")
        return outfile

    def compose_scene(self, session_id: str, session_dir: str):
        """Step 4: Create final scene JSON."""
        FOV_DEG = float(self.scene_metadata.get("fov", 75.0))

        scene_data = {
            "background": {
                "url": f"/uploads/{session_id}/input.jpg",
                "depth_url": f"/uploads/{session_id}/depth.png",
                "type": "background_layer",
            },
            "objects": [],
            "camera": {
                "fov": FOV_DEG,
                "near": 1.0,
                "far": 10.0,
            },
            "metadata": {
                "status": "complete",
            },
        }

        with open(Path(session_dir) / "scene.json", "w") as f:
            json.dump(scene_data, f, indent=2)

        return scene_data
