import os
import sys
import glob
import argparse
import logging
from pathlib import Path

# Reduce CUDA fragmentation (VGGT's aggregator allocates lots of variable-size
# attention tensors). Must be set before the first CUDA allocation to have any
# effect, so we do it at import time.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add vendor/vggt to Python path so we can import its modules
VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "vggt"
sys.path.append(str(VENDOR_DIR))

import torch
import numpy as np
import trimesh
from PIL import Image
from torchvision import transforms as TF

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VGGT's aggregator uses DINOv2 ViT with patch_size=14, so any input H/W must
# be divisible by 14. VGGT was trained at 518px (= 37 * 14).
_VGGT_PATCH_SIZE = 14


def _load_and_preprocess_images_at(
    image_path_list: list[str],
    target_size: int,
    mode: str = "pad",
) -> torch.Tensor:
    """Parameterized re-implementation of VGGT's ``load_and_preprocess_images``.

    The shipped helper at ``vggt.utils.load_fn.load_and_preprocess_images``
    hardcodes ``target_size = 518``, so we can't feed VGGT higher-resolution
    inputs through it. This is a line-for-line equivalent except that
    ``target_size`` is parameterized and validated to be a multiple of the
    ViT patch size (14).

    Args:
        image_path_list: list of paths to image files.
        target_size: output resolution in pixels. Must be a multiple of 14.
            For SEVA/ViewCrafter-style 576x576 inputs, 574 is the largest
            patch-aligned size that does not upsample.
        mode: "pad" (square target_size x target_size, largest dim fit,
            smaller padded with white) or "crop" (width=target_size,
            height=aspect, center-cropped if > target_size).

    Returns:
        torch.Tensor of shape (N, 3, H, W) in [0, 1]. For ``mode="pad"``,
        (H, W) == (target_size, target_size). For ``mode="crop"``, W ==
        target_size and H is the aspect-preserving, 14-divisible height.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ("crop", "pad"):
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if target_size % _VGGT_PATCH_SIZE != 0:
        raise ValueError(
            f"target_size must be a multiple of {_VGGT_PATCH_SIZE} (VGGT patch size); "
            f"got {target_size}. Try {target_size - target_size % _VGGT_PATCH_SIZE} "
            f"or {target_size + _VGGT_PATCH_SIZE - target_size % _VGGT_PATCH_SIZE}."
        )

    images: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        img = Image.open(image_path)

        # RGBA -> composite onto white (matches VGGT's helper).
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Largest dim = target_size; smaller dim scaled and rounded to
            # a multiple of patch_size, then padded with white to square.
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / _VGGT_PATCH_SIZE)
                    * _VGGT_PATCH_SIZE
                )
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / _VGGT_PATCH_SIZE)
                    * _VGGT_PATCH_SIZE
                )
        else:  # mode == "crop"
            new_width = target_size
            new_height = (
                round(height * (new_width / width) / _VGGT_PATCH_SIZE)
                * _VGGT_PATCH_SIZE
            )

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img_t = to_tensor(img)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_t = img_t[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img_t.shape[1]
            w_padding = target_size - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = torch.nn.functional.pad(
                    img_t,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img_t.shape[1], img_t.shape[2]))
        images.append(img_t)

    # In crop mode, different aspect ratios can land on different H; pad any
    # odd frames up to the max H/W so we can stack (matches VGGT's helper).
    if len(shapes) > 1:
        logger.warning(f"Images have different shapes after preprocessing: {shapes}")
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded: list[torch.Tensor] = []
        for img_t in images:
            h_padding = max_h - img_t.shape[1]
            w_padding = max_w - img_t.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_t = torch.nn.functional.pad(
                    img_t,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded.append(img_t)
        images = padded

    batched = torch.stack(images)
    if len(image_path_list) == 1 and batched.dim() == 3:
        batched = batched.unsqueeze(0)
    return batched


class VGGTReconstructor:
    """Feed-forward 3D reconstruction using Meta's VGGT (CVPR 2025).

    Mirrors the interface of :class:`Dust3rReconstructor`: construct once
    (weights loaded onto *device*), then call :meth:`reconstruct` with a folder
    of images and an output folder; returns the path to a colored ``.glb``.
    """

    HF_MODEL_ID = "facebook/VGGT-1B"

    def __init__(
        self,
        device: str | None = None,
        drop_unused_heads: bool = True,
    ):
        """
        Args:
            device: 'cuda' / 'mps' / 'cpu' (auto-detected if None).
            drop_unused_heads: if True (default), free the point_head and
                track_head immediately after loading. We never call them in
                :meth:`reconstruct`, and on a 12 GB card keeping them around
                plus running ``model.forward`` (which always runs point_head)
                is the difference between OOM and fitting.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading VGGT model {self.HF_MODEL_ID} onto {self.device}...")
        self.model = VGGT.from_pretrained(self.HF_MODEL_ID).to(self.device).eval()
        logger.info("VGGT model loaded successfully.")

        if drop_unused_heads:
            # We only need aggregator + camera_head + depth_head. Dropping the
            # other two frees their weights (point_head is a DPT head like
            # depth_head; track_head is CoTracker-sized) AND means we never
            # allocate the dense (S,H,W,3) point-map output.
            if getattr(self.model, "point_head", None) is not None:
                self.model.point_head = None
            if getattr(self.model, "track_head", None) is not None:
                self.model.track_head = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Dropped point_head and track_head to save VRAM.")

        # VGGT recommends bfloat16 on Ampere+ (SM 8.0+), else float16.
        # On CPU/MPS we skip autocast entirely.
        if self.device == "cuda":
            major = torch.cuda.get_device_capability()[0]
            self._amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self._amp_dtype = None

    def set_device(self, device: str) -> None:
        """Move weights to *device* (matches Dust3rReconstructor.set_device)."""
        if self.model is None:
            return
        self.device = device
        self.model.to(device)
        if device == "cuda":
            major = torch.cuda.get_device_capability()[0]
            self._amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self._amp_dtype = None

    def _predict_from_images_dir(
        self,
        images_dir: str,
        target_size: int = 574,
        depth_chunk_size: int = 2,
        max_images: int | None = None,
    ) -> dict:
        """Glob + preprocess + run VGGT forward on a folder of images.

        Shared between :meth:`reconstruct` and downstream consumers (e.g.
        ``VGGTDepthProReconstructor``) that need the raw cameras + depth to do
        their own unprojection at a different resolution.

        Returns:
            dict with keys:
              - ``img_pths``      (list[str])         : paths after subsampling
              - ``images``        (np.ndarray, float) : (S, 3, H, W) in [0, 1],
                    the preprocessed VGGT-input images (pad-mode, square)
              - ``depth``         (np.ndarray, float) : (S, H, W, 1) depth (m-like,
                    in whatever canonical unit VGGT produced)
              - ``depth_conf``    (np.ndarray, float) : (S, H, W) per-pixel conf
              - ``extrinsic``     (np.ndarray, float) : (S, 3, 4) cam-from-world
              - ``intrinsic``     (np.ndarray, float) : (S, 3, 3) in pixels, for (H, W)
              - ``H``, ``W``      (int, int)          : output image dims (== target_size
                    in pad mode for square inputs)
        """
        escaped_dir = glob.escape(images_dir)
        img_pths: list[str] = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.webp", "*.WEBP"]:
            img_pths.extend(glob.glob(os.path.join(escaped_dir, ext)))
        img_pths = sorted(img_pths)

        if len(img_pths) == 0:
            raise ValueError(f"No images found in {images_dir}")
        logger.info(f"Found {len(img_pths)} images for reconstruction.")

        if max_images is not None and len(img_pths) > max_images:
            idx = np.linspace(0, len(img_pths) - 1, max_images).round().astype(int)
            img_pths = [img_pths[i] for i in idx]
            logger.info(
                f"Subsampled to {len(img_pths)} images (max_images={max_images})."
            )

        # Pad-mode preserves the full framing of orbit-style inputs (square
        # target_size x target_size with white padding), which is usually what
        # you want for SEVA/ViewCrafter-style synthesized novel views.
        logger.info(
            f"Loading and preprocessing images (VGGT pad mode, {target_size}px)..."
        )
        images = _load_and_preprocess_images_at(
            img_pths, target_size=target_size, mode="pad"
        ).to(self.device)
        H_pp, W_pp = images.shape[-2], images.shape[-1]
        tokens_per_frame = (H_pp // _VGGT_PATCH_SIZE) * (W_pp // _VGGT_PATCH_SIZE)
        total_tokens = images.shape[0] * tokens_per_frame
        logger.info(
            f"Preprocessed tensor shape: {tuple(images.shape)} | "
            f"tokens/frame = {tokens_per_frame} "
            f"({H_pp // _VGGT_PATCH_SIZE}x{W_pp // _VGGT_PATCH_SIZE}) | "
            f"total aggregator tokens = {total_tokens}"
        )

        # Explicit submodule path (cheaper than self.model(images), which always
        # runs point_head and allocates a dense (S,H,W,3) tensor). We run the
        # aggregator once, then the two heads we care about, freeing the token
        # list between so the DPT head's fp32 upsample doesn't collide with it.
        H, W = images.shape[-2:]
        images_b = images.unsqueeze(0)  # (1, S, 3, H, W)

        logger.info("Running VGGT aggregator...")
        with torch.no_grad():
            if self._amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self._amp_dtype):
                    agg_tokens, ps_idx = self.model.aggregator(images_b)
            else:
                agg_tokens, ps_idx = self.model.aggregator(images_b)

        logger.info("Running camera head...")
        with torch.no_grad():
            pose_enc = self.model.camera_head(agg_tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))

        # camera_head is done with the full token list. The depth head only
        # reads ``self.model.depth_head.intermediate_layer_idx`` (typically
        # [4, 11, 17, 23]) out of ~24 layers, so the other ~20 can be freed
        # NOW -- each is (1, S, P, 2*embed_dim) in bf16, so at 574px / 10
        # frames / embed_dim=1024 that's ~68 MB * 20 = ~1.36 GB reclaimed.
        needed_layers = set(self.model.depth_head.intermediate_layer_idx)
        n_total = len(agg_tokens)
        for i in range(n_total):
            if i not in needed_layers:
                agg_tokens[i] = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info(
            f"Pruned aggregator tokens to {len(needed_layers)}/{n_total} "
            f"layers (keeping {sorted(needed_layers)}) before depth head."
        )

        logger.info(
            f"Running depth head (frames_chunk_size={depth_chunk_size})..."
        )
        with torch.no_grad():
            if self._amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self._amp_dtype):
                    depth, depth_conf = self.model.depth_head(
                        agg_tokens,
                        images=images_b,
                        patch_start_idx=ps_idx,
                        frames_chunk_size=depth_chunk_size,
                    )
            else:
                depth, depth_conf = self.model.depth_head(
                    agg_tokens,
                    images=images_b,
                    patch_start_idx=ps_idx,
                    frames_chunk_size=depth_chunk_size,
                )

        # Free the aggregator tokens (a list of per-layer tensors; largest live
        # allocation at this point) before moving outputs to CPU.
        del agg_tokens
        if self.device == "cuda":
            torch.cuda.empty_cache()

        depth_np = depth.detach().cpu().float().numpy().squeeze(0)              # (S, H, W, 1)
        depth_conf_np = depth_conf.detach().cpu().float().numpy().squeeze(0)    # (S, H, W)
        extrinsic_np = extrinsic.detach().cpu().float().numpy().squeeze(0)      # (S, 3, 4)
        intrinsic_np = intrinsic.detach().cpu().float().numpy().squeeze(0)      # (S, 3, 3)
        images_np = images.detach().cpu().float().numpy()                       # (S, 3, H, W)

        del depth, depth_conf, extrinsic, intrinsic, pose_enc, images_b, images
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "img_pths": img_pths,
            "images": images_np,
            "depth": depth_np,
            "depth_conf": depth_conf_np,
            "extrinsic": extrinsic_np,
            "intrinsic": intrinsic_np,
            "H": int(H),
            "W": int(W),
        }

    def reconstruct(
        self,
        images_dir: str,
        output_dir: str,
        as_pointcloud: bool = True,
        conf_threshold_pct: float = 50.0,
        min_conf_abs: float = 1e-5,
        max_images: int | None = None,
        target_size: int = 574,
        depth_chunk_size: int = 2,
        flip_yz_for_gltf: bool = True,
    ):
        """Run VGGT on a folder of images and export a colored GLB.

        Args:
            images_dir: folder containing input images.
            output_dir: folder to write ``scene.glb`` into.
            as_pointcloud: kept for interface parity with Dust3rReconstructor.
                VGGT's outputs are dense per-pixel points; we always export a
                colored point cloud regardless of this flag.
            conf_threshold_pct: drop the lowest *N*% of confidences before
                exporting. Matches VGGT's ``demo_viser.py`` slider semantic.
            min_conf_abs: additionally drop any point whose confidence is
                below this absolute floor.
            max_images: optional cap. If set and more images are found, they
                are uniformly subsampled. Useful to stay under 12 GB VRAM for
                large input sets (aggregator attention cost grows quadratically
                with total token count = S * (H/14) * (W/14)).
            target_size: input resolution VGGT sees (pixels, multiple of 14).
                Default 574 matches a 576x576 SEVA/ViewCrafter orbit view at
                the largest patch-aligned size that does not upsample. VGGT
                was trained at 518; going higher increases per-frame point
                count linearly in pixels (~(target_size/518)^2) but aggregator
                memory quadratically in token count.
            depth_chunk_size: how many frames to push through the depth head
                at once. VGGT's default is 8; we use 2 because the DPT head's
                fp32 upsample to ``target_size^2`` is the largest single
                activation allocation in the pipeline. Smaller values trade
                throughput for peak VRAM 1:1.
            flip_yz_for_gltf: if True (default), flip the Y and Z axes of the
                output point cloud (multiply by [1, -1, -1]) to convert from
                VGGT's OpenCV convention (Y down, Z forward) to glTF
                convention (Y up, Z back). Without this, online glTF viewers
                render the scene upside down. Set to False if you're feeding
                downstream code that expects raw VGGT coordinates.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        pred = self._predict_from_images_dir(
            images_dir,
            target_size=target_size,
            depth_chunk_size=depth_chunk_size,
            max_images=max_images,
        )

        logger.info("Unprojecting depth maps through predicted cameras...")
        world_points = unproject_depth_map_to_point_map(
            pred["depth"], pred["extrinsic"], pred["intrinsic"]
        )                                                                   # (S, H, W, 3)
        conf = pred["depth_conf"]

        # Colors from the (already-preprocessed) images. (S, 3, H, W) -> (S, H, W, 3).
        colors = pred["images"].transpose(0, 2, 3, 1)

        points_flat = world_points.reshape(-1, 3)
        colors_flat = np.clip(colors.reshape(-1, 3), 0.0, 1.0)
        colors_flat = (colors_flat * 255).astype(np.uint8)
        conf_flat = conf.reshape(-1)

        # Confidence filter: percentile + absolute floor (same recipe as demo_viser.py).
        thr = np.percentile(conf_flat, conf_threshold_pct)
        keep = (conf_flat >= thr) & (conf_flat > min_conf_abs)
        n_kept = int(keep.sum())
        logger.info(
            f"Keeping {n_kept}/{points_flat.shape[0]} points "
            f"(>= {conf_threshold_pct:.1f}th pct = {thr:.4g}, floor = {min_conf_abs})."
        )
        if n_kept == 0:
            raise RuntimeError("All points filtered out by confidence threshold; lower it and retry.")

        points_kept = points_flat[keep]
        colors_kept = colors_flat[keep]

        if flip_yz_for_gltf:
            # OpenCV (Y down, Z forward) -> glTF (Y up, Z back). Equivalent
            # to a 180-degree rotation around the X axis.
            points_kept = points_kept * np.array([1.0, -1.0, -1.0], dtype=points_kept.dtype)

        outfile = str(out_dir / "scene.glb")
        logger.info(f"Exporting colored point cloud to {outfile}...")
        pc = trimesh.PointCloud(vertices=points_kept, colors=colors_kept)
        scene = trimesh.Scene(pc)
        scene.export(outfile)

        logger.info(f"Reconstruction Complete! Saved to: {outfile}")
        return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder containing input images")
    parser.add_argument("--output_dir", required=True, help="Folder to save the output scene.glb")
    parser.add_argument("--conf_threshold_pct", type=float, default=50.0,
                        help="Drop lowest N%% of confidences before export (0 = keep all).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Cap # of input views (uniform subsample) to save VRAM.")
    parser.add_argument("--target_size", type=int, default=574,
                        help="VGGT input resolution (pixels, multiple of 14). "
                             "Default 574 matches 576x576 SEVA orbit views.")
    parser.add_argument("--depth_chunk_size", type=int, default=2,
                        help="Frames processed per DPT head pass (default 2, "
                             "VGGT default 8). Lower = less VRAM, slightly slower.")
    parser.add_argument("--keep_unused_heads", action="store_true",
                        help="Keep point_head and track_head in memory (uses more VRAM).")
    parser.add_argument("--no_flip", action="store_true",
                        help="Skip the OpenCV->glTF axis flip. By default, the "
                             "exported cloud has Y up / Z back so online .glb "
                             "viewers render it right-side up.")
    args = parser.parse_args()

    try:
        reconstructor = VGGTReconstructor(drop_unused_heads=not args.keep_unused_heads)
        reconstructor.reconstruct(
            args.images_dir,
            args.output_dir,
            conf_threshold_pct=args.conf_threshold_pct,
            max_images=args.max_images,
            target_size=args.target_size,
            depth_chunk_size=args.depth_chunk_size,
            flip_yz_for_gltf=not args.no_flip,
        )
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise
