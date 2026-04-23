"""VGGT + DepthPro fusion reconstructor.

This module combines two monocular/multi-view depth models to produce a denser,
sharper point cloud than either can alone on a 12 GB GPU:

    VGGT-1B        : cross-view-consistent camera poses + sparse (574 px)
                      depth, used as the *scale anchor*.
    DepthPro       : Apple's monocular metric depth at DepthPro-native
                      resolution (up to 1500 px), used for the *dense depth*.

The pipeline:

    1. Run VGGT over all images at 574 px to get (extrinsic, intrinsic, depth,
       depth_conf) per frame in VGGT's canonical world frame.
    2. Offload VGGT to CPU and clear CUDA cache.
    3. For each input image, preprocess it with VGGT's padding logic at the
       higher ``depthpro_size`` and run DepthPro monocularly to get a per-frame
       metric depth map at (depthpro_size, depthpro_size).
    4. Per frame, fit an affine ``a * d_depthpro + b ~ d_vggt`` in weighted
       least squares (weights = VGGT's depth confidence) to put DepthPro's
       output in VGGT's depth scale/shift. One round of residual-based outlier
       rejection is applied.
    5. Rescale VGGT's intrinsics from 574 px to ``depthpro_size`` px.
    6. Unproject all frames' aligned DepthPro depth through VGGT's
       (extrinsic, rescaled_intrinsic) to get world points.
    7. Use VGGT's depth_conf (bilinearly upsampled to depthpro_size) as
       per-point confidence, apply percentile + absolute filter, export
       colored ``.glb``.

The two models never coexist on GPU: VGGT is moved to CPU before DepthPro is
brought onto the device. Peak VRAM stays under what each model needs on its
own (~5 GB for DepthPro at fp16, ~2 GB for VGGT's active heads at 574 px).

Honest caveat: DepthPro's output resolution equals its *input* resolution
(it resizes internally to 1536 px, predicts, then resizes back). So feeding
576 px sources up-scaled to 1500 px gives 1500 px depth, but the actual
information content is bounded by the 576 px source. The extra pixels are
still useful -- smoother surfaces, easier meshing, DepthPro's monocular
priors produce visibly sharper edges than VGGT's DPT head -- but the 1500 px
point count is not fully independent of the 574 px count.
"""

import os
import argparse
import logging
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
import trimesh

from vggt_reconstructor import (
    VGGTReconstructor,
    _load_and_preprocess_images_at,
    _VGGT_PATCH_SIZE,
)
from depth_pro_estimator import DepthProEstimator

# Imported via vggt_reconstructor which prepends vendor/vggt to sys.path.
from vggt.utils.geometry import unproject_depth_map_to_point_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _weighted_affine_fit_1d(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    outlier_pct: float = 5.0,
) -> tuple[float, float]:
    """Fit ``y ~ a * x + b`` by weighted least squares with one outlier pass.

    Closed-form WLS: minimize sum_i w_i (a x_i + b - y_i)^2.
    After an initial fit we drop the top ``outlier_pct``% of weighted
    residuals and refit. This catches disagreements where DepthPro and VGGT
    simply disagree about geometry for a particular pixel (occlusion
    boundaries, reflective surfaces) without being derailed by them.

    Args:
        x: predictor, shape (N,). Non-finite entries are treated as masked.
        y: target, shape (N,).
        w: weights, shape (N,). Non-positive entries are treated as masked.
        outlier_pct: after the first fit, drop this many % of highest-residual
            pixels and refit. Set to 0 to skip.

    Returns:
        (a, b) tuple of floats. If the fit is underdetermined (fewer than 2
        usable samples or zero variance), returns (1.0, 0.0) which is a
        safe identity fallback.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    w = w[mask].astype(np.float64)
    if x.size < 2:
        return 1.0, 0.0

    def _solve(x_, y_, w_):
        W = w_.sum()
        if W <= 0:
            return 1.0, 0.0
        mx = (w_ * x_).sum() / W
        my = (w_ * y_).sum() / W
        vxx = (w_ * (x_ - mx) ** 2).sum()
        vxy = (w_ * (x_ - mx) * (y_ - my)).sum()
        if vxx <= 0:
            return 1.0, float(my - mx)
        a_ = vxy / vxx
        b_ = my - a_ * mx
        return float(a_), float(b_)

    a, b = _solve(x, y, w)
    if outlier_pct > 0 and x.size > 20:
        residuals = np.abs(a * x + b - y)
        thr = np.percentile(residuals, 100 - outlier_pct)
        keep = residuals <= thr
        if keep.sum() >= 2:
            a, b = _solve(x[keep], y[keep], w[keep])
    return a, b


def _bilinear_resize_2d(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2D array to ``size = (H, W)`` via torch.

    Simpler than pulling PIL for a numerical (non-image) resample: we want
    floating-point bilinear, no quantization, no anti-alias. Works for
    (H, W), returns (size[0], size[1]) float32.
    """
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    t = torch.nn.functional.interpolate(
        t, size=size, mode="bilinear", align_corners=False
    )
    return t.squeeze(0).squeeze(0).numpy()


class VGGTDepthProReconstructor:
    """Dense reconstruction via VGGT (cameras + scale anchor) + DepthPro (dense depth).

    Interface mirrors :class:`VGGTReconstructor` / :class:`Dust3rReconstructor`:
    construct once, then call :meth:`reconstruct` with an images folder and
    an output folder; returns the path to a colored ``.glb``.
    """

    def __init__(
        self,
        device: str | None = None,
        vggt: "VGGTReconstructor | None" = None,
        depthpro: "DepthProEstimator | None" = None,
    ):
        """
        Args:
            device: 'cuda' / 'mps' / 'cpu'. Auto-detects if None.
            vggt: optional existing :class:`VGGTReconstructor`. If None, one is
                constructed. Reusing an instance avoids reloading the ~5 GB
                weights when the app holds a long-lived instance.
            depthpro: optional existing :class:`DepthProEstimator`. If None,
                one is constructed.
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

        self.vggt = vggt if vggt is not None else VGGTReconstructor(device=self.device)
        # DepthPro currently doesn't accept device=None.
        self.depthpro = (
            depthpro
            if depthpro is not None
            else DepthProEstimator(device=self.device)
        )

        # Park both on CPU between reconstructs so an idle instance doesn't
        # hold ~7 GB of VRAM (VGGT-1B ~5 GB + DepthPro ~1.9 GB fp16). We bring
        # each up to ``self.device`` one at a time inside :meth:`reconstruct`
        # and push it back to CPU before activating the other.
        logger.info("Parking VGGT + DepthPro on CPU until reconstruct is called...")
        self.vggt.set_device("cpu")
        self.depthpro.set_device("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def set_device(self, device: str) -> None:
        """Move both underlying models to *device*."""
        self.device = device
        self.vggt.set_device(device)
        self.depthpro.set_device(device)

    def _run_depthpro_on_preprocessed(
        self,
        images_padded: np.ndarray,
    ) -> np.ndarray:
        """Run DepthPro per-frame on a (S, 3, H, W) [0,1] float batch.

        DepthPro's own ``transform`` starts with ``ToTensor()`` which expects
        a PIL image, so we cannot route preprocessed tensors through it. We
        replicate the remaining steps manually:

            Normalize([0.5]*3, [0.5]*3)  ->  x * 2 - 1
            ConvertImageDtype(precision) ->  .to(precision)

        and call ``self.depthpro.model.infer(x, f_px=None)`` one frame at a
        time to keep peak VRAM bounded (DepthPro at 1500 px is already a
        ~5 GB allocation).

        Args:
            images_padded: (S, 3, H, W) float32 in [0, 1].

        Returns:
            depth_stack: (S, H, W) float32 metric depth (in DepthPro's scale,
            which we do NOT trust for absolute units -- see the alignment
            step downstream).
        """
        S, _, H, W = images_padded.shape
        logger.info(
            f"Running DepthPro per-frame at {H}x{W} ({S} frames)..."
        )

        # Match DepthProEstimator's precision choice: fp16 on CUDA, fp32 otherwise.
        precision = torch.float16 if self.device == "cuda" else torch.float32

        depths: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(S):
                t = (
                    torch.from_numpy(images_padded[s])
                    .float()
                    .unsqueeze(0)                     # (1, 3, H, W)
                    .to(self.device)
                )
                t = t * 2.0 - 1.0                     # Normalize([.5,.5,.5],[.5,.5,.5])
                t = t.to(precision)
                pred = self.depthpro.model.infer(t, f_px=None)
                d = pred["depth"]                     # (H, W) or squeezed
                if d.dim() == 3:
                    d = d.squeeze(0)
                depths.append(d.detach().cpu().float().numpy())
                del t, d, pred
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return np.stack(depths, axis=0)               # (S, H, W)

    def reconstruct(
        self,
        images_dir: str,
        output_dir: str,
        as_pointcloud: bool = True,
        conf_threshold_pct: float = 10.0,
        min_conf_abs: float = 1e-5,
        max_images: int | None = None,
        vggt_target_size: int = 574,
        depthpro_size: int = 1500,
        depth_chunk_size: int = 2,
        outlier_pct: float = 5.0,
        flip_yz_for_gltf: bool = True,
    ):
        """Fuse VGGT cameras with DepthPro depth and export a dense point cloud.

        Args:
            images_dir: input image folder.
            output_dir: output folder; writes ``scene.glb``.
            as_pointcloud: parity with other reconstructors; always writes a
                point cloud (no mesh).
            conf_threshold_pct: drop the lowest N%% of VGGT-derived confidences
                (upsampled to depthpro resolution) before export. Same semantic
                as :class:`VGGTReconstructor`.
            min_conf_abs: absolute floor on confidence.
            max_images: optional cap (uniform subsample if exceeded).
            vggt_target_size: resolution VGGT sees (px, multiple of 14).
                Default 574 matches a 576 px SEVA/ViewCrafter frame.
            depthpro_size: resolution DepthPro sees AND outputs (px, multiple
                of 14 so the intrinsics rescaling aligns cleanly with VGGT's
                preprocessing). Default 1500 trades ~3.5x more points per
                frame for interpolated input (the source is 576 px).
            depth_chunk_size: forwarded to VGGT's depth_head; see
                :class:`VGGTReconstructor`.
            outlier_pct: in the per-frame affine fit, fraction of highest
                residual pixels to reject and refit without. 0 disables.
            flip_yz_for_gltf: if True (default), flip the Y and Z axes of the
                output point cloud (multiply by [1, -1, -1]) to convert from
                VGGT's OpenCV convention (Y down, Z forward) to glTF
                convention (Y up, Z back). Without this, online glTF viewers
                render the scene upside down.
        """
        if depthpro_size % _VGGT_PATCH_SIZE != 0:
            raise ValueError(
                f"depthpro_size must be a multiple of {_VGGT_PATCH_SIZE} "
                f"(so intrinsic rescaling matches VGGT's patch-aligned "
                f"preprocessing); got {depthpro_size}."
            )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. VGGT pass: cameras + sparse anchor depth at 574 px.
        # ------------------------------------------------------------------
        if self.vggt.device != self.device:
            self.vggt.set_device(self.device)
        pred = self.vggt._predict_from_images_dir(
            images_dir,
            target_size=vggt_target_size,
            depth_chunk_size=depth_chunk_size,
            max_images=max_images,
        )
        img_pths = pred["img_pths"]
        H_vggt, W_vggt = pred["H"], pred["W"]
        vggt_depth = pred["depth"].squeeze(-1)                 # (S, H_v, W_v)
        vggt_depth_conf = pred["depth_conf"]                   # (S, H_v, W_v)
        extrinsic = pred["extrinsic"]                          # (S, 3, 4)
        intrinsic = pred["intrinsic"].copy()                   # (S, 3, 3), for (H_v, W_v)
        del pred

        # ------------------------------------------------------------------
        # 2. Offload VGGT to CPU so DepthPro has the full GPU.
        # ------------------------------------------------------------------
        logger.info("Offloading VGGT to CPU before DepthPro pass...")
        self.vggt.set_device("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 3. DepthPro pass: monocular dense depth at depthpro_size.
        #    Preprocess with the SAME pad logic as VGGT so the padded image
        #    contents (and therefore the intrinsics, once rescaled) align
        #    between the two models.
        # ------------------------------------------------------------------
        logger.info(
            f"Preprocessing images for DepthPro at {depthpro_size} px (pad mode)..."
        )
        imgs_dp_t = _load_and_preprocess_images_at(
            img_pths, target_size=depthpro_size, mode="pad"
        )                                                      # (S, 3, H_d, W_d) [0,1]
        imgs_dp_np = imgs_dp_t.numpy()
        del imgs_dp_t

        H_dp, W_dp = imgs_dp_np.shape[-2], imgs_dp_np.shape[-1]

        if self.depthpro.device != self.device:
            self.depthpro.set_device(self.device)
        depth_dp = self._run_depthpro_on_preprocessed(imgs_dp_np)  # (S, H_d, W_d)

        # DepthPro can stay on GPU (it's the last model we use) but we empty
        # the cache so fragmented allocations don't hurt the numpy-side work.
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 4. Per-frame affine alignment: a * d_dp + b ~= d_vggt, in VGGT's scale.
        #    We downsample DepthPro's depth to VGGT's 574 px resolution to
        #    fit (fewer samples but still ~300k per frame, and avoids
        #    interpolating VGGT's sparser outputs up).
        # ------------------------------------------------------------------
        logger.info(
            "Fitting per-frame affine alignment (DepthPro -> VGGT scale)..."
        )
        S = vggt_depth.shape[0]
        aligned_depth = np.empty((S, H_dp, W_dp), dtype=np.float32)
        for s in range(S):
            d_dp_small = _bilinear_resize_2d(depth_dp[s], (H_vggt, W_vggt))
            x = d_dp_small.reshape(-1)
            y = vggt_depth[s].reshape(-1)
            w = vggt_depth_conf[s].reshape(-1)
            w = np.clip(w - min_conf_abs, 0.0, None)
            # Exclude non-finite depths from DepthPro (rare but they happen
            # at image borders where the canonical inverse depth blows up).
            good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            a, b = _weighted_affine_fit_1d(
                x[good], y[good], w[good], outlier_pct=outlier_pct
            )
            aligned_depth[s] = a * depth_dp[s] + b
            logger.info(
                f"  frame {s:03d}: a = {a:+.4f}, b = {b:+.4f}, "
                f"used_px = {int(good.sum())}/{x.size}"
            )

        # ------------------------------------------------------------------
        # 5. Rescale VGGT intrinsics from (H_vggt, W_vggt) -> (H_dp, W_dp).
        #    Both coordinate systems have origin at the top-left, so scaling
        #    fx, cx by W_dp/W_vggt and fy, cy by H_dp/H_vggt is correct.
        # ------------------------------------------------------------------
        sx = W_dp / W_vggt
        sy = H_dp / H_vggt
        intrinsic_dp = intrinsic.copy()
        intrinsic_dp[:, 0, 0] *= sx                            # fx
        intrinsic_dp[:, 0, 2] *= sx                            # cx
        intrinsic_dp[:, 1, 1] *= sy                            # fy
        intrinsic_dp[:, 1, 2] *= sy                            # cy

        # ------------------------------------------------------------------
        # 6. Unproject aligned DepthPro depths through VGGT's cameras.
        # ------------------------------------------------------------------
        logger.info(
            f"Unprojecting aligned depth at {H_dp}x{W_dp} "
            f"({S * H_dp * W_dp / 1e6:.2f} M candidate points)..."
        )
        world_points = unproject_depth_map_to_point_map(
            aligned_depth[..., None],                          # (S, H_d, W_d, 1)
            extrinsic,
            intrinsic_dp,
        )                                                      # (S, H_d, W_d, 3)

        # ------------------------------------------------------------------
        # 7. Confidence: bilinear-upsample VGGT's depth_conf to depthpro res.
        # ------------------------------------------------------------------
        logger.info("Upsampling VGGT confidence to DepthPro resolution...")
        conf_dp = np.empty((S, H_dp, W_dp), dtype=np.float32)
        for s in range(S):
            conf_dp[s] = _bilinear_resize_2d(vggt_depth_conf[s], (H_dp, W_dp))

        colors = imgs_dp_np.transpose(0, 2, 3, 1)              # (S, H_d, W_d, 3) [0,1]

        points_flat = world_points.reshape(-1, 3)
        colors_flat = np.clip(colors.reshape(-1, 3), 0.0, 1.0)
        colors_flat = (colors_flat * 255).astype(np.uint8)
        conf_flat = conf_dp.reshape(-1)

        thr = np.percentile(conf_flat, conf_threshold_pct)
        keep = (conf_flat >= thr) & (conf_flat > min_conf_abs)
        # Also drop any non-finite points the fit may have produced.
        keep &= np.all(np.isfinite(points_flat), axis=1)
        n_kept = int(keep.sum())
        logger.info(
            f"Keeping {n_kept}/{points_flat.shape[0]} points "
            f"(>= {conf_threshold_pct:.1f}th pct = {thr:.4g}, "
            f"floor = {min_conf_abs})."
        )
        if n_kept == 0:
            raise RuntimeError(
                "All points filtered out by confidence threshold; lower it and retry."
            )

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

        # Return to the idle-VRAM-free invariant so the long-lived app
        # instance doesn't keep DepthPro resident on the GPU.
        logger.info("Parking DepthPro back on CPU...")
        self.depthpro.set_device("cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Reconstruction Complete! Saved to: {outfile}")
        return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder containing input images")
    parser.add_argument("--output_dir", required=True, help="Folder to save the output scene.glb")
    parser.add_argument("--conf_threshold_pct", type=float, default=10.0,
                        help="Drop lowest N%% of confidences before export (default 10).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Cap # of input views (uniform subsample) to save VRAM.")
    parser.add_argument("--vggt_target_size", type=int, default=574,
                        help="VGGT input resolution (pixels, multiple of 14).")
    parser.add_argument("--depthpro_size", type=int, default=1500,
                        help="DepthPro input/output resolution (pixels, multiple of 14). "
                             "Default 1500 gives ~3.5x more points per frame than 574 px, "
                             "but extra detail is bounded by the source image resolution.")
    parser.add_argument("--depth_chunk_size", type=int, default=2,
                        help="Frames processed per VGGT DPT head pass (default 2).")
    parser.add_argument("--outlier_pct", type=float, default=5.0,
                        help="Per-frame affine fit: %% of highest-residual pixels "
                             "to reject before the second (final) fit. 0 disables.")
    parser.add_argument("--no_flip", action="store_true",
                        help="Skip the OpenCV->glTF axis flip. By default, the "
                             "exported cloud has Y up / Z back so online .glb "
                             "viewers render it right-side up.")
    args = parser.parse_args()

    try:
        reconstructor = VGGTDepthProReconstructor()
        reconstructor.reconstruct(
            args.images_dir,
            args.output_dir,
            conf_threshold_pct=args.conf_threshold_pct,
            max_images=args.max_images,
            vggt_target_size=args.vggt_target_size,
            depthpro_size=args.depthpro_size,
            depth_chunk_size=args.depth_chunk_size,
            outlier_pct=args.outlier_pct,
            flip_yz_for_gltf=not args.no_flip,
        )
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise
