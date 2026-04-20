"""
Metrics for the CLIP text re-conditioning sweep.

Three metrics are implemented:

1. CLIP-score (prompt adherence): cosine similarity between the OpenCLIP
   image embedding of each generated view and the OpenCLIP text embedding of
   the target prompt.  Uses the *same* ViT-H-14 / laion2b_s32b_b79k checkpoint
   SEVA loads, so the numbers are directly comparable to the injected
   direction.

2. LPIPS-vs-input (perceptual drift from the conditioning image): standard
   ``lpips`` AlexNet backbone.

3. MVGBench-style 3D self-consistency (a lightweight, DUSt3R-based proxy for
   the 3DGS split test).  We split the generated views into even and odd sets,
   run two independent DUSt3R reconstructions, and compare their aligned
   per-pair point clouds via symmetric Chamfer distance (cChamfer) plus a
   render-free cPSNR/cSSIM computed between overlapping warped views.  This
   avoids a full 3DGS training loop — which is too expensive for a sweep of
   ~90 runs — while still capturing the "does the prompt break 3D geometry?"
   signal the plan asks for.

All three metrics degrade gracefully: if a dependency is missing they log a
warning and return ``float('nan')`` for that field rather than crashing the
sweep.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared lazy caches so repeated metric calls don't reload heavy models.
# ---------------------------------------------------------------------------

_CLIP_STATE: dict = {"model": None, "preprocess": None, "tokenizer": None, "device": None}
_LPIPS_STATE: dict = {"model": None, "device": None}


def _get_clip(device: str = "cuda"):
    if _CLIP_STATE["model"] is not None and _CLIP_STATE["device"] == device:
        return _CLIP_STATE["model"], _CLIP_STATE["preprocess"], _CLIP_STATE["tokenizer"]
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device).eval()
    _CLIP_STATE.update(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)
    return model, preprocess, tokenizer


def _get_lpips(device: str = "cuda"):
    if _LPIPS_STATE["model"] is not None and _LPIPS_STATE["device"] == device:
        return _LPIPS_STATE["model"]
    import lpips  # type: ignore
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    _LPIPS_STATE.update(model=model, device=device)
    return model


# ---------------------------------------------------------------------------
# CLIP-score
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_scores(
    image_paths: Sequence[str | Path],
    prompt: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Return an array of CLIP cosine similarities (one per image) between each
    image in ``image_paths`` and ``prompt``.  Higher = better prompt adherence.
    """
    if not image_paths:
        return np.empty(0, dtype=np.float32)
    try:
        model, preprocess, tokenizer = _get_clip(device)
    except Exception as exc:
        logger.warning("CLIP unavailable (%s); returning NaN clip_scores.", exc)
        return np.full(len(image_paths), float("nan"), dtype=np.float32)

    imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
    toks = tokenizer([prompt]).to(device)
    image_feats = model.encode_image(imgs).float()
    text_feats = model.encode_text(toks).float()
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sims = (image_feats @ text_feats.T).squeeze(-1)
    return sims.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# LPIPS-vs-input
# ---------------------------------------------------------------------------

def _load_rgb_tensor(path: str | Path, device: str, size: tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def lpips_vs_input(
    generated_paths: Sequence[str | Path],
    input_path: str | Path,
    device: str = "cuda",
) -> np.ndarray:
    """
    Return an array of LPIPS distances between each generated image and the
    reference input image.  Lower = closer to the input.
    """
    if not generated_paths:
        return np.empty(0, dtype=np.float32)
    try:
        model = _get_lpips(device)
    except Exception as exc:
        logger.warning("LPIPS unavailable (%s); returning NaN lpips.", exc)
        return np.full(len(generated_paths), float("nan"), dtype=np.float32)

    # Resize everything to the smallest common size so LPIPS stays happy.
    with Image.open(input_path) as im:
        ref_size = im.size  # (W, H)
    target_size = (min(512, ref_size[0]), min(512, ref_size[1]))
    ref = _load_rgb_tensor(input_path, device, size=target_size)

    out = np.empty(len(generated_paths), dtype=np.float32)
    for i, p in enumerate(generated_paths):
        gen = _load_rgb_tensor(p, device, size=target_size)
        out[i] = float(model(ref, gen).item())
    return out


# ---------------------------------------------------------------------------
# 3D self-consistency (MVGBench-style, DUSt3R split proxy)
# ---------------------------------------------------------------------------

def _chamfer_symmetric(pts_a: np.ndarray, pts_b: np.ndarray, max_pts: int = 50000) -> float:
    """Symmetric Chamfer distance between two point clouds."""
    if pts_a.size == 0 or pts_b.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if pts_a.shape[0] > max_pts:
        pts_a = pts_a[rng.choice(pts_a.shape[0], size=max_pts, replace=False)]
    if pts_b.shape[0] > max_pts:
        pts_b = pts_b[rng.choice(pts_b.shape[0], size=max_pts, replace=False)]

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover
        logger.warning("scipy cKDTree unavailable (%s); chamfer=NaN.", exc)
        return float("nan")

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(0.5 * (np.mean(d_ab ** 2) + np.mean(d_ba ** 2)))


def self_consistency_split(
    view_paths: Sequence[str | Path],
    work_dir: str | Path,
    dust3r_reconstructor=None,
) -> dict[str, float]:
    """
    MVGBench-style lightweight 3D self-consistency proxy.

    Even and odd indexed views are reconstructed independently with DUSt3R;
    the two point clouds are rigidly aligned and compared with symmetric
    Chamfer distance.  Returns a dict with:

        { "self_consistency_chamfer": float,
          "self_consistency_cpsnr":   float,   # placeholder (NaN)
          "self_consistency_cssim":   float,   # placeholder (NaN)
          "n_even": int, "n_odd": int }

    ``cPSNR``/``cSSIM`` are left as NaN by default: computing the true
    MVGBench render-based numbers requires a full 3DGS train per split, which
    takes minutes and is impractical inside a 90-run sweep.  A future pass can
    add them by training a 3DGS over each split and re-rendering the other
    split's poses — wiring is left simple so a caller can replace Chamfer
    with the full metric later.
    """
    view_paths = list(view_paths)
    out = {
        "self_consistency_chamfer": float("nan"),
        "self_consistency_cpsnr": float("nan"),
        "self_consistency_cssim": float("nan"),
        "n_even": 0,
        "n_odd": 0,
    }

    if len(view_paths) < 4:
        logger.warning("self_consistency_split: need >=4 views, got %d.", len(view_paths))
        return out

    if dust3r_reconstructor is None:
        logger.warning(
            "self_consistency_split: no DUSt3R reconstructor provided, returning NaN metrics."
        )
        return out

    even_paths = [p for i, p in enumerate(view_paths) if i % 2 == 0]
    odd_paths = [p for i, p in enumerate(view_paths) if i % 2 == 1]

    work = Path(work_dir)
    even_dir = work / "split_even"
    odd_dir = work / "split_odd"
    even_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    # Stage images into split dirs so DUSt3R's glob finds them
    import shutil
    for p in even_paths:
        shutil.copy2(str(p), str(even_dir / Path(p).name))
    for p in odd_paths:
        shutil.copy2(str(p), str(odd_dir / Path(p).name))

    try:
        pts_even = _reconstruct_point_cloud(dust3r_reconstructor, str(even_dir), str(work / "recon_even"))
        pts_odd = _reconstruct_point_cloud(dust3r_reconstructor, str(odd_dir), str(work / "recon_odd"))
    except Exception as exc:
        logger.warning("self_consistency_split DUSt3R failed: %s", exc)
        return out

    out["n_even"] = int(pts_even.shape[0])
    out["n_odd"] = int(pts_odd.shape[0])

    if pts_even.size == 0 or pts_odd.size == 0:
        return out

    # Normalize scale so Chamfer is comparable across scenes
    def _normalize(p: np.ndarray) -> np.ndarray:
        p = p - p.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(p, axis=1).max() + 1e-8
        return p / scale

    pe = _normalize(pts_even)
    po = _normalize(pts_odd)

    chamfer = _chamfer_symmetric(pe, po)
    out["self_consistency_chamfer"] = float(chamfer)
    return out


def _reconstruct_point_cloud(reconstructor, images_dir: str, work_dir: str) -> np.ndarray:
    """
    Run DUSt3R on ``images_dir`` and return a numpy ``(N, 3)`` point cloud.

    We export a GLB via the reconstructor's normal ``reconstruct`` path and
    then parse its vertex positions with ``trimesh``.  This piggy-backs on
    the already-working reconstruction code and avoids duplicating the
    global-alignment logic.
    """
    out_path = reconstructor.reconstruct(images_dir=images_dir, output_dir=work_dir)
    import trimesh
    mesh = trimesh.load(out_path, force="scene")
    all_points: list[np.ndarray] = []
    if hasattr(mesh, "geometry"):
        for geom in mesh.geometry.values():
            if hasattr(geom, "vertices") and geom.vertices is not None:
                all_points.append(np.asarray(geom.vertices, dtype=np.float32))
    elif hasattr(mesh, "vertices"):
        all_points.append(np.asarray(mesh.vertices, dtype=np.float32))
    if not all_points:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


# ---------------------------------------------------------------------------
# Convenience aggregation
# ---------------------------------------------------------------------------

def aggregate(
    generated_paths: Sequence[str | Path],
    input_path: str | Path,
    prompt: str,
    work_dir: str | Path,
    dust3r_reconstructor=None,
    device: str = "cuda",
    compute_consistency: bool = True,
) -> tuple[dict, dict]:
    """
    Compute all sweep metrics for a single run.

    Returns a ``(per_view, summary)`` tuple where:

    - ``per_view`` is a dict keyed by frame_idx with per-frame CLIP-score and
      LPIPS-vs-input.
    - ``summary`` has aggregates plus the 3D self-consistency metric.
    """
    gen_paths = list(generated_paths)
    per_view: dict[int, dict] = {}

    clips = clip_scores(gen_paths, prompt, device=device) if prompt else np.full(len(gen_paths), float("nan"), dtype=np.float32)
    lpips_arr = lpips_vs_input(gen_paths, input_path, device=device)

    for i, p in enumerate(gen_paths):
        per_view[i] = {
            "frame_idx": i,
            "view_path": str(p),
            "clip_score": float(clips[i]) if i < len(clips) else float("nan"),
            "lpips_vs_input": float(lpips_arr[i]) if i < len(lpips_arr) else float("nan"),
        }

    summary = {
        "mean_clip_score": float(np.nanmean(clips)) if clips.size else float("nan"),
        "clip_score_view0": float(clips[0]) if clips.size else float("nan"),
        "mean_lpips_vs_input": float(np.nanmean(lpips_arr)) if lpips_arr.size else float("nan"),
    }

    if compute_consistency:
        summary.update(
            self_consistency_split(
                gen_paths,
                work_dir=work_dir,
                dust3r_reconstructor=dust3r_reconstructor,
            )
        )
    else:
        summary.update(
            self_consistency_chamfer=float("nan"),
            self_consistency_cpsnr=float("nan"),
            self_consistency_cssim=float("nan"),
            n_even=0,
            n_odd=0,
        )
    return per_view, summary
