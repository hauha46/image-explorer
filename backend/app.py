"""
AI-Enhanced 3D Scene Reconstruction API

Decomposes a single image into a 3D scene with:
1. Depth Estimation (DepthPro)
2. Novel View Synthesis (SVD / ViewCrafter / VIVID / SEVA / SV3D / Zero123++)
3. 3D Reconstruction (Dust3r)
4. Scene Composition
"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import uuid
import logging
from pathlib import Path
from scene_processor import SceneProcessor
from reconstruction.dust3r_reconstructor import Dust3rReconstructor
from contextlib import asynccontextmanager
from depth_pro_estimator import DepthProEstimator
from novel_view_synthesis import get_synthesizer, AVAILABLE_MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(BaseModel):
    session_id: str
    status: str
    progress: int
    current_step: str
    steps_completed: List[str]


# In-memory status tracking
processing_status = {}


# Global dict to hold models
global_model_instances = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Dust3r model into memory...")
    global_model_instances["dust3r"] = Dust3rReconstructor()
    global_model_instances["depth-pro"] = DepthProEstimator()

    print("\n>>> Frontend GUI ready at: http://localhost:9876/app <<<\n")

    yield

    print("Cleaning up ml model instances...")
    global_model_instances.clear()



app = FastAPI(
    title="Image to 3D Scene Reconstruction",
    description="Decompose single images into fully navigable 3D scenes via NVS + Dust3r",
    version="4.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")




def update_status(session_id: str, status: str, progress: int, step: str):
    """Update processing status."""
    if session_id not in processing_status:
        processing_status[session_id] = {
            "status": status,
            "progress": progress,
            "current_step": step,
            "steps_completed": []
        }
    else:
        processing_status[session_id]["status"] = status
        processing_status[session_id]["progress"] = progress
        if processing_status[session_id]["current_step"] != step:
            processing_status[session_id]["steps_completed"].append(
                processing_status[session_id]["current_step"]
            )
        processing_status[session_id]["current_step"] = step


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get processing status."""
    if session_id in processing_status:
        return {
            "session_id": session_id,
            **processing_status[session_id]
        }
    return {"session_id": session_id, "status": "unknown"}


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    model: str = Form(default="svd"),
    prompt: Optional[str] = Form(default=None),
    clip_lambda: Optional[float] = Form(default=None),
    neutral_prompt: Optional[str] = Form(default=None),
):
    """
    Upload image and start 3D scene reconstruction.

    ``model`` selects the NVS backend: ``svd`` | ``viewcrafter`` | ``vivid`` | ``panodreamer`` | ``zero123pp`` | ``sv3d`` | ``seva``.
    ``prompt`` is an optional text prompt (used by ViewCrafter and PanoDreamer
    for scene guidance; now also used by SEVA via training-free CLIP-direction
    re-conditioning — see ``docs/RESEARCH_NOTES.md``).
    ``clip_lambda`` controls the strength of CLIP-direction injection for SEVA
    (default ~0.2).  ``neutral_prompt`` optionally overrides the neutral anchor
    (default "a photo of a living room interior").
    Returns session_id immediately.
    """
    if model not in AVAILABLE_MODELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model '{model}'. Choose from {AVAILABLE_MODELS}"},
        )

    session_id = str(uuid.uuid4())[:8]
    session_dir = Path(f"uploads/{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)

    img_path = session_dir / "input.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from PIL import Image as PILImage
    MAX_DIM = 1920
    img = PILImage.open(img_path)
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        logger.info(f"Resizing input from {w}x{h} to {new_w}x{new_h}")
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
        img.save(img_path, quality=95)
    img.close()

    logger.info(f"Starting scene reconstruction v4.0.0: session={session_id}, model={model}")
    update_status(session_id, "processing", 0, "Initializing")

    background_tasks.add_task(
        run_scene_pipeline,
        session_id,
        str(session_dir),
        str(img_path),
        model,
        prompt,
        clip_lambda,
        neutral_prompt,
    )

    return {
        "session_id": session_id,
        "model": model,
        "prompt": prompt,
        "clip_lambda": clip_lambda,
        "neutral_prompt": neutral_prompt,
    }


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _list_files(directory: str, relative_to: str) -> list[tuple[str, int]]:
    """Walk *directory* and return (relative_path, size_bytes) pairs."""
    results = []
    base = Path(relative_to)
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel = fpath.relative_to(base)
            results.append((str(rel), fpath.stat().st_size))
    return sorted(results)


def _git_sha() -> str | None:
    """Best-effort git SHA of the repo containing backend/.  Never raises."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def _summarize_vc_params(vc_params: dict | None) -> list[str]:
    """Compact one-line summaries of ViewCrafter last_run_params for run_report.txt."""
    if not vc_params:
        return []
    lines: list[str] = []
    tr = vc_params.get("trajectory", {}) or {}
    df = vc_params.get("diffusion", {}) or {}
    if tr:
        lines.append(
            f"  trajectory: d_phi={tr.get('d_phi')} d_theta={tr.get('d_theta')} "
            f"d_r={tr.get('d_r')} elevation={tr.get('elevation')}"
        )
    if df:
        lines.append(
            f"  diffusion:  ddim_steps={df.get('ddim_steps')} "
            f"cfg={df.get('cfg')} seed={df.get('seed')} "
            f"frame_stride={df.get('frame_stride')}"
        )
    return lines


def _write_run_report(
    session_dir: str,
    session_id: str,
    model_name: str,
    prompt: str | None,
    img_path: str,
    timestamp: str,
    timings: dict[str, float],
    artifacts: dict[str, list[str]],
    fov_deg: float | None,
    clip_lambda: float | None = None,
    neutral_prompt: str | None = None,
    dtype: str | None = None,
    num_steps: int | None = None,
    vc_params: dict | None = None,
    mesh_status: str | None = None,
    mesh_error: str | None = None,
) -> None:
    """Write a comprehensive run_report.txt into the session directory."""
    from PIL import Image as PILImage

    sd = Path(session_dir)

    try:
        with PILImage.open(img_path) as img:
            w, h = img.size
        dims = f"{w}x{h}"
    except Exception:
        dims = "unknown"

    lines: list[str] = []
    lines.append("=" * 48)
    lines.append(f" RUN REPORT \u2014 session {session_id}")
    lines.append("=" * 48)
    lines.append(f"Model:      {model_name}")
    lines.append(f"Prompt:     {prompt or '(none)'}")
    lines.append(f"Input:      {os.path.basename(img_path)} ({dims})")
    lines.append(f"Timestamp:  {timestamp}")
    lines.append("")

    # Stage 1
    lines.append("\u2500\u2500 Stage 1: Depth Estimation (DepthPro) \u2500\u2500")
    lines.append(f"  Time: {timings.get('depth', 0):.2f}s")
    if fov_deg is not None:
        lines.append(f"  FOV:  {fov_deg:.1f} degrees")
    lines.append("  Files:")
    for f in artifacts.get("depth", []):
        fp = sd / f
        sz = _format_size(fp.stat().st_size) if fp.exists() else "?"
        lines.append(f"    {f:<30s} ({sz})")
    lines.append("")

    # Stage 2
    view_files = artifacts.get("nvs", [])
    lines.append(f"\u2500\u2500 Stage 2: Novel View Synthesis ({model_name}) \u2500\u2500")
    lines.append(f"  Time: {timings.get('nvs', 0):.2f}s")
    lines.append(f"  Views generated: {len(view_files)}")
    # SEVA-specific NVS params (only populated when model_name == "seva")
    if model_name == "seva":
        if clip_lambda is not None:
            lines.append(f"  clip_lambda:    {clip_lambda}")
        if neutral_prompt:
            lines.append(f"  neutral_prompt: {neutral_prompt}")
        if dtype:
            lines.append(f"  dtype:          {dtype}")
        if num_steps is not None:
            lines.append(f"  num_steps:      {num_steps}")
    # ViewCrafter-specific one-line summary
    if model_name == "viewcrafter":
        lines.extend(_summarize_vc_params(vc_params))
    lines.append("  Files:")
    for f in view_files:
        fp = sd / f
        sz = _format_size(fp.stat().st_size) if fp.exists() else "?"
        lines.append(f"    {f:<30s} ({sz})")
    lines.append("")

    # Stage 3
    lines.append("\u2500\u2500 Stage 3: 3D Reconstruction (DUSt3R) \u2500\u2500")
    lines.append(f"  Time: {timings.get('reconstruction', 0):.2f}s")
    lines.append("  Files:")
    for f in artifacts.get("reconstruction", []):
        fp = sd / f
        sz = _format_size(fp.stat().st_size) if fp.exists() else "?"
        lines.append(f"    {f:<30s} ({sz})")
    lines.append("")

    # Stage 5 (soft-fail; lives between DUSt3R and compose_scene)
    lines.append("\u2500\u2500 Stage 5: Mesh Generation (Open3D ball-pivoting) \u2500\u2500")
    lines.append(f"  Time: {timings.get('mesh', 0):.2f}s")
    if mesh_status is not None:
        lines.append(f"  Status: {mesh_status}")
    if mesh_error:
        lines.append(f"  Error:  {mesh_error[:200]}")
    lines.append("  Files:")
    for f in artifacts.get("mesh", []):
        fp = sd / f
        sz = _format_size(fp.stat().st_size) if fp.exists() else "?"
        lines.append(f"    {f:<30s} ({sz})")
    lines.append("")

    # Stage 4
    lines.append("\u2500\u2500 Stage 4: Scene Composition \u2500\u2500")
    lines.append(f"  Time: {timings.get('composition', 0):.2f}s")
    lines.append("  Files:")
    for f in artifacts.get("composition", []):
        fp = sd / f
        sz = _format_size(fp.stat().st_size) if fp.exists() else "?"
        lines.append(f"    {f:<30s} ({sz})")
    lines.append("")

    # Summary
    total_time = timings.get("total", 0)
    all_files = _list_files(session_dir, session_dir)
    total_size = sum(s for _, s in all_files)
    lines.append("\u2500\u2500 Summary \u2500\u2500")
    lines.append(f"  Total pipeline time: {total_time:.2f}s")
    lines.append(f"  Total files: {len(all_files)}")
    lines.append(f"  Total size:  {_format_size(total_size)}")
    lines.append("")

    # Full file manifest
    lines.append("\u2500\u2500 Full File Manifest \u2500\u2500")
    for rel, sz in all_files:
        lines.append(f"  {rel:<40s} {_format_size(sz):>10s}")
    lines.append("")

    report_path = sd / "run_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Run report written to {report_path}")


def _write_run_info_json(
    session_dir: str,
    session_id: str,
    model_name: str,
    prompt: str | None,
    img_path: str,
    timestamp: str,
    timings: dict[str, float],
    artifacts: dict[str, list[str]],
    fov_deg: float | None,
    status: str,
    clip_lambda: float | None = None,
    neutral_prompt: str | None = None,
    dtype: str | None = None,
    num_steps: int | None = None,
    vc_params: dict | None = None,
    mesh_status: str | None = None,
    mesh_error: str | None = None,
) -> dict:
    """Write a machine-readable run_info.json mirroring run_report.txt.

    Returns the dict so callers can forward a subset to sessions_index.jsonl.
    """
    import json
    from PIL import Image as PILImage

    sd = Path(session_dir)
    view_files = artifacts.get("nvs", [])
    num_views = len(view_files)

    try:
        with PILImage.open(img_path) as img:
            w, h = img.size
    except Exception:
        w, h = None, None

    try:
        input_bytes = Path(img_path).stat().st_size
    except Exception:
        input_bytes = None

    def _file_entry(rel: str) -> dict:
        fp = sd / rel
        return {
            "path": rel,
            "bytes": fp.stat().st_size if fp.exists() else None,
            "exists": fp.exists(),
        }

    # Trajectory sidecar (SEVA c2ws/Ks/fovs or ViewCrafter parametric)
    trajectory_rel: str | None = None
    if (sd / "views" / "trajectory.json").exists():
        trajectory_rel = "views/trajectory.json"

    # nvs_params block
    nvs_params: dict = {
        "prompt": prompt,
        "clip_lambda": clip_lambda,
        "neutral_prompt": neutral_prompt,
        "num_views": num_views,
    }
    if model_name == "seva":
        nvs_params["dtype"] = dtype
        nvs_params["num_steps"] = num_steps
    if model_name == "viewcrafter" and vc_params is not None:
        nvs_params["vc_params"] = vc_params

    # Stage dicts
    depth_info = {
        "fov_deg": fov_deg,
        "files": [_file_entry(f) for f in artifacts.get("depth", [])],
    }
    views_info = {
        "count": num_views,
        "files": [_file_entry(f) for f in view_files],
        "trajectory_json": trajectory_rel,
    }
    recon_files = artifacts.get("reconstruction", [])
    recon_info = {
        "files": [_file_entry(f) for f in recon_files],
        "glb": recon_files[0] if recon_files else None,
    }
    mesh_files = artifacts.get("mesh", [])
    mesh_info: dict = {
        "status": mesh_status,
        "files": [_file_entry(f) for f in mesh_files],
        "glb": mesh_files[0] if mesh_files else None,
    }
    if mesh_error:
        mesh_info["error_message"] = mesh_error
    comp_info = {
        "files": [_file_entry(f) for f in artifacts.get("composition", [])],
        "scene_json": "scene.json" if (sd / "scene.json").exists() else None,
    }

    timings_info = {
        "model_load": timings.get("model_load"),
        "depth": timings.get("depth"),
        "nvs": timings.get("nvs"),
        "reconstruction": timings.get("reconstruction"),
        "mesh": timings.get("mesh"),
        "composition": timings.get("composition"),
        "total": timings.get("total"),
    }

    manifest = [
        {"path": rel, "bytes": sz}
        for rel, sz in _list_files(session_dir, session_dir)
    ]

    info = {
        "session_id": session_id,
        "timestamp": timestamp,
        "status": status,
        "model_name": model_name,
        "git_sha": _git_sha(),
        "input": {
            "path": os.path.basename(img_path),
            "filename": os.path.basename(img_path),
            "width": w,
            "height": h,
            "bytes": input_bytes,
        },
        "nvs_params": nvs_params,
        "depth": depth_info,
        "views": views_info,
        "reconstruction": recon_info,
        "mesh": mesh_info,
        "composition": comp_info,
        "timings_seconds": timings_info,
        "file_manifest": manifest,
    }

    info_path = sd / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    logger.info(f"Run info JSON written to {info_path}")
    return info


def _append_sessions_index(
    session_id: str,
    session_dir: str,
    timestamp: str,
    model_name: str,
    img_path: str,
    prompt: str | None,
    clip_lambda: float | None,
    neutral_prompt: str | None,
    dtype: str | None,
    num_steps: int | None,
    fov_deg: float | None,
    num_views: int,
    total_time_s: float | None,
    mesh_status: str | None,
    status: str,
) -> None:
    """Append one compact JSON line per completed run to uploads/sessions_index.jsonl.

    Same atomic-append pattern as backend/experiments/artifact_layout.py::append_master_row.
    Called on both success and error branches so failures still show up.
    """
    import json
    row = {
        "timestamp": timestamp,
        "session_id": session_id,
        "model_name": model_name,
        "input_filename": os.path.basename(img_path),
        "prompt": prompt,
        "clip_lambda": clip_lambda,
        "neutral_prompt": neutral_prompt,
        "dtype": dtype,
        "num_steps": num_steps,
        "fov_deg": fov_deg,
        "num_views": num_views,
        "total_time_s": total_time_s,
        "mesh_status": mesh_status,
        "status": status,
        "session_dir": session_dir,
    }
    index_path = Path("uploads") / "sessions_index.jsonl"
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to append {index_path}: {exc}")


async def run_scene_pipeline(session_id: str, session_dir: str, img_path: str,
                             model_name: str = "svd", prompt: str = None,
                             clip_lambda: float | None = None,
                             neutral_prompt: str | None = None):
    """Execute the full NVS + 3D reconstruction pipeline."""
    import time
    import glob as _glob
    from datetime import datetime

    timings: dict[str, float] = {}
    artifacts: dict[str, list[str]] = {}
    fov_deg: float | None = None
    timestamp = datetime.now().isoformat()

    # Fields populated throughout the pipeline that need to reach the report
    # writers AND the error branch below.  Initialised here so the except
    # handler can always emit a well-formed run_info.json / sessions row.
    nvs_dtype: str | None = None
    nvs_steps: int | None = None
    nvs_neutral: str | None = neutral_prompt
    vc_params: dict | None = None
    mesh_status: str | None = None
    mesh_error: str | None = None

    try:
        t_pipeline = time.time()

        depth_model = global_model_instances["depth-pro"]
        dust3r_model = global_model_instances["dust3r"]

        # Lazy-load the chosen synthesizer (keeps VRAM free until needed)
        synth_key = f"synth-{model_name}"
        t0 = time.time()
        if synth_key not in global_model_instances:
            logger.info(f"Loading synthesizer '{model_name}' for the first time \u2026")
            synthesizer = get_synthesizer(model_name)
            synthesizer.load_model(device="cuda")
            global_model_instances[synth_key] = synthesizer
        synthesizer = global_model_instances[synth_key]
        timings["model_load"] = time.time() - t0
        logger.info(f"[Timer] Synthesizer load: {timings['model_load']:.2f}s")

        # Capture SEVA-specific state (dtype / num_steps / default neutral
        # prompt).  Non-SEVA backends leave these as None via getattr default.
        nvs_dtype = getattr(synthesizer, "dtype_name", None)
        nvs_steps = getattr(synthesizer, "num_steps", None)
        nvs_neutral = (
            neutral_prompt
            if neutral_prompt is not None
            else getattr(synthesizer, "neutral_prompt", None)
        )

        processor = SceneProcessor(depth_model, dust3r_model, synthesizer)

        # 1. Depth Estimation
        update_status(session_id, "processing", 10, "Estimating Depth")
        t0 = time.time()
        depth_arr = await processor.estimate_depth(img_path, session_dir)
        timings["depth"] = time.time() - t0
        logger.info(f"[Timer] Depth Estimation: {timings['depth']:.2f}s")

        fov_deg = processor.scene_metadata.get("fov")
        depth_files = []
        for name in ("depth.png", "depth_metric.npy", "depth.npy"):
            if (Path(session_dir) / name).exists():
                depth_files.append(name)
        artifacts["depth"] = depth_files

        # 2. Novel View Synthesis
        update_status(session_id, "processing", 30, "Generating Novel Views")
        t0 = time.time()
        needs_vram_offload = model_name in ("viewcrafter", "panodreamer", "zero123pp", "sv3d", "seva")
        if needs_vram_offload:
            import torch
            logger.info(
                f"Offloading DepthPro + project DUSt3R to CPU for {model_name} VRAM headroom \u2026"
            )
            depth_model.set_device("cpu")
            dust3r_model.set_device("cpu")
            torch.cuda.empty_cache()
        try:
            await processor.generate_novel_views(
                img_path,
                session_dir,
                depth_map=depth_arr,
                num_views=10,
                prompt=prompt,
                clip_lambda=clip_lambda,
                neutral_prompt=neutral_prompt,
            )
        finally:
            if needs_vram_offload:
                import torch
                depth_model.set_device("cuda")
                dust3r_model.set_device("cuda")
                torch.cuda.empty_cache()
        timings["nvs"] = time.time() - t0
        logger.info(f"[Timer] Novel View Synthesis: {timings['nvs']:.2f}s")

        # ViewCrafter stashes its exact run config on the instance so we can
        # log it post-hoc.  SEVA just uses dtype_name / num_steps captured above.
        vc_params = getattr(synthesizer, "last_run_params", None)

        views_dir = f"{session_dir}/views"
        view_pngs = sorted(_glob.glob(os.path.join(views_dir, "*.png")))
        artifacts["nvs"] = [
            f"views/{Path(f).name}" for f in view_pngs
        ]

        # 3. 3D Reconstruction (Dust3r)
        update_status(session_id, "processing", 60, "Reconstructing 3D Point Cloud")
        t0 = time.time()
        await processor.reconstruct_3d(views_dir, session_dir)
        timings["reconstruction"] = time.time() - t0
        logger.info(f"[Timer] 3D Reconstruction: {timings['reconstruction']:.2f}s")

        recon_files = []
        for name in ("scene.glb",):
            if (Path(session_dir) / name).exists():
                recon_files.append(name)
        artifacts["reconstruction"] = recon_files

        # 5. Mesh Generation (soft-fail; DUSt3R clouds are often sparse so
        # ball-pivoting occasionally fails — we continue with the valid
        # scene.glb + views rather than killing the whole session).
        update_status(session_id, "processing", 75, "Generating Solid Mesh")
        mesh_status = "skipped"
        mesh_error = None
        t0 = time.time()
        scene_glb = Path(session_dir) / "scene.glb"
        mesh_glb = Path(session_dir) / "scene_mesh.glb"
        if not scene_glb.exists():
            mesh_status = "skipped"
            mesh_error = "scene.glb not produced by Stage 3"
            logger.warning("Stage 5: scene.glb missing, skipping mesh generation.")
        else:
            try:
                from reconstruction.mesh_generator import point_cloud_to_mesh
                point_cloud_to_mesh(
                    glb_path=str(scene_glb),
                    output_path=str(mesh_glb),
                    algo="bpa",
                    outlier_std=1.2,
                )
                mesh_status = "ok" if mesh_glb.exists() else "failed"
                if mesh_status == "failed":
                    mesh_error = "point_cloud_to_mesh returned without writing output"
            except Exception as e:
                mesh_status = "failed"
                mesh_error = str(e)
                logger.warning(
                    f"Stage 5 (mesh generation) failed: {e}", exc_info=True
                )
        timings["mesh"] = time.time() - t0
        logger.info(
            f"[Timer] Mesh Generation: {timings['mesh']:.2f}s (status={mesh_status})"
        )

        artifacts["mesh"] = []
        if mesh_glb.exists():
            artifacts["mesh"].append("scene_mesh.glb")

        # 4. Scene Composition
        update_status(session_id, "processing", 90, "Composing Scene")
        t0 = time.time()
        processor.compose_scene(session_id, session_dir)
        timings["composition"] = time.time() - t0
        logger.info(f"[Timer] Scene Composition: {timings['composition']:.2f}s")

        artifacts["composition"] = ["scene.json"]

        timings["total"] = time.time() - t_pipeline
        logger.info(f"[Timer] Total pipeline: {timings['total']:.2f}s")

        _write_run_report(
            session_dir=session_dir,
            session_id=session_id,
            model_name=model_name,
            prompt=prompt,
            img_path=img_path,
            timestamp=timestamp,
            timings=timings,
            artifacts=artifacts,
            fov_deg=fov_deg,
            clip_lambda=clip_lambda,
            neutral_prompt=nvs_neutral,
            dtype=nvs_dtype,
            num_steps=nvs_steps,
            vc_params=vc_params,
            mesh_status=mesh_status,
            mesh_error=mesh_error,
        )
        _write_run_info_json(
            session_dir=session_dir,
            session_id=session_id,
            model_name=model_name,
            prompt=prompt,
            img_path=img_path,
            timestamp=timestamp,
            timings=timings,
            artifacts=artifacts,
            fov_deg=fov_deg,
            status="complete",
            clip_lambda=clip_lambda,
            neutral_prompt=nvs_neutral,
            dtype=nvs_dtype,
            num_steps=nvs_steps,
            vc_params=vc_params,
            mesh_status=mesh_status,
            mesh_error=mesh_error,
        )
        _append_sessions_index(
            session_id=session_id,
            session_dir=session_dir,
            timestamp=timestamp,
            model_name=model_name,
            img_path=img_path,
            prompt=prompt,
            clip_lambda=clip_lambda,
            neutral_prompt=nvs_neutral,
            dtype=nvs_dtype,
            num_steps=nvs_steps,
            fov_deg=fov_deg,
            num_views=len(artifacts.get("nvs", [])),
            total_time_s=timings.get("total"),
            mesh_status=mesh_status,
            status="complete",
        )

        update_status(session_id, "complete", 100, "Done")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_status(session_id, "error", 0, str(e))
        # Emit a best-effort run_info.json + sessions_index row even on
        # failure so partial artifacts are still discoverable post-download.
        try:
            _write_run_info_json(
                session_dir=session_dir,
                session_id=session_id,
                model_name=model_name,
                prompt=prompt,
                img_path=img_path,
                timestamp=timestamp,
                timings=timings,
                artifacts=artifacts,
                fov_deg=fov_deg,
                status=f"error: {e}",
                clip_lambda=clip_lambda,
                neutral_prompt=nvs_neutral,
                dtype=nvs_dtype,
                num_steps=nvs_steps,
                vc_params=vc_params,
                mesh_status=mesh_status,
                mesh_error=mesh_error,
            )
        except Exception as inner:  # pragma: no cover - defensive
            logger.warning(f"Failed to write error-branch run_info.json: {inner}")
        try:
            _append_sessions_index(
                session_id=session_id,
                session_dir=session_dir,
                timestamp=timestamp,
                model_name=model_name,
                img_path=img_path,
                prompt=prompt,
                clip_lambda=clip_lambda,
                neutral_prompt=nvs_neutral,
                dtype=nvs_dtype,
                num_steps=nvs_steps,
                fov_deg=fov_deg,
                num_views=len(artifacts.get("nvs", [])),
                total_time_s=timings.get("total"),
                mesh_status=mesh_status,
                status=f"error: {e}",
            )
        except Exception as inner:  # pragma: no cover - defensive
            logger.warning(f"Failed to append error-branch sessions_index row: {inner}")


@app.get("/views/{session_id}")
async def get_views(session_id: str):
    """Return the list of generated view image URLs for a session."""
    import glob as _glob
    views_dir = Path(f"uploads/{session_id}/views")
    if not views_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"No views found for session '{session_id}'"},
        )
    view_files = sorted(_glob.glob(str(views_dir / "view_*.png")))
    urls = [f"/uploads/{session_id}/views/{Path(f).name}" for f in view_files]
    return {"session_id": session_id, "views": urls, "count": len(urls)}


@app.get("/app")
async def serve_app():
    """Serve the frontend GUI."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/")
async def root():
    return {
        "name": "AI 3D Scene Reconstructor",
        "version": "4.0.0",
        "available_models": AVAILABLE_MODELS,
        "endpoints": {
            "/app": "GET - Open the GUI in your browser",
            "/process": "POST - Upload image (form fields: file, model, prompt, clip_lambda, neutral_prompt)",
            "/status/{session_id}": "GET - Check status",
            "/views/{session_id}": "GET - List generated view image URLs",
            "/uploads/{session_id}/*": "GET - Assets",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9876)