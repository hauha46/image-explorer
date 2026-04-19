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
from dust3r_reconstructor import Dust3rReconstructor
from contextlib import asynccontextmanager
from depth_pro_estimator import DepthProEstimator
from synthesizers import get_synthesizer, AVAILABLE_MODELS

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
):
    """
    Upload image and start 3D scene reconstruction.

    ``model`` selects the NVS backend: ``svd`` | ``viewcrafter`` | ``vivid`` | ``panodreamer`` | ``zero123pp`` | ``sv3d`` | ``seva``.
    ``prompt`` is an optional text prompt (used by ViewCrafter and PanoDreamer for scene guidance;
    ignored by SVD, Zero123++, SV3D, and SEVA).
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
        run_scene_pipeline, session_id, str(session_dir), str(img_path), model, prompt
    )

    return {"session_id": session_id, "model": model, "prompt": prompt}


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


async def run_scene_pipeline(session_id: str, session_dir: str, img_path: str,
                             model_name: str = "svd", prompt: str = None):
    """Execute the full NVS + 3D reconstruction pipeline."""
    import time
    import glob as _glob
    from datetime import datetime

    timings: dict[str, float] = {}
    artifacts: dict[str, list[str]] = {}
    fov_deg: float | None = None
    timestamp = datetime.now().isoformat()

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
                img_path, session_dir, depth_map=depth_arr, num_views=10, prompt=prompt
            )
        finally:
            if needs_vram_offload:
                import torch
                depth_model.set_device("cuda")
                dust3r_model.set_device("cuda")
                torch.cuda.empty_cache()
        timings["nvs"] = time.time() - t0
        logger.info(f"[Timer] Novel View Synthesis: {timings['nvs']:.2f}s")

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
        )

        update_status(session_id, "complete", 100, "Done")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_status(session_id, "error", 0, str(e))


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
            "/process": "POST - Upload image (form fields: file, model, prompt)",
            "/status/{session_id}": "GET - Check status",
            "/views/{session_id}": "GET - List generated view image URLs",
            "/uploads/{session_id}/*": "GET - Assets",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9876)