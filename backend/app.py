"""
AI-Enhanced 3D Scene Reconstruction API

Decomposes a single image into a 3D scene with:
1. Depth Estimation (DepthPro)
2. Novel View Synthesis (SVD / ViewCrafter / VIVID)
3. 3D Reconstruction (Dust3r)
4. Scene Composition
"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    # region agent log
    import torch, json, time, os as _os; _dbg=r"c:\Users\bcliu\Documents\Northeastern\DeepLearning_Jiang\Final-Project\howie_dan_section\.cursor\debug.log"; _os.makedirs(_os.path.dirname(_dbg),exist_ok=True)
    _d={"location":"app.py:lifespan","message":"torch_cuda_info","data":{"cuda_available":torch.cuda.is_available(),"torch_version":torch.__version__,"cuda_version":getattr(torch.version,"cuda","none"),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A","arch_list":str(getattr(torch.cuda,"get_arch_list",lambda:[])())},"timestamp":int(time.time()*1000),"runId":"run2","hypothesisId":"A"}
    with open(_dbg,"a") as _f: _f.write(json.dumps(_d)+"\n")
    # endregion
    print("Loading Dust3r model into memory...")
    global_model_instances["dust3r"] = Dust3rReconstructor()
    # region agent log
    _d2={"location":"app.py:lifespan:post-dust3r","message":"dust3r_loaded","data":{"success":True},"timestamp":int(time.time()*1000),"runId":"run2","hypothesisId":"B"}
    with open(_dbg,"a") as _f: _f.write(json.dumps(_d2)+"\n")
    # endregion
    try:
        global_model_instances["depth-pro"] = DepthProEstimator()
        # region agent log
        _d3={"location":"app.py:lifespan:post-depthpro","message":"depthpro_loaded","data":{"success":True},"timestamp":int(time.time()*1000),"runId":"run2","hypothesisId":"C"}
        with open(_dbg,"a") as _f: _f.write(json.dumps(_d3)+"\n")
        # endregion
    except Exception as _e:
        # region agent log
        _d4={"location":"app.py:lifespan:depthpro-error","message":"depthpro_failed","data":{"error":str(_e)[:500]},"timestamp":int(time.time()*1000),"runId":"run2","hypothesisId":"C"}
        with open(_dbg,"a") as _f: _f.write(json.dumps(_d4)+"\n")
        # endregion
        raise

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

    ``model`` selects the NVS backend: ``svd`` | ``viewcrafter`` | ``vivid``.
    ``prompt`` is an optional text prompt used by ViewCrafter for scene guidance.
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


async def run_scene_pipeline(session_id: str, session_dir: str, img_path: str,
                             model_name: str = "svd", prompt: str = None):
    """Execute the full NVS + 3D reconstruction pipeline."""
    import time
    try:
        t_pipeline = time.time()
        depth_model = global_model_instances["depth-pro"]
        dust3r_model = global_model_instances["dust3r"]

        # Lazy-load the chosen synthesizer (keeps VRAM free until needed)
        synth_key = f"synth-{model_name}"
        t0 = time.time()
        if synth_key not in global_model_instances:
            logger.info(f"Loading synthesizer '{model_name}' for the first time …")
            synthesizer = get_synthesizer(model_name)
            synthesizer.load_model(device="cuda")
            global_model_instances[synth_key] = synthesizer
        synthesizer = global_model_instances[synth_key]
        logger.info(f"[Timer] Synthesizer load: {time.time() - t0:.2f}s")

        processor = SceneProcessor(depth_model, dust3r_model, synthesizer)

        # 1. Depth Estimation
        update_status(session_id, "processing", 10, "Estimating Depth")
        t0 = time.time()
        depth_arr = await processor.estimate_depth(img_path, session_dir)
        logger.info(f"[Timer] Depth Estimation: {time.time() - t0:.2f}s")

        # 2. Novel View Synthesis
        update_status(session_id, "processing", 30, "Generating Novel Views")
        t0 = time.time()
        if model_name == "viewcrafter":
            import torch
            logger.info(
                "Offloading DepthPro + project DUSt3R to CPU for ViewCrafter VRAM headroom …"
            )
            depth_model.set_device("cpu")
            dust3r_model.set_device("cpu")
            torch.cuda.empty_cache()
        try:
            await processor.generate_novel_views(
                img_path, session_dir, depth_map=depth_arr, num_views=2, prompt=prompt
            )
        finally:
            if model_name == "viewcrafter":
                import torch
                depth_model.set_device("cuda")
                dust3r_model.set_device("cuda")
                torch.cuda.empty_cache()
        logger.info(f"[Timer] Novel View Synthesis: {time.time() - t0:.2f}s")

        # 3. 3D Reconstruction (Dust3r)
        update_status(session_id, "processing", 60, "Reconstructing 3D Point Cloud")
        t0 = time.time()
        views_dir = f"{session_dir}/views"
        await processor.reconstruct_3d(views_dir, session_dir)
        logger.info(f"[Timer] 3D Reconstruction: {time.time() - t0:.2f}s")

        # 4. Scene Composition
        update_status(session_id, "processing", 90, "Composing Scene")
        t0 = time.time()
        processor.compose_scene(session_id, session_dir)
        logger.info(f"[Timer] Scene Composition: {time.time() - t0:.2f}s")

        logger.info(f"[Timer] Total pipeline: {time.time() - t_pipeline:.2f}s")
        update_status(session_id, "complete", 100, "Done")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_status(session_id, "error", 0, str(e))


@app.get("/")
async def root():
    return {
        "name": "AI 3D Scene Reconstructor",
        "version": "4.0.0",
        "available_models": AVAILABLE_MODELS,
        "endpoints": {
            "/process": "POST - Upload image (form fields: file, model)",
            "/status/{session_id}": "GET - Check status",
            "/uploads/{session_id}/*": "GET - Assets",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9876)