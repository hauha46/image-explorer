"""
AI-Enhanced 3D Scene Reconstruction API

Decomposes a single image into a 3D scene with:
1. Spatial Understanding (Depth)
2. Object Separation (Segmentation)
3. 3D Object Reconstruction (TripoSR)
4. Background Inpainting
"""
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core components
from depth_estimator import DepthEstimator

# Global depth estimator reference
_depth_estimator = None


class ProcessingStatus(BaseModel):
    session_id: str
    status: str
    progress: int
    current_step: str
    steps_completed: List[str]


# In-memory status tracking
processing_status = {}


app = FastAPI(
    title="Image to 3D Scene Reconstruction",
    description="Decompose single images into fully navigable 3D scenes with valid object geometry",
    version="3.0.1"
)

# Initialize depth estimator (always needed)
try:
    estimator = DepthEstimator(model_size="large")
except Exception as e:
    logger.warning(f"Failed to load large depth model, falling back to small: {e}")
    estimator = DepthEstimator(model_size="small")

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
async def process_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload image and start 3D scene reconstruction.
    Returns session_id immediately.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = Path(f"uploads/{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    img_path = session_dir / "input.jpg"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Resize if too large (LaMa/TripoSR struggle with very high-res images)
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
    
    logger.info(f"Starting scene reconstruction v3.0.1: session={session_id}")
    update_status(session_id, "processing", 0, "Initializing")
    
    # Start background processing
    background_tasks.add_task(run_scene_pipeline, session_id, str(session_dir), str(img_path))
    
    return {"session_id": session_id}


async def run_scene_pipeline(session_id: str, session_dir: str, img_path: str):
    """Execute Layered Depth Scene pipeline (2.5D)."""
    try:
        from scene_processor import SceneProcessor
        
        # Initialize processor with the global estimator
        processor = SceneProcessor(estimator)
        
        # 1. Depth Estimation
        update_status(session_id, "processing", 10, "Estimating Depth")
        depth_arr = await processor.estimate_depth(img_path, session_dir)
        
        # 2. Object Detection & Segmentation
        update_status(session_id, "processing", 30, "Analyzing Scene")
        objects = await processor.detect_and_segment(img_path, session_dir)
        
        # 3. Background Inpainting
        update_status(session_id, "processing", 50, "Cleaning Background")
        bg_path = await processor.inpaint_background(img_path, objects, session_dir)
        
        # 4. Layer Extraction (instead of 3D gen)
        update_status(session_id, "processing", 70, "Extracting Layers")
        layers = await processor.extract_object_layers(img_path, objects, session_dir)
        
        # 5. Scene Composition
        update_status(session_id, "processing", 90, "Composing Scene")
        scene_data = processor.compose_scene(session_id, session_dir, bg_path, layers)
        
        update_status(session_id, "complete", 100, "Done")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        update_status(session_id, "error", 0, str(e))


@app.get("/")
async def root():
    return {
        "name": "AI 3D Scene Reconstructor",
        "version": "3.0.1",
        "endpoints": {
            "/process": "POST - Upload image",
            "/status/{session_id}": "GET - Check status",
            "/uploads/{session_id}/*": "GET - Assets"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)