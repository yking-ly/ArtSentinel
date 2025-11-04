# app/main.py
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import time
import os
import io
from typing import Optional

# ✅ timm-based inference with switching / hot-reload
from app.model_timm_infer import get_active_model, load_model

app = FastAPI(title="ArtSentinel API")

MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}  # soft check only

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- helpers ----------
def safe_image_open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))  # header check
        img.verify()
        img = Image.open(io.BytesIO(data))  # reopen for decode
        return img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="unsupported_type")

# Pretty-name map (normalize any legacy labels)
PRETTY = {
    "AiArtData": "Bot-Made",
    "AI Art": "Bot-Made",
    "AI_Art": "Bot-Made",
    "RealArt": "Brush-Made",
    "Real Art": "Brush-Made",
    "Real_Art": "Brush-Made",
    "Bot-Made": "Bot-Made",
    "Brush-Made": "Brush-Made",
}

# -------- routes ----------
@app.get("/health")
def health():
    info = {}
    try:
        # Hints the model to load if not yet loaded (using env or auto-detect)
        m = get_active_model()
        info = {
            "modelName": getattr(m, "model_name", "unknown"),
            "inputSize": getattr(m, "input_size", None),
        }
    except Exception:
        info = {"modelName": "unavailable"}
    return {
        "status": "ok",
        "modelVersion": MODEL_VERSION,
        **info,
        "serverTimeMs": int(time.time() * 1000),
    }

@app.get("/model-info")
def model_info():
    """
    Inspect which checkpoint is active.
    """
    m = get_active_model()
    return {
        "status": "ok",
        "checkpoint": getattr(m, "checkpoint_path", "unknown"),
        "modelName": getattr(m, "model_name", "unknown"),
        "temperature": getattr(m, "temperature", 1.0),
        "inputSize": getattr(m, "input_size", None),
        "classNames": ["Bot-Made", "Brush-Made"],
        "modelVersion": MODEL_VERSION,
    }

@app.post("/reload")
def reload_model(path: Optional[str] = Body(None, embed=True)):
    """
    Hot-swap to a different .pth at runtime.
    Example body:
      { "path": "C:\\ArtSentinel\\models\\runs\\convnext_tiny\\best_model.pth" }
    If no path provided, will try MODEL_PATH env or auto-detect newest under models/runs.
    """
    try:
        new_path = path or os.getenv("MODEL_PATH") or None
        m = load_model(new_path)  # if None -> auto-detect newest
        return {
            "ok": True,
            "loaded": getattr(m, "checkpoint_path", new_path),
            "modelName": getattr(m, "model_name", "unknown"),
            "inputSize": getattr(m, "input_size", None),
            "temperature": getattr(m, "temperature", 1.0),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"reload_error:{e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Soft MIME check (kept permissive for curl)
    if file.content_type and (file.content_type not in ALLOWED_MIME):
        raise HTTPException(status_code=400, detail="unsupported_mime")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty_file")

    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"file_too_large:{MAX_IMAGE_MB}MB")

    # Validate & decode
    try:
        image = safe_image_open(data)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="unsupported_type")

    started = time.time()
    try:
        model = get_active_model()
        out = model.predict(image)  # {label, confidence, all_probs, model_name, temperature, ...}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_error:{e}")

    ms = int((time.time() - started) * 1000)

    # Normalize label + per-class probs to Bot-Made / Brush-Made
    label = PRETTY.get(out.get("label", ""), out.get("label", ""))
    probs_in = out.get("all_probs", {})
    all_probs = {PRETTY.get(k, k): v for k, v in probs_in.items()}

    return {
        "label": label,
        "score": round(float(out.get("confidence", 0.0)), 4),
        "classNames": ["Bot-Made", "Brush-Made"],  # always these in UI
        "modelVersion": MODEL_VERSION,
        "processingMs": ms,
        "details": {
            "all_probs": all_probs,
            "model": out.get("model_name", "unknown"),
            "temperature": out.get("temperature", 1.0),
            "input_size": out.get("input_size", None),
            "checkpoint": out.get("checkpoint_path", None),
        },
    }
