"""
app/api.py
==========
Production FastAPI backend for Face Mask & Helmet Detection.

Runs inference on uploaded images using the trained Keras models.
Exposes a /predict endpoint and a /health check.

Author : Bilal Ahmed (231980028) — GIFT University
"""

import os
import sys
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from utils.data_loader import TASK_CONFIG

app = FastAPI(
    title="Safety Compliance API",
    description="Backend API for Face Mask & Helmet Detection",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Model Caching / Loading
# ─────────────────────────────────────────────
MODELS = {}

def get_model(task: str):
    """Load model if not already cached."""
    if task not in TASK_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}")

    if task in MODELS:
        return MODELS[task]

    config = TASK_CONFIG[task]
    model_path = config["model_path"]

    if not os.path.exists(model_path):
        # Graceful fallback or raising exception
        raise HTTPException(
            status_code=503,
            detail=f"Model for task '{task}' not found at {model_path}. Please train/load it first."
        )

    try:
        MODELS[task] = tf.keras.models.load_model(model_path)
        return MODELS[task]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model '{task}': {str(e)}"
        )


# ─────────────────────────────────────────────
# Response Schema
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    task: str
    class_names: list
    prediction_label: str
    confidence: float
    raw_probability: float
    is_compliant: bool
    status: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint for container environments / orchestrators."""
    loaded_models = {task: os.path.exists(cfg["model_path"]) for task, cfg in TASK_CONFIG.items()}
    return {
        "status": "healthy",
        "models_status": loaded_models,
        "tensorflow_version": tf.__version__
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    task: str = Query("mask", enum=["mask", "helmet"]),
    file: UploadFile = File(...)
):
    """
    Accepts an uploaded image and runs binary classification for safety compliance.
    """
    # 1. Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Allowed: JPG, JPEG, PNG, BMP, WEBP"
        )

    # 2. Get the model
    model = get_model(task)
    class_names = TASK_CONFIG[task]["class_names"]

    try:
        # 3. Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 224x224 (MobileNetV2 size)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Preprocess using MobileNetV2 scaling: (x / 127.5) - 1.0
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # 4. Predict
        prob = float(model.predict(img_batch, verbose=0)[0][0])
        pred_idx = int(prob >= 0.5)
        pred_label = class_names[pred_idx]
        confidence = prob if pred_idx == 1 else 1.0 - prob

        # 5. Check compliance rules
        # Mask task: 'with_mask' (index 0 or 1 depending on class_names order) is compliant.
        # Let's inspect class names: ["with_mask", "without_mask"]
        # Index 0 is with_mask.
        # Helmet task: ["helmet", "no_helmet"] -> Index 0 is helmet.
        is_compliant = False
        if task == "mask" and pred_label == "with_mask":
            is_compliant = True
        elif task == "helmet" and pred_label == "helmet":
            is_compliant = True

        status_msg = "Compliant" if is_compliant else "Violation"

        return PredictionResponse(
            task=task,
            class_names=class_names,
            prediction_label=pred_label,
            confidence=round(confidence, 4),
            raw_probability=round(prob, 4),
            is_compliant=is_compliant,
            status=status_msg
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
