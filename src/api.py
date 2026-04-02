from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import pickle
import json
import zipfile
import tempfile
from pathlib import Path
import subprocess
import asyncio
from typing import List
import time

# --- THE MONKEY PATCH ---
class SafeDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)
# ------------------------

app = FastAPI(title="Image Classification API", version="1.0.0")

# Global variables for model and config
model = None
class_names = None
preprocess_config = None
MODEL_PATH = "models/final_model.h5"
CLASS_NAMES_PATH = "models/class_names.pkl"
CONFIG_PATH = "models/preprocess_config.json"

# Model uptime tracking
start_time = time.time()

@app.on_event("startup")
async def load_model():
    global model, class_names, preprocess_config
    try:
        # --- APPLIED THE PATCH HERE ---
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'Dense': SafeDense})
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        with open(CONFIG_PATH, 'r') as f:
            preprocess_config = json.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    # Resize image
    target_size = tuple(preprocess_config['image_size'])
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
async def root():
    return {"message": "Image Classification API is running", "status": "active"}

@app.get("/health")
async def health_check():
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict single image"""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {"class": class_names[idx], "confidence": float(predictions[0][idx])}
            for idx in top_3_idx
        ]
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "success": True
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-bulk")
async def upload_bulk_data(files: List[UploadFile] = File(...)):
    """Upload multiple images for retraining"""
    try:
        upload_dir = "data/new_training_data/"
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.zip')):
                file_path = os.path.join(upload_dir, file.filename)
                contents = await file.read()
                
                # Handle zip files
                if file.filename.endswith('.zip'):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zip_path = os.path.join(tmpdir, file.filename)
                        with open(zip_path, 'wb') as f:
                            f.write(contents)
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(upload_dir)
                    uploaded_files.append(f"{file.filename} (extracted)")
                else:
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    uploaded_files.append(file.filename)
        
        return JSONResponse({
            "message": f"Successfully uploaded {len(uploaded_files)} items",
            "files": uploaded_files,
            "location": upload_dir
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Trigger model retraining in background"""
    background_tasks.add_task(run_retraining)
    return {
        "message": "Retraining started in background",
        "status": "processing",
        "estimated_time": "5-10 minutes"
    }

async def run_retraining():
    """Background task for model retraining"""
    try:
        # Call the retraining script
        result = subprocess.run(
            ["python", "../src/train_model.py", "--retrain"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            # Reload the new model
            await load_model()
            print("Model retrained and reloaded successfully")
        else:
            print(f"Retraining failed: {result.stderr}")
    
    except Exception as e:
        print(f"Error during retraining: {e}")

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    return {
        "model_path": MODEL_PATH,
        "classes": class_names,
        "num_classes": len(class_names),
        "input_shape": preprocess_config['image_size']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)