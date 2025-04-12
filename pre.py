from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from model_utils import fine_tune_clip, predict_class
from PIL import Image
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/fine-tune/")
async def fine_tune(cat: UploadFile = File(...), dog: UploadFile = File(...)):
    cat_img = Image.open(io.BytesIO(await cat.read())).convert("RGB")
    dog_img = Image.open(io.BytesIO(await dog.read())).convert("RGB")
    fine_tune_clip(cat_img, dog_img)
    return {"status": "success", "message": "Model fine-tuned!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    label, confidence, heatmap_path = predict_class(image)
    filename = os.path.basename(heatmap_path)
    return {
        "label": label,
        "confidence": confidence,
        "heatmap": f"/heatmap/{filename}"
    }

@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str):
    return FileResponse(f"heatmaps/{filename}")
