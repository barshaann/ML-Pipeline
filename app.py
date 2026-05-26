from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(title="Strength Prediction API", 
              description="API for predicting material strength using a trained ML pipeline.")

MODEL_PATH = "best_model_pipeline.joblib"
model_pipeline = None

try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Please run ml_pipeline.py first.")
except Exception as e:
    print(f"Failed to load model: {e}")

class PredictionInput(BaseModel):
    cement: float = Field(..., description="Amount of cement")
    sand: float = Field(..., description="Amount of sand")
    water: float = Field(..., description="Amount of water")
    nca: float = Field(..., description="Natural coarse aggregate")
    rca: float = Field(..., description="Recycled coarse aggregate")
    w_c: float = Field(..., description="Water to cement ratio")
    shape_factor: Optional[float] = None
    density: Optional[float] = None
    slump: Optional[float] = None
    width_dia: Optional[float] = None
    length: Optional[float] = None
    cs_area: Optional[float] = None
    ultimate_load: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found in the current directory.</h1>", status_code=404)

@app.get("/concrete-bg.png")
async def get_bg_image():
    if os.path.exists("concrete-bg.png"):
        return FileResponse("concrete-bg.png")
    return HTMLResponse(status_code=404)

@app.get("/health")
def health_check():
    if model_pipeline is None:
        return {"status": "error", "message": "Model not loaded. Train the model using ml_pipeline.py first."}
    return {"status": "ok", "message": "API and model are ready."}

@app.post("/predict")
def predict_strength(input_data: PredictionInput):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server. Please run the training script first.")

    data_dict = input_data.model_dump()

    required_fields = ["cement", "sand", "water", "nca", "w_c"]
    for field in required_fields:
        val = data_dict.get(field)
        if val is None or val <= 0:
            raise HTTPException(status_code=422, detail=f"Invalid input: '{field}' must be greater than zero.")

    df = pd.DataFrame([data_dict])
    try:
        prediction = model_pipeline.predict(df)
        return {"predicted_strength": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
