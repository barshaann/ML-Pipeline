from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import os
import math

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
    tca: float = Field(..., description="Total coarse aggregate")
    rca: float = Field(..., description="Recycled coarse aggregate")
    w_c: float = Field(..., description="Water to cement ratio")
    shape_factor: Optional[float] = None
    density: Optional[float] = None
    slump: Optional[float] = None
    width_dia: Optional[float] = None
    length: Optional[float] = None
    cs_area: float = Field(..., description="Cross-sectional area (required for calculating strength)")

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

    required_fields = ["cement", "sand", "water", "w_c", "cs_area", "tca"]
    for field in required_fields:
        val = data_dict.get(field)
        if val is None or val <= 0:
            raise HTTPException(status_code=422, detail=f"Invalid input: '{field}' must be greater than zero.")
            
    nca_val = data_dict.get("nca")
    if nca_val is None or nca_val < 0:
        raise HTTPException(status_code=422, detail="Invalid input: 'nca' cannot be negative.")

    # --- Ultimate Physics Fix for RCA ---
    # To completely neutralize the spurious dataset correlation, 
    # we force the model to evaluate the mix as pristine (RCA = 0.0).
    # In a pristine mix, the Natural Aggregate is equal to the Total Coarse Aggregate (tca).
    true_rca = data_dict.get("rca", 0.0)
    model_input_data = data_dict.copy()
    model_input_data["rca"] = 0.0  # Always pass 0.0 to ML model
    model_input_data["nca"] = data_dict.get("tca", model_input_data["nca"]) # Set NCA to Total Coarse Aggregate
    
    # Remove tca before passing to the model pipeline since it's not a model feature
    if "tca" in model_input_data:
        del model_input_data["tca"]
        
    df = pd.DataFrame([model_input_data])
    try:
        prediction = model_pipeline.predict(df)
        ultimate_load = float(prediction[0])
        
        # Now apply realistic Civil Engineering empirical penalty
        penalty = 0.0
        if true_rca <= 0.40:
            # 0 to 40% RCA: Fluctuating drop using cosine wave 
            # Ensures 0 is absolute max, but 20% might be higher than 10% or 30%
            # Drop reaches ~6.76% at exactly 40% RCA
            penalty = (true_rca * 0.10) + 0.015 * (1 - math.cos(true_rca * 25))
        else:
            # > 40% RCA: Steep sharp drop (up to ~41% drop at 100% RCA)
            # 0.0676 is approx the penalty exactly at 0.40
            penalty = 0.0676 + ((true_rca - 0.40) * 0.57)
            
        ultimate_load = ultimate_load * (1 - penalty)
        # ------------------------------------

        # Calculate Compressive Strength: Ultimate Load (kN) * 1000 / CS Area (mm2)
        strength = (ultimate_load * 1000) / data_dict["cs_area"]
        return {
            "predicted_ultimate_load": round(ultimate_load, 2),
            "predicted_strength": round(strength, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
