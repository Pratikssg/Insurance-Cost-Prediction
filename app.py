# app.py
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # <-- Import StaticFiles
from pydantic import BaseModel

# Initialize the FastAPI app and templates
app = FastAPI(title="Insurance Cost Prediction Website")

# --- Mount the static directory ---
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load the trained pipeline
pipeline = joblib.load("model/pipeline.joblib")


# --- Feature Engineering Function ---
def add_interaction_features(X):
    X_mod = X.copy()
    smoker_binary = X_mod['smoker'].map({'yes': 1, 'no': 0})
    X_mod['smoker_age_interaction'] = smoker_binary * X_mod['age']
    X_mod['smoker_bmi_interaction'] = smoker_binary * X_mod['bmi']
    return X_mod


# --- Pydantic Model for API Input ---
class InputFeatures(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# --- Page Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "page": "home"})


@app.get("/predictor", response_class=HTMLResponse)
async def predictor(request: Request):
    return templates.TemplateResponse("predictor.html", {"request": request, "page": "predictor"})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "page": "about"})


# --- Prediction API Endpoint ---
@app.post("/predict")
def predict(features: InputFeatures):
    data = features.dict()
    input_df = pd.DataFrame([data])
    input_df_featured = add_interaction_features(input_df)

    prediction = pipeline.predict(input_df_featured)
    original_prediction = round(float(np.expm1(prediction[0])), 2)

    return {"predicted_charge": original_prediction}