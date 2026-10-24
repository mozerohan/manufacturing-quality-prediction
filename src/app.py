
from pathlib import Path

import joblib
from fastapi import FastAPI

from pydantic import BaseModel, Field


# 1. Find the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Build the path to the saved ML model
MODEL_PATH = PROJECT_ROOT / "models" / "final_quality_model.joblib"

# 3. Check that the model file exists
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}"
    )

# 4. Load the trained model
model = joblib.load(MODEL_PATH)

# 5. Create the FastAPI application
app = FastAPI(
    title="Manufacturing Quality Prediction API",
    description=(
        "An educational API for predicting manufacturing quality "
        "using a model trained on synthetic data."
    ),
    version="1.0.0",
)


# Define the manufacturing data expected by the API
class ManufacturingInput(BaseModel):
    Rated_Power_VA: float = Field(gt=0)
    Input_Voltage_V: float = Field(gt=0)
    Winding_Time_Min: float = Field(gt=0)
    Assembly_Time_Min: float = Field(gt=0)
    Core_Weight_Kg: float = Field(gt=0)
    Copper_Weight_Kg: float = Field(gt=0)
    Winding_Temperature_C: float = Field(ge=0)

    Product_Type: str
    Production_Line: str
    Shift: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Manufacturing Quality Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "disclaimer": (
            "Educational demo using synthetic manufacturing data. "
            "Not validated for real-world quality decisions."
        ),
    }


# Health-check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_quality(data: ManufacturingInput):
    # Convert the validated request into a dictionary
    payload = (
        data.model_dump()
    )

    # Convert the dictionary into a one-row DataFrame
    import pandas as pd

    input_df = pd.DataFrame([payload])

    # Generate the predicted quality label
    prediction = model.predict(input_df)[0]

    # Generate model-estimated probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Match each probability to its class label
    probability_by_class = {
        str(label): round(float(probability), 4)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "prediction": str(prediction),
        "probabilities": probability_by_class,
        "disclaimer": (
            "Educational demo using synthetic data. "
            "Probabilities are model estimates and may not be calibrated."
        ),
    }
