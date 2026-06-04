from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn

from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)

# Load registered model
model = mlflow.sklearn.load_model(
    "models:/credit_risk_model/1"
)


@app.get("/")
def home():
    return {"message": "Credit Risk Model API Running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])

    probability = model.predict_proba(input_df)[0][1]

    return PredictionResponse(
        risk_probability=float(probability)
    )
