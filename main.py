# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from model import heart_disease_model
import pandas as pd

app = FastAPI()

class PredictionData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def create_prediction(data: PredictionData):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, 
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, 
        data.ca, data.thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
        'ca', 'thal'
    ])

    # Make prediction using the trained model
    prediction, probability = heart_disease_model.predict(input_data)

    # Calculate risk factor
    risk_factor = calculate_risk_factor(data)
    prediction_summary = summarize_prediction(data)

    return {
        "message": "Prediction data processed successfully!",
        "data": data,
        "prediction": {
            "prediction": int(prediction),
            "probability": probability  # Probability as a percentage
        },
        "risk_factor": risk_factor,
        "summary": prediction_summary
    }

def calculate_risk_factor(prediction: PredictionData) -> float:
    # Simplified example calculation
    risk_factor = (prediction.chol / prediction.trestbps) + (prediction.age / 50)
    return risk_factor

def summarize_prediction(prediction: PredictionData) -> str:
    summary = (
        f"Age: {prediction.age}\n"
        f"Sex: {'Male' if prediction.sex == 1 else 'Female'}\n"
        f"Chest Pain Type: {prediction.cp}\n"
        f"Resting Blood Pressure: {prediction.trestbps}\n"
        f"Cholesterol Level: {prediction.chol}\n"
        f"Max Heart Rate: {prediction.thalach}\n"
        f"Oldpeak: {prediction.oldpeak}\n"
    )
    return summary

# Run the application using the following command:
# uvicorn main:app --reload
