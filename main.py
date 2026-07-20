from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Loan Risk Prediction API",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = load_model(
    os.path.join(BASE_DIR, "loan_risk_ann_model.keras")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)


class LoanInput(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str


@app.get("/")
def home():
    return {
        "message": "Loan Risk Prediction API is Running"
    }


@app.post("/predict")
def predict(data: LoanInput):

    input_data = pd.DataFrame([{

        "ApplicantIncome": data.ApplicantIncome,
        "CoapplicantIncome": data.CoapplicantIncome,
        "LoanAmount": data.LoanAmount,
        "Loan_Amount_Term": data.Loan_Amount_Term,
        "Credit_History": data.Credit_History,

        "Gender_Male": 1 if data.Gender == "Male" else 0,
        "Married_Yes": 1 if data.Married == "Yes" else 0,

        "Dependents_1": 1 if data.Dependents == "1" else 0,
        "Dependents_2": 1 if data.Dependents == "2" else 0,
        "Dependents_3+": 1 if data.Dependents == "3+" else 0,

        "Education_Not Graduate": 1 if data.Education == "Not Graduate" else 0,

        "Self_Employed_Yes": 1 if data.Self_Employed == "Yes" else 0,

        "Property_Area_Semiurban": 1 if data.Property_Area == "Semiurban" else 0,

        "Property_Area_Urban": 1 if data.Property_Area == "Urban" else 0

    }])

    columns = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Gender_Male',
        'Married_Yes',
        'Dependents_1',
        'Dependents_2',
        'Dependents_3+',
        'Education_Not Graduate',
        'Self_Employed_Yes',
        'Property_Area_Semiurban',
        'Property_Area_Urban'
    ]

    input_data = input_data[columns]

    input_scaled = scaler.transform(input_data)

    probability = float(model.predict(input_scaled)[0][0])

    if probability >= 0.5:
        prediction = "High Risk"
    else:
        prediction = "Low Risk"

    return {
        "Prediction": prediction,
        "Probability": round(probability, 4),
        "Confidence": round(probability * 100, 2)
    }