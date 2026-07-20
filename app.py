from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

from tensorflow.keras.models import load_model


app = Flask(__name__)


# ==========================
# Project Base Path
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================
# Load ANN Model
# ==========================

model = load_model(
    os.path.join(BASE_DIR, "loan_risk_ann_model.keras")
)


# ==========================
# Load Scaler
# ==========================

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)



# ==========================
# Home Page
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )



# ==========================
# Prediction Route
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Numerical Inputs

        applicant_income = float(
            request.form["ApplicantIncome"]
        )

        coapplicant_income = float(
            request.form["CoapplicantIncome"]
        )

        loan_amount = float(
            request.form["LoanAmount"]
        )

        loan_term = float(
            request.form["Loan_Amount_Term"]
        )

        credit_history = int(
            request.form["Credit_History"]
        )


        # Categorical Inputs

        gender = request.form["Gender"]

        married = request.form["Married"]

        dependents = request.form["Dependents"]

        education = request.form["Education"]

        self_employed = request.form["Self_Employed"]

        property_area = request.form["Property_Area"]



        # Create DataFrame

        input_data = pd.DataFrame({

            "ApplicantIncome":[applicant_income],

            "CoapplicantIncome":[coapplicant_income],

            "LoanAmount":[loan_amount],

            "Loan_Amount_Term":[loan_term],

            "Credit_History":[credit_history],


            "Gender_Male":[
                1 if gender=="Male" else 0
            ],


            "Married_Yes":[
                1 if married=="Yes" else 0
            ],


            "Dependents_1":[
                1 if dependents=="1" else 0
            ],


            "Dependents_2":[
                1 if dependents=="2" else 0
            ],


            "Dependents_3+":[
                1 if dependents=="3+" else 0
            ],


            "Education_Not Graduate":[
                1 if education=="Not Graduate" else 0
            ],


            "Self_Employed_Yes":[
                1 if self_employed=="Yes" else 0
            ],


            "Property_Area_Semiurban":[
                1 if property_area=="Semiurban" else 0
            ],


            "Property_Area_Urban":[
                1 if property_area=="Urban" else 0
            ]

        })



        # Same order as training

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



        # Scaling

        input_scaled = scaler.transform(
            input_data
        )



        # Prediction

        probability = model.predict(
            input_scaled
        )[0][0]


        confidence = round(
            float(probability)*100,
            2
        )



        # Result

        if probability >= 0.5:

            result = "High Risk"

            message = (
                "Customer has higher chances of loan default. "
                "Bank should review carefully."
            )


        else:

            result = "Low Risk"

            message = (
                "Customer is likely to repay the loan. "
                "Application can be considered."
            )



        return render_template(

            "index.html",

            prediction=result,

            confidence=confidence,

            message=message

        )



    except Exception as e:


        return render_template(

            "index.html",

            prediction="Error",

            confidence=0,

            message=str(e)

        )




# ==========================
# Run App
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )