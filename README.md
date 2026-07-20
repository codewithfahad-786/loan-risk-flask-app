# 🏦 Loan Risk Prediction System using Artificial Neural Network (ANN)

A Machine Learning and Deep Learning based web application that predicts whether a loan applicant is **High Risk** or **Low Risk** using an Artificial Neural Network (ANN). The project includes a Flask web application for the user interface and a FastAPI REST API for prediction.

---

## 📌 Project Overview

Banks receive thousands of loan applications every day. This project helps financial institutions assess the risk of loan default by analyzing applicant information and predicting the likelihood of repayment.

---

## 🚀 Features

- Predict Loan Risk (High Risk / Low Risk)
- Artificial Neural Network (ANN) Model
- Flask Web Application
- FastAPI Prediction API
- Responsive User Interface
- Data Preprocessing using StandardScaler
- Prediction Confidence Score
- REST API Support

---

## 🛠️ Technologies Used

- Python
- Flask
- FastAPI
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- HTML5
- CSS3
- Jinja2

---

## 📂 Project Structure

```
loan-risk-flask-app/
│
├── app.py
├── loan_risk_ann_model.keras
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── api/
│   └── main.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## 📊 Input Features

The model uses the following customer information:

- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Gender
- Married
- Dependents
- Education
- Self Employed
- Property Area

---

## 🎯 Output

The system predicts:

- ✅ High Risk
- ✅ Low Risk

It also returns:

- Prediction Probability
- Confidence Percentage

---

## ▶️ How to Run the Flask App

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/loan-risk-flask-app.git
```

### Move to Project Folder

```bash
cd loan-risk-flask-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Flask Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

# 🚀 FastAPI API

Run the API:

```bash
uvicorn api.main:app --reload
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 📈 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Feature Scaling
5. ANN Model Training
6. Model Evaluation
7. Model Saving
8. Flask Deployment
9. FastAPI Deployment

---

## 📸 Screenshots

### Home Page

(Add Screenshot Here)

### Prediction Result

(Add Screenshot Here)

### FastAPI Swagger UI

(Add Screenshot Here)

---

## 👨‍💻 Author

**Muhammad Fahad**

Machine Learning & Deep Learning Engineer

GitHub:
https://github.com/YOUR_USERNAME

LinkedIn:
(Add Your LinkedIn Profile)

---

## 📄 License

This project is developed for educational and portfolio purposes.
