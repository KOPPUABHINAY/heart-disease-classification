import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown("## ❤️ Heart Disease Prediction System")
st.write("Enter patient details below to predict risk:")

# Input layout (2 columns for better UI)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    restecg = st.selectbox("Rest ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0–2)", [0, 1, 2])

# Prediction button
if st.button("🔍 Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write(f"Confidence: {(1 - probability[0][1])*100:.2f}%")