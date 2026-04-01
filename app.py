import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Header
st.markdown("""
# ❤️ Heart Disease Prediction System
### 🧠 AI-powered health risk assessment
""")

st.markdown("---")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This app predicts the likelihood of heart disease based on patient health data.
- Model: Random Forest
- Accuracy: ~85%
""")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Patient Details")

    age = st.slider("Age", 1, 120, 25)
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)

    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

with col2:
    st.subheader("📊 Medical Parameters")

    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 60, 220, 150)

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2])

st.markdown("---")

# Prediction Button
if st.button("🚀 Predict Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    risk = probability[0][1]

    st.markdown("## 📈 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk ({risk*100:.2f}% probability)")
        st.progress(int(risk * 100))
    else:
        st.success(f"✅ Low Risk ({(1-risk)*100:.2f}% confidence)")
        st.progress(int((1 - risk) * 100))

    st.markdown("---")
    st.caption("⚡ Built with Streamlit | Machine Learning Model: Random Forest")