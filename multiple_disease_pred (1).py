# multiple_disease_pred.py
# Clean & corrected Streamlit app (Cloud-safe, Python 3.13 compatible)

import os
import pickle
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide"
)

st.title("🩺 Multiple Disease Prediction System")

# --------------------------------------------------
# Utility: load pickle safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------
# Load models
# --------------------------------------------------
try:
    diabetes_scaler = load_pickle("diabetes_scaler.sav")
    diabetes_model = load_pickle("diabetis_pred.sav")
    heart_model = load_pickle("heart_prediction.sav")
    parkinson_scaler = load_pickle("parkinson_scaler.sav")
    parkinson_model = load_pickle("parkinson_dataTodo.sav")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# --------------------------------------------------
# Sidebar navigation (NO external libraries)
# --------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    selected = st.radio(
        "Select Prediction",
        ["Diabetes", "Heart Disease", "Parkinson"]
    )

# ==================================================
# DIABETES
# ==================================================
if selected == "Diabetes":
    st.subheader("🧪 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose", min_value=0.0)
    with col2:
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
    with col3:
        insulin = st.number_input("Insulin", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)

    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1, step=1)

    if st.button("Predict Diabetes"):
        user_input = [[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]]
        scaled = diabetes_scaler.transform(user_input)
        pred = diabetes_model.predict(scaled)

        if int(pred[0]) == 1:
            st.error("❌ Diabetic")
        else:
            st.success("✅ Not Diabetic")

# ==================================================
# HEART DISEASE
# ==================================================
elif selected == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, step=1)
        sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1)
        cp = st.number_input("Chest Pain Type", min_value=0)
        trestbps = st.number_input("Resting BP", min_value=0.0)
        chol = st.number_input("Cholesterol", min_value=0.0)
    with col2:
        fbs = st.number_input("Fasting Blood Sugar", min_value=0)
        restecg = st.number_input("Rest ECG", min_value=0)
        thalach = st.number_input("Max Heart Rate", min_value=0.0)
        exang = st.number_input("Exercise Induced Angina", min_value=0, max_value=1)
        oldpeak = st.number_input("Oldpeak", min_value=0.0)
    with col3:
        slope = st.number_input("Slope", min_value=0)
        ca = st.number_input("CA", min_value=0)
        thal = st.number_input("Thal", min_value=0)

    if st.button("Predict Heart Disease"):
        user_input = [[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]]
        pred = heart_model.predict(user_input)

        if int(pred[0]) == 1:
            st.error("❌ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")

# ==================================================
# PARKINSON
# ==================================================
else:
    st.subheader("🧠 Parkinson Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", format="%.5f")
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", format="%.5f")
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", format="%.5f")
        jitter_percent = st.number_input("Jitter (%)", format="%.6f")
        jitter_abs = st.number_input("Jitter (Abs)", format="%.6f")
        rap = st.number_input("RAP", format="%.6f")
        ppq = st.number_input("PPQ", format="%.6f")
        ddp = st.number_input("DDP", format="%.6f")
    with col2:
        shimmer = st.number_input("Shimmer", format="%.6f")
        shimmer_db = st.number_input("Shimmer (dB)", format="%.6f")
        apq3 = st.number_input("APQ3", format="%.6f")
        apq5 = st.number_input("APQ5", format="%.6f")
        apq = st.number_input("APQ", format="%.6f")
        dda = st.number_input("DDA", format="%.6f")
        nhr = st.number_input("NHR", format="%.6f")
        hnr = st.number_input("HNR", format="%.6f")
    with col3:
        rpde = st.number_input("RPDE", format="%.6f")
        dfa = st.number_input("DFA", format="%.6f")
        spread1 = st.number_input("Spread1", format="%.6f")
        spread2 = st.number_input("Spread2", format="%.6f")
        d2 = st.number_input("D2", format="%.6f")
        ppe = st.number_input("PPE", format="%.6f")

    if st.button("Predict Parkinson"):
        user_input = [[
            mdvp_fo, mdvp_fhi, mdvp_flo, jitter_percent, jitter_abs,
            rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5,
            apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]]
        scaled = parkinson_scaler.transform(user_input)
        pred = parkinson_model.predict(scaled)

        if int(pred[0]) == 1:
            st.error("❌ Parkinson Detected")
        else:
            st.success("✅ No Parkinson Detected")
