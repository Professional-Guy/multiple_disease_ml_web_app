# multiple_disease_pred.py
# Clean, Cloud-safe version (NO external UI libraries)

import os
import pickle
import streamlit as st

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 Multiple Disease Prediction System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

diabetes_scaler = load_pickle(os.path.join(BASE_DIR, "diabetes_scaler.sav"))
diabetes_model  = load_pickle(os.path.join(BASE_DIR, "diabetis_pred.sav"))
heart_model     = load_pickle(os.path.join(BASE_DIR, "heart_prediction.sav"))
parkinson_scaler = load_pickle(os.path.join(BASE_DIR, "parkinson_scaler.sav"))
parkinson_model  = load_pickle(os.path.join(BASE_DIR, "parkinson_dataTodo.sav"))

with st.sidebar:
    st.header("Select Prediction Type")
    selected = st.radio(
        "Choose Disease",
        ("Diabetes Prediction", "Heart Disease Prediction", "Parkinson Prediction")
    )

if selected == "Diabetes Prediction":
    st.subheader("🧪 Diabetes Prediction Using ML")
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input("Pregnancies")
        glucose = st.text_input("Glucose")
    with col2:
        blood_pressure = st.text_input("Blood Pressure")
        skin_thickness = st.text_input("Skin Thickness")
    with col3:
        insulin = st.text_input("Insulin")
        bmi = st.text_input("BMI")

    col4, col5 = st.columns(2)
    with col4:
        dpf = st.text_input("Diabetes Pedigree Function")
    with col5:
        age = st.text_input("Age")

    if st.button("Predict Diabetes"):
        try:
            user_input = [[
                int(pregnancies), float(glucose), float(blood_pressure),
                float(skin_thickness), float(insulin),
                float(bmi), float(dpf), int(age)
            ]]
            scaled = diabetes_scaler.transform(user_input)
            prediction = diabetes_model.predict(scaled)
            st.success("Result: " + ("Diabetic" if prediction[0] == 1 else "Not Diabetic"))
        except ValueError:
            st.warning("Enter valid numeric values")

elif selected == "Heart Disease Prediction":
    st.subheader("❤️ Heart Disease Prediction Using ML")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
        sex = st.text_input("Sex")
        cp = st.text_input("Chest Pain")
        trestbps = st.text_input("Rest BP")
        chol = st.text_input("Cholesterol")
    with col2:
        fbs = st.text_input("FBS")
        restecg = st.text_input("Rest ECG")
        thalach = st.text_input("Max HR")
        exang = st.text_input("Exang")
        oldpeak = st.text_input("Oldpeak")
    with col3:
        slope = st.text_input("Slope")
        ca = st.text_input("CA")
        thal = st.text_input("Thal")

    if st.button("Predict Heart Disease"):
        try:
            user_input = [[
                float(age), float(sex), float(cp), float(trestbps),
                float(chol), float(fbs), float(restecg),
                float(thalach), float(exang), float(oldpeak),
                float(slope), float(ca), float(thal)
            ]]
            prediction = heart_model.predict(user_input)
            st.success("Result: " + ("Heart Disease" if prediction[0] == 1 else "Healthy"))
        except ValueError:
            st.warning("Enter valid numeric values")

else:
    st.subheader("🧠 Parkinson Prediction Using ML")
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
            rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda,
            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]]
        scaled = parkinson_scaler.transform(user_input)
        prediction = parkinson_model.predict(scaled)
        st.success("Result: " + ("Parkinson Detected" if prediction[0] == 1 else "No Parkinson"))
