# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 19:52:00 2025

@author: chetan
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


scaler = pickle.load(open('C:/Users/chetan/OneDrive/Desktop/saved_ones/multiple_disease/diabetes_scaler.sav', 'rb'))
diabetes_model=pickle.load(open('C:/Users/chetan/OneDrive/Desktop/saved_ones/multiple_disease/diabetis_pred.sav', 'rb'))


heart_model=pickle.load(open('C:/Users/chetan/OneDrive/Desktop/saved_ones/multiple_disease/heart_prediction.sav' , 'rb'))

parkinson_model=pickle.load(open('C:/Users/chetan/OneDrive/Desktop/saved_ones/multiple_disease/parkinson_dataTodo.sav' , 'rb'))
parkinson_scaler = pickle.load(open(
    'C:/Users/chetan/OneDrive/Desktop/saved_ones/multiple_disease/parkinson_scaler.sav',
    'rb'
))


with st.sidebar:
    
    selected=option_menu('Multiple disease prediction system' , 
                         ['Diabetis prediction' , 
                          'Heart disease prediction' ,
                          'parkinson prediction'],
                         
                         icons=['activity','heart','person-fill'],
                         
                         
                         default_index=0)
    
if (selected=='Diabetis prediction'):
    
    st.title('Diabetis prediction using ML')
    
    col1 , col2 , col3=st.columns(3)
    
    with col1:
        
        Pregnancies=st.text_input('enter the no of pregnancies')
        
    with col2:
        
        Glucose=st.text_input('enter the glucose level')
        
    with col3:
        
        BloodPressure=st.text_input('enter the blood pressure level')
        
    with col1:
        
        SkinThickness=st.text_input('enter the skin thickness')
        
    with col2:
        
        Insulin=st.text_input('enter the insulin level')
        
    with col3:
        
        BMI=st.text_input('enter the BMI')
        
    with col1:
        
        DiabetesPedigreeFunction=st.text_input('enter the diabetis pedigree function')
        
    with col2:
        
        Age=st.text_input('enter the Age')
        
    diagnosis=''
    
    if st.button('Diabetis test result'):
        try:
            user_input = [[
                int(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                int(Age)
                        ]]

            user_input_scaled = scaler.transform(user_input)
            prediction = diabetes_model.predict(user_input_scaled)

            if prediction[0] == 1:
                diagnosis = 'The person is diabetic'
            else:
                diagnosis = 'The person is not diabetic'

        except ValueError:
            diagnosis = '⚠️ Please enter valid numeric values in all fields'

    st.success(diagnosis)

    
    
if (selected == 'Heart disease prediction'):

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        sex = st.text_input('sex')
        cp = st.text_input('cp')
        trestbps = st.text_input('trestbps')
        chol = st.text_input('chol')

    with col2:
        fbs = st.text_input('fbs')
        restecg = st.text_input('restecg')
        thalach = st.text_input('thalach')
        exang = st.text_input('exang')
        oldpeak = st.text_input('oldpeak')

    with col3:
        slope= st.text_input('slope')
        ca = st.text_input('ca')
        thal = st.text_input('thal')

    heart_diagnosis = ''

    if st.button('Heart Test Result'):
        try:
            user_input = [[
                float(age),
                float(sex),
                float(cp),
                float(trestbps),
                float(chol),
                float(fbs),
                float(restecg),
                float(thalach),
                float(exang),
                float(oldpeak),
                float(slope),
                float(ca),
                float(thal)
            ]]

            prediction = heart_model.predict(user_input)

            if prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'

        except ValueError:
            heart_diagnosis = '⚠️ Please enter valid numeric values in all fields'

    st.success(heart_diagnosis)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if (selected=='parkinson prediction'):
    
    st.title('parkinson prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:

        mdvp_fo = st.number_input('MDVP:Fo(Hz)', format="%.5f")
        mdvp_fhi = st.number_input('MDVP:Fhi(Hz)', format="%.5f")
        mdvp_flo = st.number_input('MDVP:Flo(Hz)', format="%.5f")
        
        jitter_percent = st.number_input('MDVP:Jitter(%)', format="%.6f")
        jitter_abs = st.number_input('MDVP:Jitter(Abs)', format="%.6f")
        rap = st.number_input('MDVP:RAP', format="%.6f")
        ppq = st.number_input('MDVP:PPQ', format="%.6f")
        ddp = st.number_input('Jitter:DDP', format="%.6f")
    
    with col2:
        
        shimmer = st.number_input('MDVP:Shimmer', format="%.6f")
        shimmer_db = st.number_input('MDVP:Shimmer(dB)', format="%.6f")
        apq3 = st.number_input('Shimmer:APQ3', format="%.6f")
        apq5 = st.number_input('Shimmer:APQ5', format="%.6f")
        apq = st.number_input('MDVP:APQ', format="%.6f")
        dda = st.number_input('Shimmer:DDA', format="%.6f")
        
        nhr = st.number_input('NHR', format="%.6f")
        hnr = st.number_input('HNR', format="%.6f")
        
    with col3:
        rpde = st.number_input('RPDE', format="%.6f")
        dfa = st.number_input('DFA', format="%.6f")
        
        spread1 = st.number_input('spread1', format="%.6f")
        spread2 = st.number_input('spread2', format="%.6f")
        
        d2 = st.number_input('D2', format="%.6f")
        ppe = st.number_input('PPE', format="%.6f")


    parkinson_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [[
                mdvp_fo, mdvp_fhi, mdvp_flo,
                jitter_percent, jitter_abs,
                rap, ppq, ddp,
                shimmer, shimmer_db,
                apq3, apq5, apq, dda,
                nhr, hnr,
                rpde, dfa,
                spread1, spread2,
                d2, ppe
            ]]



            user_input_scaled = parkinson_scaler.transform(user_input)
            prediction = parkinson_model.predict(user_input_scaled)

            if prediction[0] == 1:
                parkinson_diagnosis = "The person has Parkinson’s disease"
            else:
                parkinson_diagnosis = "The person does NOT have Parkinson’s disease"

        except ValueError:
            parkinson_diagnosis = "⚠️ Please enter valid numeric values in all fields"

    st.success(parkinson_diagnosis)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    