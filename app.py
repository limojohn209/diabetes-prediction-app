import streamlit as st
import pandas as pd
import joblib

# Load model na scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction App")
st.write("Weka taarifa za mgonjwa ili kupata utabiri wa uwezekano wa kisukari.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    sample = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                             insulin, bmi, dpf, age]],
                           columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                    'Insulin','BMI','DiabetesPedigreeFunction','Age'])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Matokeo: Ana uwezekano wa Kisukari (Probability: {probability:.2%})")
    else:
        st.success(f"Matokeo: Hana Kisukari (Probability: {probability:.2%})")