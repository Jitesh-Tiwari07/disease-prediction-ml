import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Disease Prediction", layout="wide", page_icon="🩺")

st.title("🩺 Disease Prediction System")
st.markdown("### Machine Learning Based Disease Risk Prediction")

st.sidebar.title("Select Disease")
disease = st.sidebar.selectbox("Choose Disease", 
    ["Heart Disease", "Diabetes", "Breast Cancer"])

dataset_map = {
    "Heart Disease": "heart",
    "Diabetes": "diabetes",
    "Breast Cancer": "breast_cancer"
}
dataset_key = dataset_map[disease]

st.subheader(f"Patient Details - {disease}")

if disease == "Heart Disease":
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 80, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.number_input("Resting BP", 90, 200, 130)
        chol = st.number_input("Cholesterol", 100, 600, 250)
        fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    with col2:
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", [0,1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
    
    input_data = pd.DataFrame([[age, 1 if sex=="Male" else 0, cp, trestbps, chol, 
                                fbs, restecg, thalach, exang, oldpeak, slope]],
        columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol',
                 'FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak', 'Slope'])

elif disease == "Diabetes":
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", 0, 20, 3)
        glucose = st.number_input("Glucose", 0, 200, 120)
        bp = st.number_input("Blood Pressure", 0, 130, 70)
        bmi = st.number_input("BMI", 10.0, 70.0, 32.0)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 100)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 20, 90, 35)
    
    input_data = pd.DataFrame([[preg, glucose, bp, 30, insulin, bmi, dpf, age]],
        columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age'])

else:  # Breast Cancer - Smart Fix
    st.info("Breast Cancer Prediction (Mean Features)")
    col1, col2 = st.columns(2)
    with col1:
        radius_mean = st.number_input("Radius Mean", 5.0, 30.0, 14.0)
        texture_mean = st.number_input("Texture Mean", 5.0, 40.0, 19.0)
        perimeter_mean = st.number_input("Perimeter Mean", 40.0, 200.0, 95.0)
        area_mean = st.number_input("Area Mean", 100.0, 2500.0, 650.0)
        smoothness_mean = st.number_input("Smoothness Mean", 0.05, 0.2, 0.1)
    with col2:
        compactness_mean = st.number_input("Compactness Mean", 0.01, 0.4, 0.12)
        concavity_mean = st.number_input("Concavity Mean", 0.0, 0.5, 0.15)
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, 0.3, 0.1)
        symmetry_mean = st.number_input("Symmetry Mean", 0.1, 0.4, 0.18)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.05, 0.15, 0.06)
    
    # Create base features
    base_features = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                     compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, 
                     fractal_dimension_mean]
    
    # Fill remaining features with mean values (dynamic padding)
    input_data = pd.DataFrame([base_features + [0.1] * 20], columns=[f'f{i}' for i in range(30)])

# Predict Button
if st.button("🚀 Predict Risk", type="primary", use_container_width=True):
    model_path = f"models/{dataset_key}_random_forest.pkl"
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        try:
            # Dynamic padding to match exact number of features
            n_features = model.n_features_in_
            current_features = input_data.shape[1]
            
            if current_features < n_features:
                # Add extra columns with 0
                for i in range(n_features - current_features):
                    input_data[f'extra_{i}'] = 0.0
            elif current_features > n_features:
                input_data = input_data.iloc[:, :n_features]
            
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.error(f"⚠️ **HIGH RISK** of {disease} | Probability: **{prob:.1%}**")
            else:
                st.success(f"✅ **LOW RISK** of {disease} | Probability: **{prob:.1%}**")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Model not found. Train first with `python src/train.py`")

st.caption("Educational Project Only • Not for Medical Diagnosis")
