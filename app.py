import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
knn = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter your health details below and click Predict.")

# Input fields for user data
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar (> 120 mg/dL, 1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", value=0.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Make Prediction
if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, 
                            exang, oldpeak, slope, ca, thal]])
    user_input_scaled = scaler.transform(user_input)  # Scale input
    prediction = knn.predict(user_input_scaled)[0]
    
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    st.subheader(f"Prediction: {result}")