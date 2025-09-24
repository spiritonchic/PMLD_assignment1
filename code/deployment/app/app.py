import streamlit as st
import requests

# App title
st.title("Diabetes Prediction App")

st.write("Enter patient parameters to predict diabetes probability:")

# Input fields for features
pregnancies = st.number_input("Pregnancies", min_value=0.0, step=1.0)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0.0, step=1.0)

# Button to send the request
if st.button("Predict"):
    # Prepare JSON payload for API
    input_data = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "BMI": bmi,
        "diabetes_pedigree_function": diabetes_pedigree_function,
        "age": age
    }

    # Send POST request to FastAPI
    try:
        response = requests.post("http://api:8000/predict", json=input_data)
        result = response.json()

        st.subheader("Prediction Result:")
        # st.write(f"Prediction (0=No, 1=Diabetes): {result['prediction']}")
        st.write(f"Has diabetes: {result['has_diabete']}")
        st.write(f"Probability: {result['probability']:.2f}")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")
