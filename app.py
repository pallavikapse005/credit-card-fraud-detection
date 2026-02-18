import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it is **Fraudulent or Legitimate**.")

# Input fields
input_data = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0)
    input_data.append(value)

amount = st.number_input("Transaction Amount", value=0.0)
input_data.append(amount)

# Prediction
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
