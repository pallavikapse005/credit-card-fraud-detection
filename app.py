import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Custom CSS for Black + Purple Theme
st.markdown("""
<style>
body {
    background-color: #0f0f1a;
    color: white;
}
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a0033);
}
h1, h2, h3 {
    color: #bb86fc;
    text-align: center;
}
.stButton>button {
    background-color: #7b2cbf;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #9d4edd;
    color: black;
}
.stNumberInput label {
    color: #e0aaff;
}
.result-safe {
    background-color: #1b4332;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}
.result-fraud {
    background-color: #6a040f;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3>Machine Learning Based Fraud Prediction System</h3>", unsafe_allow_html=True)

st.write("---")

# Input section
st.subheader("🔢 Enter Transaction Details")

input_data = []

# PCA feature inputs
with st.expander("📌 Transaction Features (V1 – V28)", expanded=False):
    for i in range(1, 29):
        value = st.number_input(
            f"V{i}",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )
        input_data.append(value)

# Amount input
amount = st.number_input(
    "💰 Transaction Amount",
    min_value=0.0,
    max_value=100000.0,
    value=100.0,
    step=10.0
)
input_data.append(amount)

st.write("---")

# Predict button
if st.button("🔍 Predict Transaction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-fraud'>⚠️ Fraudulent Transaction Detected!</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-safe'>✅ Legitimate Transaction</div>",
            unsafe_allow_html=True
        )

# Footer
st.write("---")
st.markdown(
    "<center style='color:#c77dff;'>Developed using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
