import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ================== CUSTOM THEME (BLACK + PURPLE) ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a0033);
    color: white;
}
h1, h2, h3 {
    color: #c77dff;
    text-align: center;
}
.stNumberInput label {
    color: #e0aaff;
}
.stButton>button {
    background-color: #7b2cbf;
    color: white;
    border-radius: 12px;
    font-size: 18px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #9d4edd;
    color: black;
}
div[data-testid="stExpander"] {
    background-color: #240046;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
# ===================================================================

# Title
st.markdown("<h1>💳 Credit Card Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<center style='color:#e0aaff;'>Enter transaction behavior details</center>",
    unsafe_allow_html=True
)

# User-friendly feature names (mapped to V1–V28)
feature_names = [
    "Transaction Speed Deviation",
    "Transaction Frequency Score",
    "Merchant Risk Indicator",
    "Location Change Score",
    "Spending Pattern Shift",
    "Device Usage Pattern",
    "Transaction Timing Anomaly",
    "Purchase Behavior Consistency",
    "Account Activity Score",
    "Unusual Transaction Indicator",
    "Customer Risk Profile",
    "Payment Method Stability",
    "Transaction Confidence Score",
    "Historical Fraud Similarity",
    "Transaction Trust Index",
    "Behavioral Risk Factor",
    "Security Signal Strength",
    "Usage Irregularity Score",
    "Transaction Reliability",
    "Risk Correlation Index",
    "Behavior Deviation Level",
    "Fraud Pattern Match Score",
    "Purchase Location Trust",
    "Spending Regularity Index",
    "Account Stability Score",
    "Transaction Validation Level",
    "Risk Amplification Factor",
    "Anomaly Confidence Score"
]

input_data = []

# Feature inputs
with st.expander("📌 Transaction Behavior Features"):
    for feature in feature_names:
        value = st.number_input(
            feature,
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

# Prediction
if st.button("🔍 Predict Transaction"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

# Footer
st.markdown(
    "<center style='color:#c77dff;'>Developed using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)

