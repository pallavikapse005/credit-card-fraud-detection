import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ===================== CUSTOM DARK THEME =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f0f1a, #1a0033);
    color: white;
}
h1, h2, h3 {
    color: #c77dff;
}
.sidebar .sidebar-content {
    background-color: #10002b;
}
.stButton>button {
    background-color: #7b2cbf;
    color: white;
    border-radius: 12px;
    font-size: 16px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #9d4edd;
    color: black;
}
.metric-box {
    background-color: #240046;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)
# =============================================================

# ===================== SIDEBAR =====================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Fraud Prediction", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **ML Project Demo**")
st.sidebar.markdown("💳 Credit Card Fraud Detection")

# ===================== FEATURE NAMES =====================
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

# ===================== PAGE 1 =====================
if page == "Fraud Prediction":

    st.markdown("<h1>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
    st.write("Predict whether a transaction is **fraudulent or legitimate** using Machine Learning.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📌 Transaction Behavior Inputs")
        input_data = []

        with st.expander("Enter Feature Values", expanded=True):
            for feature in feature_names:
                val = st.slider(feature, -10.0, 10.0, 0.0, 0.1)
                input_data.append(val)

        amount = st.number_input(
            "💰 Transaction Amount",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=10.0
        )
        input_data.append(amount)

        st.write("")

        col_a, col_b = st.columns(2)

        # Auto-fill demo buttons
        with col_a:
            if st.button("⚠️ Load Fraud Example"):
                input_data = [-4, -3.5, -5, 3.2] + [0]*24 + [0]

        with col_b:
            if st.button("✅ Load Legit Example"):
                input_data = [0.2, -0.1, 0.3] + [0]*25 + [120]

    with col2:
        st.subheader("📊 Prediction Result")

        if st.button("🔮 Predict Transaction"):
            X = np.array(input_data).reshape(1, -1)
            X_scaled = scaler.transform(X)

            prediction = model.predict(X)[0]
            probability = model.predict_proba(X_scaled)[0][1] * 100

            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")

            st.markdown(
                f"<div class='metric-box'>Fraud Probability<br><b>{probability:.2f}%</b></div>",
                unsafe_allow_html=True
            )

            st.progress(int(probability))

# ===================== PAGE 2 =====================
else:
    st.markdown("<h1>ℹ️ About This Project</h1>", unsafe_allow_html=True)

    st.markdown("""
    **Credit Card Fraud Detection using Machine Learning**

    - This system analyzes transaction behavior patterns to identify fraud.
    - PCA-transformed features are used to protect user privacy.
    - A machine learning classification model predicts fraud in real time.
    - The application is built using **Python, Scikit-learn, and Streamlit**.
    """)

    st.markdown("🎓 *Designed as an academic and real-world demo project.*")

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    "<center style='color:#c77dff;'>© Credit Card Fraud Detection | ML Project</center>",
    unsafe_allow_html=True
)
