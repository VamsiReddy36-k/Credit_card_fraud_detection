import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("Model not found")
    st.stop()

st.set_page_config(page_title="Fraud Dashboard", layout="wide", page_icon="💳")

# ---------- SIDEBAR NAV ----------
st.sidebar.title("🏦 Bank Dashboard")
page = st.sidebar.radio("Navigate", ["🔍 Prediction", "📊 Analytics", "ℹ️ About"])

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>💳 Fraud Detection Dashboard</h1>
<p style='text-align:center;color:gray;'>Real-time transaction monitoring system</p>
""", unsafe_allow_html=True)

# ===============================
# 🔍 PAGE 1: PREDICTION
# ===============================
if page == "🔍 Prediction":

    st.markdown("### 🧾 Transaction Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input("⏱ Time", min_value=0.0)
    with col2:
        amount = st.number_input("💰 Amount", min_value=0.0)
    with col3:
        location = st.selectbox("🌍 Location", ["Online", "ATM", "POS"])

    st.markdown("### 🔢 PCA Features")

    cols = st.columns(4)
    v_features = []
    for i in range(28):
        with cols[i % 4]:
            v_features.append(st.number_input(f"V{i+1}", value=0.0))

    if st.button("🚀 Analyze Transaction", use_container_width=True):

        input_data = np.array([[time] + v_features + [amount]])
        prediction = model.predict(input_data)[0]

        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

        st.markdown("---")

        # RESULT METRICS
        col1, col2, col3 = st.columns(3)

        if prediction == 1:
            col1.metric("Status", "Fraud 🚨")
        else:
            col1.metric("Status", "Genuine ✅")

        col2.metric("Amount", f"${amount}")

        if prob:
            col3.metric("Fraud Risk", f"{prob:.2%}")

        # ALERT BOX
        if prediction == 1:
            st.error("🚨 High Risk Transaction Detected!")
        else:
            st.success("✅ Transaction is Safe")

        if prob:
            st.progress(prob)

# ===============================
# 📊 PAGE 2: ANALYTICS
# ===============================
elif page == "📊 Analytics":

    st.markdown("### 📊 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492")
    col3.metric("Accuracy", "99.2%")

    st.markdown("---")

    st.markdown("### 📈 Transaction Insights")

    # Dummy chart
    data = pd.DataFrame({
        "Type": ["Genuine", "Fraud"],
        "Count": [284315, 492]
    })

    st.bar_chart(data.set_index("Type"))

# ===============================
# ℹ️ PAGE 3: ABOUT
# ===============================
else:

    st.markdown("### ℹ️ About Project")

    st.info("""
    This system uses Machine Learning to detect fraudulent transactions.

    Models used:
    - Logistic Regression
    - Random Forest
    - XGBoost

    Features:
    - Real-time prediction
    - Fraud probability scoring
    - Dashboard analytics
    """)