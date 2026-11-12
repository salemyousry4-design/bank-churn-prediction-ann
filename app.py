"""Streamlit app: predict whether a bank customer is likely to leave.

Run with:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "churn_model.keras")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    return model, preprocessor


st.title("🏦 Bank Customer Churn Prediction")
st.caption(
    "An artificial neural network trained on 10,000 bank customers estimates "
    "how likely a customer is to close their account."
)

if not (MODEL_DIR / "churn_model.keras").exists():
    st.error("Model not found. Train it first with `python train.py`.")
    st.stop()

model, preprocessor = load_artifacts()

with st.form("customer"):
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.slider("Tenure (years with the bank)", 0, 10, 3)
        credit_score = st.number_input("Credit score", 300, 850, 600)
    with col2:
        balance = st.number_input("Balance ($)", 0.0, 300_000.0, 60_000.0, step=1_000.0)
        salary = st.number_input("Estimated salary ($)", 0.0, 300_000.0, 50_000.0, step=1_000.0)
        products = st.slider("Number of products", 1, 4, 2)
        has_card = st.checkbox("Has a credit card", value=True)
        active = st.checkbox("Active member", value=True)
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    customer = pd.DataFrame(
        [
            {
                "Geography": geography,
                "Gender": gender,
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": products,
                "HasCrCard": int(has_card),
                "IsActiveMember": int(active),
                "EstimatedSalary": salary,
            }
        ]
    )
    probability = float(model.predict(preprocessor.transform(customer), verbose=0)[0][0])

    st.metric("Churn probability", f"{probability:.0%}")
    st.progress(probability)
    if probability >= 0.5:
        st.error("⚠️ High risk: this customer is likely to leave the bank.")
    else:
        st.success("✅ Low risk: this customer is likely to stay.")
