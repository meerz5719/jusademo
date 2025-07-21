import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils import preprocess_input

# -- PAGE SETUP --
st.set_page_config(page_title="🧠 Demand Forecasting App", layout="wide")
st.title("📈 Product Demand Forecasting")
st.markdown("Upload a CSV or manually enter values to predict **sales** using different models.")

# -- MODEL SELECTION --
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox("Choose a Model", ["CatBoost", "XGBoost", "Stacked Model"])

model_path = {
    "CatBoost": "model/CAT_model.pkl",
    "XGBoost": "model/XGB_model.pkl",
    "Stacked Model": "model/BEST_STACKED_MODEL_OPTUNA.pkl"
}

try:
    model = pickle.load(open(model_path[model_choice], "rb"))
except FileNotFoundError:
    st.error(f"❌ {model_choice} model file not found at `{model_path[model_choice]}`.")
    st.stop()

# -- TAB UI --
tab1, tab2 = st.tabs(["📁 Upload CSV", "🖊️ Manual Entry"])

# === 📁 FILE UPLOAD MODE ===
with tab1:
    st.markdown("### Upload your product data (CSV):")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.subheader("📋 Raw Input")
            st.dataframe(input_df.head())

            processed_df = preprocess_input(input_df)
            st.subheader("⚙️ Preprocessed Input")
            st.dataframe(processed_df.head())

            predictions = model.predict(processed_df)
            input_df['Predicted Sales'] = predictions

            st.subheader("🔮 Predictions")
            st.dataframe(input_df[['Predicted Sales']].head())

            # Download CSV
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "predicted_output.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

# === 🖊️ MANUAL INPUT MODE ===
with tab2:
    st.markdown("### Manually enter product features:")

    date = st.date_input("Date")
    family = st.selectbox("Product Family", ['BEVERAGES', 'GROCERY I', 'DAIRY', 'SEAFOOD', 'HOME AND KITCHEN I'])
    city = st.selectbox("City", ['QUITO', 'CUENCA', 'GUAYAQUIL', 'AMBATO'])
    holiday_type = st.selectbox("Holiday Type", ['Holiday', 'Additional', 'Event', 'Work Day', 'Transfer'])
    transactions = st.number_input("Number of Transactions", min_value=0, step=1)
    oil_price = st.number_input("Oil Price (dcoilwtico)", min_value=0.0, step=0.1)

    if st.button("🚀 Predict Sales"):
        try:
            input_dict = {
                'id': [1],
                'date': [pd.to_datetime(date)],
                'family': [family],
                'city': [city],
                'holiday_type': [holiday_type],
                'transactions': [transactions],
                'dcoilwtico': [oil_price]
            }
            input_df = pd.DataFrame(input_dict)

            st.subheader("🔍 Input Summary")
            st.dataframe(input_df)

            processed_df = preprocess_input(input_df)
            prediction = model.predict(processed_df)[0]

            st.success(f"📊 Predicted Sales ({model_choice}): **{prediction:.2f}** units")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
