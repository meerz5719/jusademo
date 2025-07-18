import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_input

# Load model
model = pickle.load(open('model/BEST_STACKED_MODEL_OPTUNA.pkl', 'rb'))

# UI
st.set_page_config(page_title="📈 Product Demand Forecasting", layout="wide")
st.title("🛒 Product Demand Forecasting App")

st.write("Upload your product sales data (CSV with same format as training data):")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Input")
        st.dataframe(input_df.head())

        # Preprocess input
        processed_df = preprocess_input(input_df)

        st.subheader("⚙️ Preprocessed Input")
        st.dataframe(processed_df.head())

        # Predict
        predictions = model.predict(processed_df)
        input_df['Predicted Sales'] = predictions

        st.subheader("🔮 Predictions")
        st.dataframe(input_df[['Predicted Sales']].head())

        # Download results
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, "predicted_output.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
