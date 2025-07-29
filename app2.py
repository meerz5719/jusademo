import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Load model
model = joblib.load("XGB_model.pkl")

# Title
st.title("📈 Product Demand Forecasting")
st.markdown("Predict future demand for a product based on historical data and ML models.")

# Sidebar Inputs
st.sidebar.header("🧾 Input Features")

store_nbr = st.sidebar.selectbox("Select Store Number", [1, 2, 3, 4, 5, 6, 7, 8, 9])  # Replace with your actual range
date_input = st.sidebar.date_input("Select Date", datetime.today())
promo = st.sidebar.selectbox("Promotion Active?", [0, 1])

# Optional: auto-fill or allow manual input
day_of_week = date_input.weekday()  # Monday=0, Sunday=6
transactions = st.sidebar.number_input("Transactions (optional)", min_value=0, step=10, value=1000)

# Prepare input for model
input_df = pd.DataFrame({
    "store_nbr": [store_nbr],
    "date": [date_input],
    "dayofweek": [day_of_week],
    "promo": [promo],
    "transactions": [transactions],
    # Add any other features your model expects here...
})

# Feature engineering if needed
input_df["year"] = input_df["date"].dt.year
input_df["month"] = input_df["date"].dt.month
input_df["day"] = input_df["date"].dt.day
input_df.drop(columns=["date"], inplace=True)

# Prediction
if st.button("🔮 Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"📦 Predicted Demand: **{round(prediction)} units**")

    # Plot placeholder (optional)
    st.subheader("📊 Predicted Demand Over Time (Sample)")
    sample_dates = pd.date_range(start=date_input, periods=10, freq='D')
    sample_preds = [model.predict(input_df)[0] + np.random.randint(-10, 10) for _ in range(10)]

    fig, ax = plt.subplots()
    ax.plot(sample_dates, sample_preds, marker='o')
    ax.set_title("Predicted Demand (Next 10 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Units")
    st.pyplot(fig)

    # Download
    download_df = pd.DataFrame({"Date": sample_dates, "Predicted Demand": sample_preds})
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

# Optional: Feature Importance (only if model supports)
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Feature Importance")
    feat_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))
