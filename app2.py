import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Load the XGBoost model
model = joblib.load("XGB_model.pkl")

# Title
st.title("📦 Product Demand Forecasting App")
st.markdown("Predict future product demand using an XGBoost model.")

# Sidebar inputs
st.sidebar.header("🧾 Enter Input Features")

store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))  # update range if needed
date_input = st.sidebar.date_input("Select Date", value=datetime.today())
promo = st.sidebar.selectbox("Is Promotion Active?", [0, 1])
transactions = st.sidebar.number_input("Transactions", min_value=0, value=1000, step=50)

# Prepare input
input_df = pd.DataFrame({
    "store_nbr": [store_nbr],
    "date": [date_input],
    "promo": [promo],
    "transactions": [transactions],
})

# Convert date and engineer features
input_df["date"] = pd.to_datetime(input_df["date"])
input_df["year"] = input_df["date"].dt.year
input_df["month"] = input_df["date"].dt.month
input_df["day"] = input_df["date"].dt.day
input_df["dayofweek"] = input_df["date"].dt.dayofweek
input_df.drop(columns=["date"], inplace=True)

# ✅ Ensure input_df has same features as model
expected_features = ['store_nbr', 'onpromotion', 'cluster', 'transactions', 'year', 'month', 'day', 'family_AUTOMOTIVE', 'family_BEAUTY', 'family_CELEBRATION', 'family_CLEANING', 'family_CLOTHING', 'family_FOODS', 'family_GROCERY', 'family_HARDWARE', 'family_HOME', 'family_LADIESWEAR', 'family_LAWN AND GARDEN', 'family_LIQUOR,WINE,BEER', 'family_PET SUPPLIES', 'family_STATIONERY', 'city_Ambato', 'city_Babahoyo', 'city_Cayambe', 'city_Cuenca', 'city_Daule', 'city_El Carmen', 'city_Esmeraldas', 'city_Guaranda', 'city_Guayaquil', 'city_Ibarra', 'city_Latacunga', 'city_Libertad', 'city_Loja', 'city_Machala', 'city_Manta', 'city_Playas', 'city_Puyo', 'city_Quevedo', 'city_Quito', 'city_Riobamba', 'city_Salinas', 'city_Santo Domingo', 'holiday_type_Additional', 'holiday_type_Bridge', 'holiday_type_Event', 'holiday_type_Holiday', 'holiday_type_Transfer', 'holiday_type_Work Day']
input_df = input_df[expected_features]

# Prediction button
if st.button("🔮 Predict Demand"):
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")

    # Generate future forecast (optional)
    st.subheader("📊 Forecast for Next 10 Days")
    future_dates = pd.date_range(start=date_input, periods=10)
    future_forecasts = []

    for dt in future_dates:
        row = pd.DataFrame({
            "store_nbr": [store_nbr],
            "promo": [promo],
            "transactions": [transactions],
            "year": [dt.year],
            "month": [dt.month],
            "day": [dt.day],
            "dayofweek": [dt.weekday()],
        })
        pred = model.predict(row)[0]
        future_forecasts.append(pred)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Demand": future_forecasts
    })

    # Plot
    st.line_chart(forecast_df.set_index("Date"))

    # Download
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
