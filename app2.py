import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load trained XGBRegressor model
model = joblib.load("XGB_model.pkl")

# Features model was trained on (use model.get_booster().feature_names if needed)
expected_features = [
    'store_nbr', 'onpromotion', 'cluster', 'transactions', 'year', 'month', 'day',
    'family_AUTOMOTIVE', 'family_BEAUTY', 'family_CELEBRATION', 'family_CLEANING', 'family_CLOTHING',
    'family_FOODS', 'family_GROCERY', 'family_HARDWARE', 'family_HOME', 'family_LADIESWEAR',
    'family_LAWN AND GARDEN', 'family_LIQUOR,WINE,BEER', 'family_PET SUPPLIES', 'family_STATIONERY',
    'city_Ambato', 'city_Babahoyo', 'city_Cayambe', 'city_Cuenca', 'city_Daule', 'city_El Carmen',
    'city_Esmeraldas', 'city_Guaranda', 'city_Guayaquil', 'city_Ibarra', 'city_Latacunga', 'city_Libertad',
    'city_Loja', 'city_Machala', 'city_Manta', 'city_Playas', 'city_Puyo', 'city_Quevedo', 'city_Quito',
    'city_Riobamba', 'city_Salinas', 'city_Santo Domingo',
    'holiday_type_Additional', 'holiday_type_Bridge', 'holiday_type_Event',
    'holiday_type_Holiday', 'holiday_type_Transfer', 'holiday_type_Work Day'
]

# Set page title
st.set_page_config(page_title="Product Demand Forecasting", layout="centered")
st.title("📦 Product Demand Forecasting App")

st.sidebar.header("🧾 Input Parameters")

# Sidebar inputs
store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))
date_input = st.sidebar.date_input("Select Date", value=datetime.today())
onpromotion = st.sidebar.selectbox("Is Promotion Active?", [0, 1])
transactions = st.sidebar.number_input("Transactions", min_value=0, value=1000)
cluster = st.sidebar.selectbox("Cluster", list(range(1, 18)))
family = st.sidebar.selectbox("Product Family", [
    "AUTOMOTIVE", "BEAUTY", "CELEBRATION", "CLEANING", "CLOTHING", "FOODS",
    "GROCERY", "HARDWARE", "HOME", "LADIESWEAR", "LAWN AND GARDEN", "LIQUOR,WINE,BEER",
    "PET SUPPLIES", "STATIONERY"
])
city = st.sidebar.selectbox("City", [
    "Ambato", "Babahoyo", "Cayambe", "Cuenca", "Daule", "El Carmen", "Esmeraldas", "Guaranda", "Guayaquil",
    "Ibarra", "Latacunga", "Libertad", "Loja", "Machala", "Manta", "Playas", "Puyo", "Quevedo", "Quito",
    "Riobamba", "Salinas", "Santo Domingo"
])
holiday_type = st.sidebar.selectbox("Holiday Type", [
    "Additional", "Bridge", "Event", "Holiday", "Transfer", "Work Day"
])

# Build input dataframe
def create_input_df(selected_date):
    df = pd.DataFrame({
        "store_nbr": [store_nbr],
        "onpromotion": [onpromotion],
        "transactions": [transactions],
        "cluster": [cluster],
        "date": [pd.to_datetime(selected_date)],
        "family": [family],
        "city": [city],
        "holiday_type": [holiday_type]
    })
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df.drop(columns=["date"], inplace=True)

    # One-hot encode
    df = pd.get_dummies(df, columns=["family", "city", "holiday_type"], prefix=["family", "city", "holiday_type"])

    # Add missing columns (set to 0)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    df = df[expected_features]
    df = df.astype(np.float32)

    return df

# Predict for selected date
if st.button("🔮 Predict Demand for Selected Date"):
    input_df = create_input_df(date_input)
    prediction = model.predict(input_df.values)[0]
    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")

    # Forecast next 10 days
    st.subheader("📊 10-Day Demand Forecast")
    future_dates = pd.date_range(start=date_input, periods=10)
    forecast_data = []

    for dt in future_dates:
        row = create_input_df(dt)
        pred = model.predict(row.values)[0]
        forecast_data.append((dt, round(pred)))

    forecast_df = pd.DataFrame(forecast_data, columns=["Date", "Forecasted Demand"])
    st.line_chart(forecast_df.set_index("Date"))

    # Download button
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")
