import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Load model
model = joblib.load("XGB_model.pkl")

# Model expected features (from model.get_booster().feature_names)
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

# Sidebar Inputs
st.sidebar.header("🧾 Input Parameters")
store_nbr = st.sidebar.selectbox("Store Number", list(range(1, 55)))
date_input = st.sidebar.date_input("Date", value=datetime.today())
onpromotion = st.sidebar.selectbox("Is On Promotion?", [0, 1])
transactions = st.sidebar.number_input("Transactions", min_value=0, value=1000)
cluster = st.sidebar.selectbox("Cluster", list(range(1, 18)))  # if you know your cluster range
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

# Build base input
input_df = pd.DataFrame({
    "store_nbr": [store_nbr],
    "onpromotion": [onpromotion],
    "transactions": [transactions],
    "cluster": [cluster],
    "date": [pd.to_datetime(date_input)],
    "family": [family],
    "city": [city],
    "holiday_type": [holiday_type]
})

# Extract time features
input_df["year"] = input_df["date"].dt.year
input_df["month"] = input_df["date"].dt.month
input_df["day"] = input_df["date"].dt.day
input_df.drop(columns=["date"], inplace=True)

# One-hot encode categorical features
input_df = pd.get_dummies(input_df, columns=["family", "city", "holiday_type"], prefix=["family", "city", "holiday_type"])

# ✅ Add missing one-hot columns with 0s
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0

# ✅ Ensure correct column order
input_df = input_df[expected_features]

# Final safety check: ensure input has correct features
missing = [col for col in expected_features if col not in input_df.columns]
if missing:
    st.error(f"❌ Missing features: {missing}")
else:
    input_df = input_df[expected_features]  # enforce correct order
    input_df = input_df.astype(np.float32)  # XGBoost prefers float32
    dmatrix = xgb.DMatrix(input_df)
    
    prediction = model.predict(dmatrix)[0]
    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")


# Predict button
import xgboost as xgb



if st.button("🔮 Predict Demand"):
    dmatrix = xgb.DMatrix(input_df)  # Convert your input DataFrame
    prediction = model.predict(dmatrix)[0]
    st.success(f"📈 Predicted Demand: **{round(prediction)} units**")
