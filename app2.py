import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open("XGB_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load original feature columns
with open("xgb_features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Sales Prediction App", layout="wide")

st.title("🛍️ Sales Prediction App")
st.markdown("This app predicts sales patterns of Corporation Favorita over time in different stores in Ecuador based on the inputs.")

# Sidebar or main columns
col1, col2 = st.columns(2)

with col1:
    store_nbr = st.slider("Store Number", 1, 54, 1)
    product_family = st.selectbox("Product Family", [
        'AUTOMOTIVE', 'BEAUTY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
        'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN',
        'HOME APPLIANCES', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER',
        'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
        'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
    ])
    onpromotion = st.number_input("Number of Items on Promotion", 0, 1000, 0)
    state = st.selectbox("State Where The Store Is Located", [
        'Pichincha', 'Guayas', 'Manabi', 'Azuay', 'El Oro', 'Santo Domingo de los Tsachilas',
        'Loja', 'Esmeraldas', 'Tungurahua', 'Imbabura', 'Cotopaxi', 'Chimborazo', 'Pastaza'
    ])
    transactions = st.number_input("Number of Transactions", 0, 10000, 0)

with col2:
    store_type = st.selectbox("Store Type", ['A', 'B', 'C', 'D', 'E'])
    cluster = st.number_input("Cluster", 1, 17, 1)
    oil_price = st.number_input("Crude Oil Price", format="%.4f")
    year = st.number_input("Year", 2012, 2025, 2013)
    month = st.slider("Month", 1, 12, 1)
    day = st.slider("Day", 1, 31, 1)
    day_of_week = st.number_input("Day of Week (0=Sunday to 6=Saturday)", 0, 6, 0)

# Predict button
if st.button("Predict"):
    # Create base input
    input_dict = {
        'store_nbr': store_nbr,
        'onpromotion': onpromotion,
        'cluster': cluster,
        'transactions': transactions,
        'year': year,
        'month': month,
        'day': day
    }

    # One-hot encode categorical vars
    families = [f'family_{col}' for col in feature_columns if col.startswith('family_')]
    for fam in families:
        input_dict[fam] = 1.0 if fam == f'family_{product_family}' else 0.0

    cities = [f'city_{col}' for col in feature_columns if col.startswith('city_')]
    for city in cities:
        input_dict[city] = 1.0 if city == f'city_{state}' else 0.0

    store_types = [f'store_type_{col}' for col in feature_columns if col.startswith('store_type_')]
    for typ in store_types:
        input_dict[typ] = 1.0 if typ == f'store_type_{store_type}' else 0.0

    day_names = [f'dayofweek_{i}' for i in range(7)]
    for d in day_names:
        input_dict[d] = 1.0 if d == f'dayofweek_{day_of_week}' else 0.0

    # Oil price feature
    input_dict['dcoilwtico'] = oil_price

    # Align input to feature columns
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0.0)

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_original_scale = np.expm1(prediction)  # if model was trained on log1p(y)

    st.subheader("📈 Forecasted Demand")
    st.success(f"Predicted Sales (standardized): {prediction:.4f}")
    st.success(f"Predicted Sales (original scale): {prediction_original_scale:.2f} units")
