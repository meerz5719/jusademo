import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained XGBoost model
with open('XGB_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load column names used in training
with open('xgb_features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("🧮 Product Demand Forecast App (XGBoost)")

st.markdown("Enter the details below to forecast product demand:")

# Create input form
input_data = {}
for feature in feature_names:
    if 'family_' in feature or 'city_' in feature or 'holiday_type_' in feature:
        input_data[feature] = st.selectbox(f"{feature}", [0.0, 1.0])
    elif feature in ['store_nbr', 'cluster', 'month', 'day', 'year']:
        input_data[feature] = st.number_input(f"{feature}", value=1, step=1)
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.6f")

# Submit button
if st.button("Predict Demand"):
    # Construct input DataFrame
    user_input_df = pd.DataFrame([input_data])
    
    # Ensure correct order of columns
    user_input_df = user_input_df[feature_names]

    # Predict
    prediction = model.predict(user_input_df)[0]

    st.success(f"📦 Predicted Sales (standardized): {prediction:.4f}")

    # If you had used inverse scaling or log transform during training:
    # e.g., if you applied log1p transform during training:
    # final_prediction = np.expm1(prediction)
    # st.success(f"📦 Predicted Sales (original scale): {final_prediction:.2f} units")
