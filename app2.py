# Load column order from training (you should have saved it earlier)
with open("X_columns.pkl", "rb") as f:
    column_order = pickle.load(f)

# Create a dictionary with zero for all one-hot columns
input_dict = dict.fromkeys(column_order, 0)

# Fill actual inputs
input_dict.update({
    'store_nbr': store_number,
    'onpromotion': on_promo,
    'cluster': cluster,
    'transactions': transactions,
    'year': year,
    'month': month,
    'day': day,
    'crude_oil_price': crude_oil_price,
    f'family_{product_family}': 1,
    f'city_{state}': 1,
    'holiday_type_Work Day': 1  # or set based on calendar input
})

# Convert to DataFrame with correct order
input_df = pd.DataFrame([input_dict])[column_order]

# Predict
if st.button("🔮 Predict Demand"):
    prediction = model.predict(input_df)[0]
    prediction = np.maximum(prediction, 0)
    st.success(f"📦 Predicted Sales for {product_family}: *{prediction:.2f} units*")
