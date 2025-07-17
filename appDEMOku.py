import streamlit as st
import pickle

# Load model
model = pickle.load(open('CAT_model.pkl', 'rb'))

# UI
st.title("Your Project Title")
user_input = st.text_input("Enter your input:")

# Prediction
if st.button("Predict"):
    result = model.predict([[user_input]])  # adapt to your input type
    st.success(f"Prediction: {result}")
