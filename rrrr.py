import streamlit as st
import pandas as pd
import joblib


model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Taxi Trip Price Predictor", layout="wide")

st.title("🚕 Taxi Trip Price Prediction")
st.write("Random Forest Regression Model")

st.sidebar.header("Enter Trip Details")
user_input = {}

for col in model_columns:
    user_input[col] = st.sidebar.number_input(col, value=0.0)
input_df = pd.DataFrame([user_input])
input_df = input_df[model_columns]
prediction = model.predict(input_df)[0]

price_map = {
    "Low": 150,
    "Medium": 300,
    "High": 600
}

price = price_map[prediction]

st.success(f"💰 Estimated Trip Price: ₹ {price}")

if st.checkbox("Show Input Data"):
    st.write(input_df)
