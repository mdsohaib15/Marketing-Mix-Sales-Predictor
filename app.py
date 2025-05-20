import streamlit as st
import numpy as np
import pickle

# Load the trained model (must match training script)
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# App title
st.title("📊 Marketing Mix Sales Predictor")
st.markdown("Predict product sales based on advertising budgets for **TV**, **Radio**, and **Newspaper**.")

# Input sliders
tv = st.slider("TV Advertising Budget (in $ thousands)", min_value=0, max_value=300, value=100)
radio = st.slider("Radio Advertising Budget (in $ thousands)", min_value=0, max_value=50, value=25)
newspaper = st.slider("Newspaper Advertising Budget (in $ thousands)", min_value=0, max_value=120, value=20)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)
    st.success(f"📈 Predicted Sales: **{prediction[0]:.2f}** units")

# Optional: About
st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses a Linear Regression model trained on a marketing dataset to predict product sales.")
