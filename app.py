import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open('linear_regression_model.pkl', 'rb'))

st.title("Scikit learn Linear Regression Model")
tv = st.text_input("Enter TV Sales...")
radio = st.text_input("Enter Radio Sales...")
newspaper = st.text_input("Enter Newspaper Sales...")

if st.button("predict"):
    features = np.array([[tv,radio,newspaper]], dtype=np.float64)
    results = model.predict(features).reshape(1,-1)
    st.write("Predicted Sales" ,results[0])