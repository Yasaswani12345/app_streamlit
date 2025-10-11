import streamlit as st

st.write("Hello, Streamlit!")

import pickle
import numpy as np

# Load model
with open("breast_cancer_model.pkl", "rb") as file:
 
    model = pickle.load(file)

st.title("Breast Cancer Detection")

# User inputs
features = [st.number_input(f"Feature {i}") for i in range(1, 31)]


if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")
