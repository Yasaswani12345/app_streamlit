import streamlit as st

st.write("Hello, Streamlit!")

import pickle
import numpy as np

# Load model
with open("breast_cancer_model.pkl", "rb") as file:
 
    model = pickle.load(file)

st.title("Breast Cancer Detection")

# User inputs
features = []
features.append(st.number_input("Feature 1"))
features.append(st.number_input("Feature 2"))
features.append(st.number_input("Feature 3"))
features.append(st.number_input("Feature 4"))
features.append(st.number_input("Feature 5"))
features.append(st.number_input("Feature 6"))
features.append(st.number_input("Feature 7"))
features.append(st.number_input("Feature 8"))
features.append(st.number_input("Feature 9"))
features.append(st.number_input("Feature 10"))

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")
