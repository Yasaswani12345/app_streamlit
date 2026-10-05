import streamlit as st
import pickle
import numpy as np
import pandas as pd

dataset = pd.read_csv("BreastCancer.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

with open("breast_cancer_model.pkl", "rb") as file:
    sc, model = pickle.load(file)

st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="wide"
)

st.title("Breast Cancer Detection")

st.write(
    "Select a sample from the dataset and use the trained "
    "Logistic Regression model to predict the diagnosis."
)

sample_number = st.selectbox(
    "Select a sample",
    range(1, len(X) + 1)
)

sample_index = sample_number - 1

sample_features = X[sample_index]
actual_class = y[sample_index]

if st.button("Predict", type="primary"):

    input_data = np.array(sample_features).reshape(1, -1)

    input_scaled = sc.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_class = prediction[0]

    if predicted_class == 0:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")

    if actual_class == 0:
        actual_label = "Benign"
    else:
        actual_label = "Malignant"

    st.write(f"Actual Diagnosis: {actual_label}")

    if predicted_class == actual_class:
        st.success("Prediction matches the actual diagnosis.")
    else:
        st.warning("Prediction does not match the actual diagnosis.")

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Features", "30")

with col2:
    st.metric("Algorithm", "Logistic Regression")

with col3:
    st.metric("Test Accuracy", "97.66%")







