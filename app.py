import streamlit as st
import pickle
import numpy as np

# load model and scaler
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Segmentation")

st.write("Enter customer details")

income = st.number_input("Annual Income")
purchase = st.number_input("Purchase Amount")
frequency = st.number_input("Purchase Frequency")
loyalty = st.number_input("Loyalty Score")

if st.button("Predict Segment"):
    data = np.array([[income, purchase, frequency, loyalty]])
    data_scaled = scaler.transform(data)

    cluster = kmeans.predict(data_scaled)[0]

    st.write(f"Customer belongs to Cluster: {cluster}")