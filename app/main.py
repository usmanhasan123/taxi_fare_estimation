# from fastapi import FastAPI
# from app.inference import predict

# print("Loaded main.py")
# print(__file__)

# app = FastAPI()

# @app.get("/")
# def health_check():
#     return {"status": "ok"}


# @app.post("/predict")
# def prediction(request: int):
#     prediction = predict(request)
#     return {"prediction": prediction}

import streamlit as st
from inference import predict

st.title("Model Prediction App")

# Health/status message
st.success("App is running")

# User input
request = st.number_input(
    "Enter input value",
    min_value=0,
    step=1
)

# Prediction
if st.button("Predict"):
    prediction = predict(int(request))

    st.write("Prediction:", prediction)