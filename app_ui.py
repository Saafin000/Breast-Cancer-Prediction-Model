import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Prediction")
st.write("Fill in the tumor features to check if it is **Benign** or **Malignant**.")

col1, col2 = st.columns(2)

with col1:
    radius_worst = st.number_input("Radius Worst", value=16.0)
    perimeter_worst = st.number_input("Perimeter Worst", value=100.0)
    area_worst = st.number_input("Area Worst", value=800.0)
    concave_points_worst = st.number_input("Concave Points Worst", value=0.2)
    concavity_worst = st.number_input("Concavity Worst", value=0.3)
    compactness_worst = st.number_input("Compactness Worst", value=0.2)
    radius_mean = st.number_input("Radius Mean", value=14.0)

with col2:
    perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
    area_mean = st.number_input("Area Mean", value=700.0)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.1)
    concavity_mean = st.number_input("Concavity Mean", value=0.2)
    compactness_mean = st.number_input("Compactness Mean", value=0.1)
    texture_worst = st.number_input("Texture Worst", value=20.0)
    smoothness_worst = st.number_input("Smoothness Worst", value=0.1)
    symmetry_worst = st.number_input("Symmetry Worst", value=0.3)

if st.button("🔍 Predict"):
    data = {
        "radius_worst": radius_worst,
        "perimeter_worst": perimeter_worst,
        "area_worst": area_worst,
        "concave_points_worst": concave_points_worst,
        "concavity_worst": concavity_worst,
        "compactness_worst": compactness_worst,
        "radius_mean": radius_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
        "concave_points_mean": concave_points_mean,
        "concavity_mean": concavity_mean,
        "compactness_mean": compactness_mean,
        "texture_worst": texture_worst,
        "smoothness_worst": smoothness_worst,
        "symmetry_worst": symmetry_worst
    }

    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        result = response.json()["prediction"]
        st.success(f"✅ Prediction: {result}")
    else:
        st.error("⚠️ API Error")
