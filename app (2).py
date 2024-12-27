import pandas as pd
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the model
model = joblib.load('model_gbm.pkl')

# Constants for input parameters
MATERIAL_PROPERTIES = {
    'cement': {'density': 3.1, 'price': 1800, 'emission': 0.931},
    'slag': {'density': 2.7, 'price': 800, 'emission': 0.0196},
    'ash': {'density': 2.29, 'price': 600, 'emission': 0.0265},
    'water': {'density': 1, 'price': 20, 'emission': 0.00196},
    'superplastic': {'density': 1.07, 'price': 25000, 'emission': 0.25},
    'coarseagg': {'density': 2.74, 'price': 241, 'emission': 0.0075},
    'fineagg': {'density': 2.68, 'price': 351, 'emission': 0.0026},
}

# Functions
def du_doan_cuong_do(materials, age):
    input_data = [
        materials['cement'],
        materials['slag'],
        materials['ash'],
        materials['water'],
        materials['superplastic'],
        materials['coarseagg'],
        materials['fineagg'],
        age
    ]
    input_array = np.array(input_data).reshape(1, -1)
    return model.predict(input_array)[0]

def tinh_toan_kinh_te_va_phat_thai(materials, prediction):
    total_cost = sum(materials[mat] * MATERIAL_PROPERTIES[mat]['price'] for mat in materials)
    total_emission = sum(materials[mat] * MATERIAL_PROPERTIES[mat]['emission'] for mat in materials)
    cost_per_mpa = total_cost / prediction
    emission_per_mpa = total_emission / prediction
    return total_cost, total_emission, cost_per_mpa, emission_per_mpa

# Streamlit UI
st.title("Dự đoán cường độ bê tông và tính toán kinh tế, phát thải")

# Input material quantities
materials = {}
for mat in MATERIAL_PROPERTIES:
    materials[mat] = st.number_input(f"{mat.capitalize()} (kg):", min_value=0.0, value=0.0, step=1.0)

# Input prediction age
age = st.number_input("Tuổi bê tông (ngày):", min_value=1, value=28, step=1)

if st.button("Dự đoán và tính toán"):
    # Make prediction
    try:
        prediction = du_doan_cuong_do(materials, age)

        # Calculate economic and environmental metrics
        total_cost, total_emission, cost_per_mpa, emission_per_mpa = tinh_toan_kinh_te_va_phat_thai(materials, prediction)

        # Display results
        st.subheader("Kết quả dự đoán và tính toán")
        st.write(f"**Cường độ nén dự đoán:** {prediction:.2f} MPa")
        st.write(f"**Tổng giá thành:** {total_cost:,.2f} VNĐ")
        st.write(f"**Tổng phát thải:** {total_emission:,.2f} kg CO2")
        st.write(f"**Giá thành/MPa:** {cost_per_mpa:,.2f} VNĐ/MPa")
        st.write(f"**CO2/MPa:** {emission_per_mpa:,.2f} kg CO2/MPa")
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")
