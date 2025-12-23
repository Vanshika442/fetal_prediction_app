import streamlit as st
import joblib
import numpy as np
import pandas as pd   # ✅ ADD THIS

# Load model
model = joblib.load("randomforest_model1.joblib")

st.title("🩺 Fetal Health Prediction App")
st.write("This app predicts **fetal health status** (Normal, Suspect, Pathological).")

def safe_float_input(label, hint, default="0.0"):
    val = st.text_input(f"{label} {hint}", default)
    try:
        return float(val)
    except ValueError:
        st.warning(f"⚠️ Please enter a numeric value for: {label}. Using default {default}.")
        return float(default)

st.header("Enter Fetal Health Test Values")

baseline_value = safe_float_input("Baseline Value", "[110–160]", "120")
accelerations = safe_float_input("Accelerations", "[0.0001 – 0.02]", "0.002")
fetal_movement = safe_float_input("Fetal Movement", "[0 – 5]", "0.5")
uterine_contractions = safe_float_input("Uterine Contractions", "[0 – 1]", "0.003")
light_decelerations = safe_float_input("Light Decelerations", "[0 – 1]", "0.002")
severe_decelerations = safe_float_input("Severe Decelerations", "[0 – 0.1]", "0.0")
prolongued_decelerations = safe_float_input("Prolongued Decelerations", "[0 – 0.5]", "0.0")
mean_value_of_short_term_variability = safe_float_input("Short Term Variability", "[0.5 – 7]", "2.0")
mean_value_of_long_term_variability = safe_float_input("Long Term Variability", "[5 – 50]", "20")
histogram_mean = safe_float_input("Histogram Mean", "[100 – 150]", "120")
histogram_variance = safe_float_input("Histogram Variance", "[0 – 100]", "20")

# ✅ CREATE DATAFRAME WITH COLUMN NAMES (MOST IMPORTANT PART)
input_df = pd.DataFrame([{
    "baseline value": baseline_value,
    "accelerations": accelerations,
    "fetal_movement": fetal_movement,
    "uterine_contractions": uterine_contractions,
    "light_decelerations": light_decelerations,
    "severe_decelerations": severe_decelerations,
    "prolonged_decelerations": prolongued_decelerations,
    "mean_value_of_short_term_variability": mean_value_of_short_term_variability,
    "mean_value_of_long_term_variability": mean_value_of_long_term_variability,
    "histogram_mean": histogram_mean,
    "histogram_variance": histogram_variance
}])

# Predict
if st.button("Predict Fetal Health"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Predicted Fetal Health: **Normal**")
    elif prediction == 2:
        st.warning("⚠️ Predicted Fetal Health: **Suspect**")
    else:
        st.error("🚨 Predicted Fetal Health: **Pathological**")
