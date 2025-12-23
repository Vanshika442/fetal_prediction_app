import streamlit as st
import joblib
import numpy as np

# ✅ Load the trained RandomForest model
model = joblib.load("randomforest_model1.joblib")

st.title("🩺 Fetal Health Prediction App")
st.write("This app predicts **fetal health status** (Normal, Suspect, Pathological) using machine learning.")

# ✅ Helper function for safe float conversion
def safe_float_input(label, hint, default="0.0"):
    val = st.text_input(f"{label} {hint}", default)
    try:
        return float(val)
    except ValueError:
        st.warning(f"⚠️ Please enter a numeric value for: {label}. Using default {default}.")
        return float(default)

# ✅ Feature inputs (with guidelines for user)
st.header("Enter Fetal Health Test Values")
baseline_value = safe_float_input("Baseline Value (beats per minute)", "[typical: 110–160]", "120")
accelerations = safe_float_input("Accelerations (per second)", "[e.g., 0.0001 – 0.02]", "0.002")
fetal_movement = safe_float_input("Fetal Movement (per second)", "[e.g., 0 – 5]", "0.5")
uterine_contractions = safe_float_input("Uterine Contractions (per second)", "[e.g., 0 – 1]", "0.003")
light_decelerations = safe_float_input("Light Decelerations (per second)", "[e.g., 0 – 1]", "0.002")
severe_decelerations = safe_float_input("Severe Decelerations (per second)", "[e.g., 0 – 0.1]", "0.0")
prolongued_decelerations = safe_float_input("Prolongued Decelerations (per second)", "[e.g., 0 – 0.5]", "0.0")
mean_value_of_short_term_variability = safe_float_input("Mean Value of Short Term Variability", "[e.g., 0.5 – 7]", "2.0")
mean_value_of_long_term_variability = safe_float_input("Mean Value of Long Term Variability", "[e.g., 5 – 50]", "20")
histogram_mean = safe_float_input("Histogram Mean", "[e.g., 100 – 150]", "120")
histogram_variance = safe_float_input("Histogram Variance", "[e.g., 0 – 100]", "20")

# ✅ Collect all features
features = np.array([[
    baseline_value,
    accelerations,
    fetal_movement,
    uterine_contractions,
    light_decelerations,
    severe_decelerations,
    prolongued_decelerations,
    mean_value_of_short_term_variability,
    mean_value_of_long_term_variability,
    histogram_mean,
    histogram_variance
]])

# ✅ Predict button
if st.button("Predict Fetal Health"):
    prediction = model.predict(features)[0]

    if prediction == 1.0:
        st.success("✅ Predicted Fetal Health: **Normal**")
    elif prediction == 2.0:
        st.warning("⚠️ Predicted Fetal Health: **Suspect**")
    else:
        st.error("🚨 Predicted Fetal Health: **Pathological**")


