import streamlit as st
import joblib
import numpy as np

# ✅ Load the trained RandomForest model
model = joblib.load("randomforest_model1.joblib")

st.title("🩺 Fetal Health Prediction App")
st.write("This app predicts **fetal health status** (Normal, Suspect, Pathological) using machine learning.")

# Define feature inputs (11 features you selected)
st.header("Enter Fetal Health Test Values")

baseline_value = st.number_input("Baseline Value (beats per minute)", min_value=50.0, max_value=200.0, step=0.1)
accelerations = st.number_input("Accelerations (per second)", min_value=0.0, max_value=1.0, step=0.01)
fetal_movement = st.number_input("Fetal Movement (per second)", min_value=0.0, max_value=10.0, step=0.01)
uterine_contractions = st.number_input("Uterine Contractions (per second)", min_value=0.0, max_value=1.0, step=0.01)
light_decelerations = st.number_input("Light Decelerations (per second)", min_value=0.0, max_value=1.0, step=0.01)
severe_decelerations = st.number_input("Severe Decelerations (per second)", min_value=0.0, max_value=1.0, step=0.01)
prolongued_decelerations = st.number_input("Prolongued Decelerations (per second)", min_value=0.0, max_value=1.0, step=0.01)
mean_short_var = st.number_input("Mean Value of Short Term Variability", min_value=0.0, max_value=10.0, step=0.1)
mean_long_var = st.number_input("Mean Value of Long Term Variability", min_value=0.0, max_value=50.0, step=0.1)
histogram_mean = st.number_input("Histogram Mean", min_value=50.0, max_value=200.0, step=0.1)
histogram_variance = st.number_input("Histogram Variance", min_value=0.0, max_value=100.0, step=0.1)

# Collect all features in the same order as training
features = np.array([[
    baseline_value,
    accelerations,
    fetal_movement,
    uterine_contractions,
    light_decelerations,
    severe_decelerations,
    prolongued_decelerations,
    mean_short_var,
    mean_long_var,
    histogram_mean,
    histogram_variance
]])

# Predict button
if st.button("Predict Fetal Health"):
    prediction = model.predict(features)[0]

    if prediction == 1.0:
        st.success("✅ Predicted Fetal Health: **Normal**")
    elif prediction == 2.0:
        st.warning("⚠️ Predicted Fetal Health: **Suspect**")
    else:
        st.error("🚨 Predicted Fetal Health: **Pathological**")
