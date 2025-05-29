import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the directory this script is running from (i.e., /app)
HERE = os.path.dirname(__file__)

# Build paths to the model files
carbon_model_path = os.path.join(HERE, "layer1_rf_carbon.pkl")
strength_model_path = os.path.join(HERE, "layer2_rf_strength.pkl")

# Load the models
rf_carbon = joblib.load(carbon_model_path)
rf_strength = joblib.load(strength_model_path)

# Predict with uncertainty from Layer 1
def predict_with_uncertainty(rf_model, X_input):
    X_array = X_input.to_numpy() if hasattr(X_input, "to_numpy") else X_input
    all_preds = np.stack([tree.predict(X_array) for tree in rf_model.estimators_], axis=1)
    return np.mean(all_preds, axis=1), np.std(all_preds, axis=1)

# Streamlit interface
st.title("Concrete Strength Predictor")

st.write("Enter concrete mix proportions to predict compressive strength.")

# Example input form — customize based on your dataset
cement = st.number_input("Cement [%]", 7, 20, 12)
water = st.number_input("Water [%]", 5, 15, 7)
gravel = st.number_input("Gravel [%]", 40, 60, 50)
sand = st.number_input("Sand [%]", 20, 40, 30)
preconditioning_time = st.number_input("Preconditioning Time (days)", 5, 150, 20)

# Build input DataFrame
input_df = pd.DataFrame([{
    'cement_per': cement,
    'water_per': water,
    'crushed_gravel_per': gravel,
    'sharp_sand_per': sand,
    'preconditioning_time': preconditioning_time
    # Add more fields as needed based on your model
}])

total_pct = cement + water + gravel + sand
st.write(f"**Total: {total_pct:.1f}%**")

# Conditional button
if abs(total_pct - 100) > 0.01:
    st.error("Total must add up to 100%. Adjust your inputs.")
    st.stop()
else:
    if st.button("Predict Strength"):
        st.success("Inputs valid. Running prediction...")
        carbon_pred, carbon_std = predict_with_uncertainty(rf_carbon, input_df)
        augmented_input = np.hstack([input_df.to_numpy(), carbon_pred[:, None], carbon_std[:, None]])
        strength_pred = rf_strength.predict(augmented_input)

        st.success(f"Predicted compressive strength: **{strength_pred[0]:.2f} MPa**")
        st.caption(f"(Intermediate: carbonation = {carbon_pred[0]:.2f} ± {carbon_std[0]:.2f})")
