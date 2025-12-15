import streamlit as st
import joblib
import numpy as np
import os
import sys

# --- Configuration ---
MODEL_FILE = 'trained_house_pricing_LR_model.pkl' 
INPUT_FEATURES = ['Bedrooms', 'Bathrooms']

# --- Model Loading ---
model = None
try:
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file not found: {MODEL_FILE}. Please ensure it is in the same directory.")
        st.stop()
        
    with open(MODEL_FILE, "rb") as f:
        model = joblib.load(f)

    # Simple check to confirm the model expects 2 features
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != 2:
        st.warning(f"Model loaded expects {model.n_features_in_} features, but the app is configured for 2 (Bedrooms/Bathrooms).")

    st.sidebar.success("Model loaded successfully.")
    
except Exception as e:
    st.error(f"CRITICAL ERROR: Failed to load model. Details: {e}")
    st.error("Please check the terminal for scikit-learn version mismatch information.")
    sys.exit(1)


# --- Streamlit Application Layout ---
st.title("🏡 House Price Prediction Dashboard")
st.markdown("Use the sliders below to enter the house specifications and predict the final sale price.")

# --- Input Fields using Streamlit Sliders ---
# Use sliders to make interaction easy
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.slider(
        f"Number of {INPUT_FEATURES[0]}:", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1
    )

with col2:
    # Use a number input or slider for bathrooms, allowing half baths (step=0.5)
    bathrooms = st.number_input(
        f"Number of {INPUT_FEATURES[1]}:", 
        min_value=1.0, 
        max_value=5.0, 
        value=2.0, 
        step=0.5
    )

# --- Prediction Logic ---
if st.button("Predict House Price"):
    # 1. Create the input array
    input_data = np.array([[bedrooms, bathrooms]])
    
    try:
        # 2. Make prediction
        prediction = model.predict(input_data)[0]
        
        # 3. Format output
        formatted_price = f"${prediction:,.2f}"
        
        st.success(f"## {formatted_price}")
        st.balloons() # Optional: celebration effect

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.code(f"Input: {input_data}")

# Optional: Add a section to show the inputs for transparency
st.sidebar.subheader("Current Inputs")
st.sidebar.write(f"Bedrooms: {bedrooms}")
st.sidebar.write(f"Bathrooms: {bathrooms}")
