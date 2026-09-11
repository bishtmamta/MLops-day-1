import streamlit as st
import pandas as pd
import joblib
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Load trained model
model = load_model()

# Streamlit app
st.title("Advertising Sales Predictor")

# Input fields
tv = st.number_input("TV Budget")
radio = st.number_input("Radio Budget")
newspaper = st.number_input("Newspaper Budget")

# Prediction button
if st.button("Predict"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "radio": [radio],
        "newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Prediction Sales: {prediction[0]:.2f}"
    )
    



