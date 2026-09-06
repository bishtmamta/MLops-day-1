
import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

# MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load model
model = mlflow.sklearn.load_model(
    "models:/Sales_Prediction_Model@champion"
)

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




