# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import shutil
import joblib

def decompress_file(compressed_path, output_path):
    """ Decompress a .gz file. """
    with gzip.open(compressed_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def load_artifacts():
    # Define file paths
    compressed_model_path = "model.pkl.gz"
    compressed_preprocessor_path = "preprocessor.pkl.gz"
    
    model_path = "model.pkl"
    preprocessor_path = "preprocessor.pkl"

    # Decompress files
    decompress_file(compressed_model_path, model_path)
    decompress_file(compressed_preprocessor_path, preprocessor_path)

    # Load the model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

# Preprocess user input
def preprocess_input(input_data, preprocessor):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data
    input_transformed = preprocessor.transform(input_df)
    return input_transformed

# Make predictions
def predict(model, input_transformed):
    prediction = model.predict(input_transformed)
    # Reverse log transformation
    prediction = np.exp(prediction) - 1
    return prediction[0]

# Streamlit app
def main():
    # Set page title and description
    st.title("🏠 Household Energy Consumption Predictor")
    st.write("This app predicts the energy consumption of household appliances based on input features.")

    # Load model and preprocessor
    model, preprocessor = load_artifacts()

    # Create input fields for user data
    st.sidebar.header("Input Features")

    # Numerical features
    t_out = st.sidebar.number_input("Outdoor Temperature (°C)", value=10.0)
    press_mm_hg = st.sidebar.number_input("Atmospheric Pressure (mmHg)", value=760.0)
    rh_out = st.sidebar.number_input("Outdoor Relative Humidity (%)", value=50.0)
    windspeed = st.sidebar.number_input("Wind Speed (m/s)", value=5.0)
    visibility = st.sidebar.number_input("Visibility (km)", value=10.0)
    tdewpoint = st.sidebar.number_input("Dew Point Temperature (°C)", value=5.0)
    hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
    is_weekend = st.sidebar.selectbox("Is it a weekend?", ["No", "Yes"])
    avg_temp = st.sidebar.number_input("Average Indoor Temperature (°C)", value=20.0)
    avg_humidity = st.sidebar.number_input("Average Indoor Humidity (%)", value=50.0)
    

    # Categorical features
    day_of_week = st.sidebar.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

    # Add missing columns
    month = st.sidebar.slider("Month", 1, 12, 6)  # Default to June
    lights = st.sidebar.number_input("Lights Consumption (watts)", value=0.0)

    # Input data
    input_data = {
        'lights': lights,
        't_out': t_out,
        'press_mm_hg': press_mm_hg,
        'rh_out': rh_out,
        'windspeed': windspeed,
        'visibility': visibility,
        'tdewpoint': tdewpoint,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,  # Newly added feature
        'is_weekend': 1 if is_weekend == "Yes" else 0,
        'season': season,
        'avg_temp': avg_temp,
        'avg_humidity': avg_humidity
    }


    # Preprocess input data
    input_transformed = preprocess_input(input_data, preprocessor)

    # Make prediction
    if st.sidebar.button("Predict"):
        prediction = predict(model, input_transformed)
        st.success(f"Predicted Energy Consumption: **{prediction:.2f} watts**")

# Run the app
if __name__ == '__main__':
    main()

