import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import shutil

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Function to decompress files
def decompress_file(compressed_path, output_path):
    """ Decompress a .gz file. """
    with gzip.open(compressed_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# Function to load artifacts
def load_artifacts():
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

# Function to preprocess user input
def preprocess_input(input_data, preprocessor):
    input_df = pd.DataFrame([input_data])
    input_transformed = preprocessor.transform(input_df)
    return input_transformed

# Function to make predictions
def predict(model, input_transformed):
    prediction = model.predict(input_transformed)
    prediction = np.exp(prediction) - 1  # Reverse log transformation
    return prediction[0]

# Streamlit app
def main():
    # Add an image banner
    st.image("energy_banner.jpg", use_container_width=True)

    # Tabs for navigation
    tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ About"])

    with tab1:
        st.title("⚡ Household Energy Consumption Predictor")
        st.markdown(
            """
            **Predict the energy consumption of household appliances**  
            based on environmental and indoor conditions.  
            """
        )

        # Load model and preprocessor
        model, preprocessor = load_artifacts()

        # Sidebar with input fields
        st.sidebar.header("🔧 Input Features")
        st.sidebar.divider()

        # Organize inputs in two columns
        col1, col2 = st.columns(2)

        with col1:
            t_out = st.number_input("🌡️ Outdoor Temperature (°C)", value=10.0)
            press_mm_hg = st.number_input("🌬️ Atmospheric Pressure (mmHg)", value=760.0)
            rh_out = st.number_input("💦 Outdoor Humidity (%)", value=50.0)
            windspeed = st.number_input("🌪️ Wind Speed (m/s)", value=5.0)
            visibility = st.number_input("👀 Visibility (km)", value=10.0)

        with col2:
            tdewpoint = st.number_input("❄️ Dew Point Temperature (°C)", value=5.0)
            avg_temp = st.number_input("🏠 Indoor Temperature (°C)", value=20.0)
            avg_humidity = st.number_input("🏡 Indoor Humidity (%)", value=50.0)
            hour = st.slider("⏰ Hour of the Day", 0, 23, 12)
            is_weekend = st.selectbox("📅 Is it a weekend?", ["No", "Yes"])

        # Categorical features
        day_of_week = st.selectbox("📆 Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        season = st.selectbox("🌍 Season", ["Winter", "Spring", "Summer", "Fall"])
        month = st.slider("🗓️ Month", 1, 12, 6)  
        lights = st.number_input("💡 Lights Consumption (watts)", value=0.0)

        # Prepare input data
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
            'month': month,
            'is_weekend': 1 if is_weekend == "Yes" else 0,
            'season': season,
            'avg_temp': avg_temp,
            'avg_humidity': avg_humidity
        }

        # Preprocess input data
        input_transformed = preprocess_input(input_data, preprocessor)

        # Prediction button
        if st.button("🚀 Predict Energy Consumption"):
            prediction = predict(model, input_transformed)
            st.success(f"⚡ Predicted Energy Consumption: **{prediction:.2f} watts**")

        # Reset button
        if st.button("🔄 Reset Inputs", key="reset_button"):
            st.session_state.clear()
            st.experimental_rerun()

    with tab2:
        st.write("### ℹ️ About This App")
        st.write("This app predicts household energy consumption using machine learning.")

        with st.expander("💡 How does this work?"):
            st.write(
                """
                - Enter environmental and household conditions.
                - Click the **Predict** button.
                - The model estimates energy consumption based on historical data.
                """
            )

# Run the app
if __name__ == '__main__':
    main()
