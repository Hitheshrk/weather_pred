import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model, scaler, and label encoder
@st.cache_resource
def load_model():
    with open('best_rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return model, scaler, label_encoder

# Load the model, scaler, and label encoder
model, scaler, label_encoder = load_model()

# Prediction function
def predict_weather(tmin, humidity, pres):
    # Create DataFrame with the user input
    input_data = pd.DataFrame({
        'tmin': [tmin],
        'humidity': [humidity],
        'pres': [pres]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_data_scaled)

    # Convert the predicted label into the corresponding weather condition
    weather_condition = label_encoder.inverse_transform(prediction)[0]

    return weather_condition

# Streamlit interface
st.title("Weather Prediction App")
st.write("Enter the weather parameters to predict the condition (Sunny, Cloudy, Rainy):")

# Input sliders for humidity, temperature, and pressure
tmin = st.slider("Minimum Temperature (°C)", min_value=-10, max_value=50, value=20, step=1)
humidity = st.slider("Relative Humidity (%)", min_value=0, max_value=100, value=60, step=1)
pres = st.slider("Atmospheric Pressure (hPa)", min_value=900, max_value=1100, value=1015, step=1)

# Button to trigger prediction
if st.button("Predict"):
    prediction = predict_weather(tmin, humidity, pres)
    st.write(f"The predicted weather condition is: **{prediction}**")

