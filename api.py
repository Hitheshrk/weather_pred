from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the model, scaler, and label encoder
with open('best_rf_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# API endpoint for prediction using query parameters
@app.get("/predict")
async def predict_weather(humidity: float, pres: float, tmin: float):
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

    return {"weather_condition": weather_condition}
