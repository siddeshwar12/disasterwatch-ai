import os
import numpy as np
import joblib

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scaler = joblib.load(os.path.join(_BASE, "models", "weather_scaler.pkl"))


def prepare_weather_sequence(weather_data):

    data = np.array(weather_data)

    # Scale data
    scaled = scaler.transform(data)

    # Reshape for LSTM
    sequence = scaled.reshape(1, 6, 3)

    return sequence