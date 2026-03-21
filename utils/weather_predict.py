import numpy as np
from utils.weather_fetch import get_weather_data
from utils.weather_sequence import prepare_weather_sequence


# -----------------------------------
# WEATHER RISK PREDICTION (Lightweight)
# -----------------------------------

def predict_weather_risk(city):

    # Step 1: fetch weather data
    weather_data = get_weather_data(city)

    # Step 2: prepare sequence
    sequence = prepare_weather_sequence(weather_data)

    # Step 3: simple heuristic risk calculation
    rainfall = np.mean(sequence[:, 0])
    wind = np.mean(sequence[:, 1])
    humidity = np.mean(sequence[:, 2])

    # weighted risk score
    risk_score = 0.4 * rainfall + 0.3 * wind + 0.3 * humidity

    # normalize to 0–1 range
    risk_score = max(0, min(risk_score / 100, 1))

    return float(risk_score)


# -----------------------------------
# PREDICT FROM SEQUENCE
# -----------------------------------

def predict_weather_risk_sequence(sequence):
    # sequence shape: (1, timesteps, features) or (timesteps, features)
    seq = np.array(sequence)
    if seq.ndim == 3:
        seq = seq[0]  # collapse batch dim → (timesteps, features)

    rainfall = np.mean(seq[:, 0])
    wind = np.mean(seq[:, 1])
    humidity = np.mean(seq[:, 2])

    risk_score = 0.4 * rainfall + 0.3 * wind + 0.3 * humidity
    risk_score = max(0, min(risk_score / 100, 1))

    return float(risk_score)


# -----------------------------------
# RISK FORECAST (SIMULATION)
# -----------------------------------

def predict_weather_forecast(current_risk, steps=6):

    """
    Multi-step forecast simulation.
    """

    forecast = []

    base = current_risk

    for i in range(steps):

        drift = np.random.normal(0, 0.03)

        base = base + drift

        base = max(0, min(1, base))

        forecast.append(base)

    return forecast