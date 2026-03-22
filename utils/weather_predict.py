import numpy as np


# -----------------------------------
# PREDICT FROM SEQUENCE  (used by weather_auto.py)
# -----------------------------------

def predict_weather_risk_sequence(sequence):
    # sequence shape: (1, timesteps, features) or (timesteps, features)
    # features: [temp (°C), humidity (%), wind (m/s)]
    seq = np.array(sequence)
    if seq.ndim == 3:
        seq = seq[0]  # collapse batch dim → (timesteps, features)

    temp     = np.mean(seq[:, 0])   # °C
    humidity = np.mean(seq[:, 1])   # %
    wind     = np.mean(seq[:, 2])   # m/s

    # --- Temperature risk (0–1) ---
    # Safe zone: 10–35°C. Risk rises outside that range.
    if temp > 35:
        temp_risk = min((temp - 35) / 20.0, 1.0)    # 35→55°C = 0→1
    elif temp < 10:
        temp_risk = min((10 - temp) / 25.0, 1.0)    # 10→-15°C = 0→1
    else:
        temp_risk = 0.05   # small baseline for normal temps

    # --- Humidity risk (0–1) ---
    # Humidity always contributes — high humidity = flood/cyclone risk
    # Low humidity = drought/fire risk
    if humidity >= 70:
        humidity_risk = 0.2 + (humidity - 70) / 30.0 * 0.8   # 70%→100% = 0.2→1.0
    elif humidity >= 40:
        humidity_risk = (humidity - 40) / 30.0 * 0.2          # 40%→70% = 0→0.2
    else:
        humidity_risk = max(0.05, (40 - humidity) / 40.0 * 0.3)  # dry = mild risk

    # --- Wind risk (0–1) ---
    # Wind always contributes — higher wind = more risk
    wind_risk = min(wind / 25.0, 1.0)   # 0→25 m/s = 0→1 (linear)

    risk_score = 0.25 * temp_risk + 0.40 * humidity_risk + 0.35 * wind_risk

    return float(max(0.0, min(risk_score, 1.0)))


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