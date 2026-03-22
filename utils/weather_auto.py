import requests
import numpy as np
from geopy.geocoders import Nominatim

from utils.weather_predict import predict_weather_risk_sequence


import os
API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("OPENWEATHER_API_KEY environment variable is not set.")


def get_coordinates(city):
    geolocator = Nominatim(user_agent="weather-system")
    location = geolocator.geocode(city)
    return location.latitude, location.longitude


def fetch_past_weather(lat, lon, days=3):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    weather_data = []
    for i in range(days):
        entry = data["list"][i]
        temp = entry["main"]["temp"]
        humidity = entry["main"]["humidity"]
        wind = entry["wind"]["speed"]
        weather_data.append([temp, humidity, wind])
    return weather_data


def fetch_future_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    forecast = []
    for i in [0, 8, 16]:
        entry = data["list"][i]
        temp = entry["main"]["temp"]
        humidity = entry["main"]["humidity"]
        wind = entry["wind"]["speed"]
        forecast.append([temp, humidity, wind])
    return forecast


def build_weather_sequence(city):
    lat, lon = get_coordinates(city)
    past = fetch_past_weather(lat, lon)
    future = fetch_future_weather(lat, lon)
    sequence = np.array(past + future)
    return sequence


def automatic_weather_risk(city):
    sequence = build_weather_sequence(city)
    seq_input = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    risk = predict_weather_risk_sequence(seq_input)
    return risk


def automatic_weather_risk_with_data(city):
    """Returns (risk_score, {temp, humidity, wind, description}) using live API."""
    lat, lon = get_coordinates(city)

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url, timeout=8)
    current = resp.json()

    temp     = current["main"]["temp"]
    humidity = current["main"]["humidity"]
    wind     = current["wind"]["speed"]
    desc     = current["weather"][0]["description"].title() if current.get("weather") else "N/A"
    feels    = current["main"].get("feels_like", temp)
    pressure = current["main"].get("pressure", 1013)

    # build sequence for risk score
    sequence = build_weather_sequence(city)
    seq_input = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    risk = predict_weather_risk_sequence(seq_input)

    weather_info = {
        "temp": temp,
        "feels_like": feels,
        "humidity": humidity,
        "wind": wind,
        "description": desc,
        "pressure": pressure,
        "lat": lat,
        "lon": lon,
    }
    return risk, weather_info
