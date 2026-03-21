import requests
import numpy as np
from geopy.geocoders import Nominatim

from utils.weather_predict import predict_weather_risk_sequence


API_KEY = "c2653e60394e02dbb312652f570b0a6d"


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
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    risk = predict_weather_risk_sequence(sequence)
    return risk
