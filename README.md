---
title: DisasterWatch AI
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
---

# DisasterWatch AI

A Streamlit early-warning dashboard that combines weather signals, satellite-image classification, and text-based risk indicators to support flood and cyclone awareness.

## Features

- Weather, satellite-image, and news/text risk signals
- Flood and cyclone classification using the included ONNX model
- Interactive risk charts, geospatial map, and downloadable reports
- Manual analysis and automated location scans

## Run locally

Prerequisites: Python 3.10+ and an [OpenWeather API key](https://openweathermap.org/api).

```bash
git clone https://github.com/siddeshwar12/disasterwatch-ai.git
cd disasterwatch-ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`, add your API key, then start the dashboard:

```bash
streamlit run streamlit_app.py
```

Open the local URL that Streamlit prints, normally [http://localhost:8501](http://localhost:8501).

## Test checklist

1. Start an automated scan for a city such as Hyderabad or Chennai.
2. Confirm weather, map, and risk results load.
3. Run a manual scan with a supported image and a short news description.
4. Download the generated analysis report.

## Notes

The repository contains the model artifacts needed for local inference. Live weather and location data require a valid `OPENWEATHER_API_KEY` and an active internet connection.

## Responsible use

This project is a demonstration and decision-support tool. It is not an official alerting system and should not be used as the sole basis for emergency decisions.
