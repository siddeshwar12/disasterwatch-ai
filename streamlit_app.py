import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tempfile
import requests
from PIL import Image
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

from utils.weather_auto import automatic_weather_risk
from utils.cnn_predict import predict_cnn_risk
from utils.nlp_predict import predict_text_risk
from utils.fusion import fuse_risk, risk_level


st.set_page_config(page_title="AI Disaster Monitoring Console", layout="wide")

# -----------------------------
# SESSION STATE
# -----------------------------

if "auto_results" not in st.session_state:
    st.session_state.auto_results = None


# -----------------------------
# DARK THEME
# -----------------------------

st.markdown("""
<style>
.stApp{
background-color:#0b1623;
color:white;
}
.block-container{
padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------

st.title("AI-Driven Hybrid Multi-Modal Disaster Early Warning System")
st.success("● SYSTEM ACTIVE")

st.divider()

tab1, tab2 = st.tabs(["Manual Mode", "Automatic Mode"])


# -----------------------------
# RISK GAUGE
# -----------------------------

def risk_gauge(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text':"Disaster Risk"},
        gauge={
            'axis':{'range':[0,1]},
            'bar':{'color':"orange"},
            'steps':[
                {'range':[0,0.33],'color':"green"},
                {'range':[0.33,0.66],'color':"yellow"},
                {'range':[0.66,1],'color':"red"},
            ]
        }
    ))

    fig.update_layout(height=280)

    return fig


# -----------------------------
# CONFIDENCE
# -----------------------------

def confidence_score(weather, cnn, nlp):

    scores = np.array([weather, cnn, nlp])
    std = np.std(scores)
    confidence = 1 - std

    return max(0, min(confidence,1))


# -----------------------------
# MODEL EXPLANATIONS
# -----------------------------

def model_explanations(weather, cnn, nlp):

    explanations = {}

    if weather > 0.66:
        explanations["Weather"] = "Weather model detected severe atmospheric patterns such as heavy rainfall or pressure instability."
    elif weather > 0.33:
        explanations["Weather"] = "Weather conditions indicate moderate anomaly that may evolve into hazardous conditions."
    else:
        explanations["Weather"] = "Weather conditions appear relatively stable."

    if cnn > 0.66:
        explanations["Satellite"] = "Satellite imagery indicates strong spatial patterns associated with disaster activity."
    elif cnn > 0.33:
        explanations["Satellite"] = "Satellite imagery shows moderate environmental disturbances."
    else:
        explanations["Satellite"] = "Satellite imagery does not show strong disaster indicators."

    if nlp > 0.66:
        explanations["Social"] = "Social intelligence signals report strong disaster-related discussions and alerts."
    elif nlp > 0.33:
        explanations["Social"] = "Social signals indicate moderate public concern."
    else:
        explanations["Social"] = "Social signals show low disaster-related activity."

    return explanations


# -----------------------------
# DISASTER TYPE
# -----------------------------

def predict_disaster_type(weather, cnn, nlp):

    scores = {
        "Flood": (weather*0.5 + nlp*0.3 + cnn*0.2),
        "Cyclone/Storm": (weather*0.6 + cnn*0.3 + nlp*0.1),
        "General Environmental Risk": (weather + cnn + nlp)/3
    }

    disaster = max(scores, key=scores.get)

    return disaster


# -----------------------------
# MAIN RESULTS DISPLAY
# -----------------------------

def show_results(weather_score, cnn_score, nlp_score, lat=None, lon=None):

    final_score, w_weather, w_cnn, w_nlp = fuse_risk(
        weather_score,
        cnn_score,
        nlp_score
    )

    level = risk_level(final_score)

    confidence = confidence_score(weather_score, cnn_score, nlp_score)

    explanations = model_explanations(weather_score, cnn_score, nlp_score)

    disaster_type = predict_disaster_type(weather_score, cnn_score, nlp_score)

    if final_score > 0.7:
        st.error("🚨 CRITICAL ALERT — Immediate Action Recommended")

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.subheader("Overall Disaster Risk")
        st.plotly_chart(risk_gauge(final_score),use_container_width=True)
        st.metric("Risk Level",level)

    with c2:
        st.metric("Weather Score",round(weather_score,3))

    with c3:
        st.metric("Satellite Score",round(cnn_score,3))

    with c4:
        st.metric("Social Score",round(nlp_score,3))

    st.divider()

    # -----------------------------
    # RISK TREND
    # -----------------------------

    st.subheader("Risk Trend")

    trend = [weather_score, cnn_score, nlp_score, final_score]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=trend,
        mode='lines+markers',
        line=dict(color="#4da6ff",width=3)
    ))

    fig.update_layout(
        yaxis_title="Risk Score",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

    # -----------------------------
    # MODEL CONTRIBUTION
    # -----------------------------

    st.subheader("Model Contribution")

    labels = ["Weather", "Satellite", "Social"]
    values = [w_weather, w_cnn, w_nlp]

    fig = go.Figure(data=[go.Bar(x=labels,y=values)])

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # -----------------------------
    # DETECTION RESULTS
    # -----------------------------

    st.subheader("Detection Results")

    colA,colB = st.columns(2)

    with colA:

        st.metric("Final Risk Score",round(final_score,3))
        st.metric("Confidence Score",round(confidence,3))
        st.metric("Alert Level",level)

    with colB:

        st.subheader("Expected Disaster Type")
        st.success(disaster_type)

    # -----------------------------
    # MODEL EXPLANATIONS
    # -----------------------------

    st.divider()
    st.subheader("Model Interpretability")

    st.info(explanations["Weather"])
    st.info(explanations["Satellite"])
    st.info(explanations["Social"])

    # -----------------------------
    # MAP
    # -----------------------------

    if lat is not None and lon is not None:

        st.divider()
        st.subheader("🌍 Disaster Monitoring Map")

        if final_score < 0.33:
            color="green"
        elif final_score < 0.66:
            color="orange"
        else:
            color="red"

        map_obj = folium.Map(location=[lat, lon], zoom_start=7)

        folium.Marker(
            [lat,lon],
            popup=f"Risk Score: {round(final_score,3)}",
            icon=folium.Icon(color=color)
        ).add_to(map_obj)

        st_folium(map_obj,width=700,height=400)


# -----------------------------
# MANUAL MODE
# -----------------------------

with tab1:

    st.header("Manual Disaster Analysis")

    city = st.text_input("Location")
    text_input = st.text_area("Social Media Signals")

    uploaded_image = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

    image_path=None

    if uploaded_image:

        try:
            image = Image.open(uploaded_image)
            st.image(image,width=350)

            temp = tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
            temp.write(uploaded_image.getvalue())

            image_path=temp.name
        except Exception:
            st.error("Invalid image file. Please upload a JPG or PNG image.")
            st.stop()

    if st.button("Analyze Disaster Risk"):

        if city=="" or text_input=="" or image_path is None:

            st.warning("Please fill all inputs")
            st.stop()

        weather_score=automatic_weather_risk(city)
        cnn_score=predict_cnn_risk(image_path)
        nlp_score=predict_text_risk(text_input)

        show_results(weather_score,cnn_score,nlp_score)


# -----------------------------
# AUTOMATIC MODE
# -----------------------------

with tab2:

    st.header("Automatic Disaster Risk Prediction")

    auto_city=st.text_input("Enter Location")
    auto_date=st.date_input("Select Date")

    if st.button("Run Automatic Analysis"):

        if auto_city=="":

            st.warning("Enter location")
            st.stop()

        geolocator=Nominatim(user_agent="disaster-system")

        location=geolocator.geocode(auto_city)

        lat=location.latitude
        lon=location.longitude

        url=f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&size=600,450&z=10&l=sat"

        response=requests.get(url)

        temp=tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
        temp.write(response.content)

        sat_image=temp.name

        auto_text=f"""
        Heavy rainfall reported near {auto_city} on {auto_date}.
        Authorities monitoring possible flooding.
        """

        weather_score=automatic_weather_risk(auto_city)
        cnn_score=predict_cnn_risk(sat_image)
        nlp_score=predict_text_risk(auto_text)

        st.session_state.auto_results={
            "weather":weather_score,
            "cnn":cnn_score,
            "nlp":nlp_score,
            "lat":lat,
            "lon":lon
        }

    if st.session_state.auto_results:

        r=st.session_state.auto_results

        show_results(
            r["weather"],
            r["cnn"],
            r["nlp"],
            r["lat"],
            r["lon"]
        )

st.divider()
st.caption("AI Disaster Monitoring Console v1.0")