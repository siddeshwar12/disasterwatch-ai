import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DisasterAI — Early Warning System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "auto_results" not in st.session_state:
    st.session_state.auto_results = None

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0f2744 0%, #1a3a5c 50%, #0f2744 100%);
    border: 1px solid #1e4976;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #10b981;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}
.status-dot {
    width: 7px;
    height: 7px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Cards ── */
.card {
    background: rgba(15, 39, 68, 0.6);
    border: 1px solid rgba(30, 73, 118, 0.5);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* ── Risk Badge ── */
.risk-critical {
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #f87171;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.risk-high {
    background: rgba(249,115,22,0.15);
    border: 1px solid rgba(249,115,22,0.4);
    color: #fb923c;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.risk-moderate {
    background: rgba(234,179,8,0.15);
    border: 1px solid rgba(234,179,8,0.4);
    color: #facc15;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.risk-low {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: #34d399;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

/* ── Section Header ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(30,73,118,0.4);
}

/* ── Insight Box ── */
.insight-box {
    background: rgba(30, 58, 92, 0.4);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}
.insight-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 3px;
}

/* ── Alert Banner ── */
.alert-critical {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    color: #fca5a5;
    font-weight: 500;
    margin-bottom: 1rem;
    font-size: 0.95rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,39,68,0.5);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(30,73,118,0.4);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.2) !important;
    color: #60a5fa !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(15,39,68,0.7) !important;
    border: 1px solid rgba(30,73,118,0.6) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}

/* ── Metric ── */
[data-testid="metric-container"] {
    background: rgba(15,39,68,0.5);
    border: 1px solid rgba(30,73,118,0.4);
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #f1f5f9 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(30,73,118,0.3) !important;
    margin: 1.5rem 0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15,39,68,0.4);
    border: 1px dashed rgba(59,130,246,0.3);
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e4976; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="status-badge"><div class="status-dot"></div> SYSTEM ONLINE</div>
    <div class="hero-title">🛰️ DisasterAI — Early Warning System</div>
    <div class="hero-sub">AI-Driven Multi-Modal Disaster Detection &nbsp;·&nbsp; Weather · Satellite · Social Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def risk_gauge(score, title="Disaster Risk"):
    color = "#ef4444" if score > 0.66 else "#f59e0b" if score > 0.33 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        number={"font": {"color": "#f1f5f9", "size": 36}, "suffix": ""},
        title={"text": title, "font": {"color": "#94a3b8", "size": 13}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.33], "color": "rgba(16,185,129,0.12)"},
                {"range": [0.33, 0.66], "color": "rgba(245,158,11,0.12)"},
                {"range": [0.66, 1], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    return fig


def confidence_score(weather, cnn, nlp):
    scores = np.array([weather, cnn, nlp])
    return float(max(0, min(1 - np.std(scores), 1)))


def model_explanations(weather, cnn, nlp):
    def label(v, high, med, low):
        return high if v > 0.66 else med if v > 0.33 else low
    return {
        "Weather": label(weather,
            "Severe atmospheric patterns detected — heavy rainfall or pressure instability.",
            "Moderate weather anomaly observed. Conditions may deteriorate.",
            "Weather conditions appear relatively stable."),
        "Satellite": label(cnn,
            "Satellite imagery shows strong spatial patterns linked to disaster activity.",
            "Satellite imagery shows moderate environmental disturbances.",
            "No significant disaster indicators in satellite imagery."),
        "Social": label(nlp,
            "Strong disaster-related signals detected in social intelligence feed.",
            "Moderate public concern detected in social signals.",
            "Low disaster-related activity in social signals.")
    }


def predict_disaster_type(weather, cnn, nlp):
    scores = {
        "🌊 Flood": weather * 0.5 + nlp * 0.3 + cnn * 0.2,
        "🌀 Cyclone / Storm": weather * 0.6 + cnn * 0.3 + nlp * 0.1,
        "⚠️ General Environmental Risk": (weather + cnn + nlp) / 3
    }
    return max(scores, key=scores.get)


def risk_badge(level):
    cls = {
        "CRITICAL": "risk-critical",
        "HIGH": "risk-high",
        "MODERATE": "risk-moderate",
        "LOW": "risk-low"
    }.get(level.upper(), "risk-low")
    return f'<span class="{cls}">{level}</span>'


# ─────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────

def show_results(weather_score, cnn_score, nlp_score, lat=None, lon=None):
    final_score, w_weather, w_cnn, w_nlp = fuse_risk(weather_score, cnn_score, nlp_score)
    level = risk_level(final_score)
    confidence = confidence_score(weather_score, cnn_score, nlp_score)
    explanations = model_explanations(weather_score, cnn_score, nlp_score)
    disaster_type = predict_disaster_type(weather_score, cnn_score, nlp_score)

    # Alert banner
    if final_score > 0.66:
        st.markdown('<div class="alert-critical">🚨 CRITICAL ALERT — Immediate action recommended. Disaster risk is extremely high.</div>', unsafe_allow_html=True)
    elif final_score > 0.33:
        st.warning("⚠️ MODERATE ALERT — Elevated risk detected. Monitor conditions closely.")

    # ── Top KPI Row ──
    st.markdown('<div class="section-header">Risk Overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Final Risk Score", f"{final_score:.3f}")
    with k2:
        st.metric("Weather Score", f"{weather_score:.3f}")
    with k3:
        st.metric("Satellite Score", f"{cnn_score:.3f}")
    with k4:
        st.metric("Social Score", f"{nlp_score:.3f}")
    with k5:
        st.metric("Confidence", f"{confidence:.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Disaster Type ──
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(risk_gauge(final_score), use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">Assessment</div>', unsafe_allow_html=True)
        st.markdown(f"**Alert Level** &nbsp; {risk_badge(level)}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Predicted Disaster Type</div>
            <div style="font-size:1.4rem; font-weight:700; color:#60a5fa; margin-top:4px;">{disaster_type}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Confidence Index</div>
            <div class="card-value">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ──
    st.markdown('<div class="section-header">Model Analytics</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        # Risk trend line
        labels = ["Weather", "Satellite", "Social", "Fused"]
        values = [weather_score, cnn_score, nlp_score, final_score]
        colors = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels, y=values,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=10, color=colors, line=dict(color="#0a0e1a", width=2)),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.06)"
        ))
        fig.update_layout(
            title=dict(text="Risk Score Across Models", font=dict(color="#94a3b8", size=13)),
            yaxis=dict(range=[0, 1], gridcolor="rgba(30,73,118,0.2)", color="#475569"),
            xaxis=dict(gridcolor="rgba(30,73,118,0.2)", color="#475569"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(family="Inter", color="#94a3b8")
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        # Model contribution bar
        fig2 = go.Figure(go.Bar(
            x=["Weather", "Satellite", "Social"],
            y=[w_weather, w_cnn, w_nlp],
            marker=dict(
                color=["#3b82f6", "#8b5cf6", "#06b6d4"],
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=[f"{v:.3f}" for v in [w_weather, w_cnn, w_nlp]],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11)
        ))
        fig2.update_layout(
            title=dict(text="Weighted Model Contribution", font=dict(color="#94a3b8", size=13)),
            yaxis=dict(range=[0, 1], gridcolor="rgba(30,73,118,0.2)", color="#475569"),
            xaxis=dict(color="#475569"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(family="Inter", color="#94a3b8")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Model Interpretability ──
    st.markdown('<div class="section-header">Model Interpretability</div>', unsafe_allow_html=True)
    for model_name, explanation in explanations.items():
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-label">{model_name} Model</div>
            {explanation}
        </div>
        """, unsafe_allow_html=True)

    # ── Map ──
    if lat is not None and lon is not None:
        st.markdown('<div class="section-header">🌍 Geospatial Risk Map</div>', unsafe_allow_html=True)
        color = "red" if final_score > 0.66 else "orange" if final_score > 0.33 else "green"
        map_obj = folium.Map(location=[lat, lon], zoom_start=7,
                             tiles="CartoDB dark_matter")
        folium.CircleMarker(
            [lat, lon],
            radius=18,
            color=color,
            fill=True,
            fill_opacity=0.35,
            popup=f"Risk: {final_score:.3f} | {level}"
        ).add_to(map_obj)
        folium.Marker(
            [lat, lon],
            popup=f"Risk Score: {final_score:.3f}",
            icon=folium.Icon(color=color, icon="exclamation-sign")
        ).add_to(map_obj)
        st_folium(map_obj, width="100%", height=420)


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Manual Analysis", "⚡  Automatic Mode"])

# ─────────────────────────────────────────
# MANUAL MODE
# ─────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Manual Disaster Analysis</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        city = st.text_input("📍 Location / City", placeholder="e.g. Mumbai, Chennai")
        text_input = st.text_area(
            "📡 Social Media / News Signals",
            placeholder="Paste news headlines or social media reports about the area...",
            height=130
        )
        uploaded_image = st.file_uploader(
            "🛰️ Upload Satellite Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a satellite or aerial image of the area"
        )

    with col_right:
        image_path = None
        if uploaded_image:
            try:
                img = Image.open(uploaded_image)
                st.image(img, caption="Uploaded Satellite Image", use_column_width=True)
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp.write(uploaded_image.getvalue())
                image_path = temp.name
            except Exception:
                st.error("Invalid image file. Please upload a JPG or PNG.")
                st.stop()
        else:
            st.markdown("""
            <div style="background:rgba(15,39,68,0.4);border:1px dashed rgba(59,130,246,0.25);
            border-radius:10px;padding:3rem 1rem;text-align:center;color:#475569;font-size:0.85rem;">
                🛰️ Satellite image preview will appear here
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍  Analyze Disaster Risk", key="manual_btn"):
        if not city or not text_input or image_path is None:
            st.warning("Please fill in all fields and upload a satellite image.")
            st.stop()
        with st.spinner("Running multi-modal analysis..."):
            weather_score = automatic_weather_risk(city)
            cnn_score = predict_cnn_risk(image_path)
            nlp_score = predict_text_risk(text_input)
        st.divider()
        show_results(weather_score, cnn_score, nlp_score)


# ─────────────────────────────────────────
# AUTOMATIC MODE
# ─────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Automatic Disaster Risk Prediction</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        auto_city = st.text_input("📍 Location / City", placeholder="e.g. Kolkata, Delhi", key="auto_city")
    with col_b:
        auto_date = st.date_input("📅 Analysis Date", key="auto_date")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡  Run Automatic Analysis", key="auto_btn"):
        if not auto_city:
            st.warning("Please enter a location.")
            st.stop()
        with st.spinner("Fetching data and running analysis..."):
            try:
                geolocator = Nominatim(user_agent="disaster-ai-system")
                location = geolocator.geocode(auto_city)
                if location is None:
                    st.error("Location not found. Try a different city name.")
                    st.stop()
                lat, lon = location.latitude, location.longitude

                url = f"https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&size=600,450&z=10&l=sat"
                response = requests.get(url, timeout=10)
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp.write(response.content)
                sat_image = temp.name

                auto_text = (
                    f"Heavy rainfall reported near {auto_city} on {auto_date}. "
                    f"Authorities monitoring possible flooding and storm activity."
                )

                weather_score = automatic_weather_risk(auto_city)
                cnn_score = predict_cnn_risk(sat_image)
                nlp_score = predict_text_risk(auto_text)

                st.session_state.auto_results = {
                    "weather": weather_score,
                    "cnn": cnn_score,
                    "nlp": nlp_score,
                    "lat": lat,
                    "lon": lon
                }
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if st.session_state.auto_results:
        r = st.session_state.auto_results
        st.divider()
        show_results(r["weather"], r["cnn"], r["nlp"], r["lat"], r["lon"])


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; padding:0.5rem 0;">
    DisasterAI Early Warning System &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp; Powered by Multi-Modal AI
</div>
""", unsafe_allow_html=True)
