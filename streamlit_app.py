import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tempfile, requests, re
from datetime import datetime
from PIL import Image
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

from utils.weather_auto import automatic_weather_risk, automatic_weather_risk_with_data
from utils.cnn_predict import predict_cnn_risk
from utils.nlp_predict import predict_text_risk
from utils.fusion import fuse_risk, risk_level

st.set_page_config(page_title="DisasterWatch AI", page_icon="🌍",
                   layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════
#  LIGHT THEME CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

.stApp { background: #f0f4f8; color: #1e293b; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAV ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0.85rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
    position: sticky; top: 0; z-index: 999;
}
.brand { display:flex; align-items:center; gap:0.8rem; }
.brand-icon {
    width:40px; height:40px; border-radius:10px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex; align-items:center; justify-content:center;
    font-size:1.2rem; box-shadow:0 2px 8px rgba(37,99,235,0.3);
}
.brand-name {
    font-size:1.1rem; font-weight:800; color:#1e293b; letter-spacing:-0.02em;
}
.brand-tag { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; }
.live-pill {
    display:flex; align-items:center; gap:0.45rem;
    background:#f0fdf4; border:1px solid #bbf7d0;
    border-radius:20px; padding:0.28rem 0.85rem;
    font-size:0.72rem; font-weight:700; color:#16a34a; letter-spacing:0.05em;
}
.live-dot { width:7px; height:7px; background:#22c55e; border-radius:50%; animation:pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

/* ── WRAPPER ── */
.wrap { padding: 2rem 2.5rem 4rem; }

/* ── CARDS ── */
.card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
    padding:1.4rem 1.6rem; box-shadow:0 2px 12px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow:0 6px 24px rgba(0,0,0,0.09); transform:translateY(-1px); }

/* ── METRIC TILES ── */
.mtile {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
    padding:1.1rem 1.3rem; position:relative; overflow:hidden;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s;
}
.mtile:hover { box-shadow:0 6px 20px rgba(0,0,0,0.08); transform:translateY(-2px); }
.mtile::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,#2563eb,#7c3aed); opacity:0.5;
}
.mtile.danger::after { background:linear-gradient(90deg,#ef4444,#f97316); }
.mtile.warn::after   { background:linear-gradient(90deg,#f59e0b,#eab308); }
.mtile.safe::after   { background:linear-gradient(90deg,#22c55e,#10b981); }
.mtile.blue::after   { background:linear-gradient(90deg,#2563eb,#7c3aed); }
.mlabel { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.45rem; }
.mval   { font-size:1.9rem; font-weight:800; font-family:'JetBrains Mono',monospace; line-height:1; }
.mval.danger { color:#ef4444; }
.mval.warn   { color:#f59e0b; }
.mval.safe   { color:#22c55e; }
.mval.blue   { color:#2563eb; }
.msub { font-size:0.7rem; color:#cbd5e1; margin-top:0.3rem; }

/* ── WEATHER PILLS ── */
.wxcard {
    background:linear-gradient(135deg,#eff6ff,#f5f3ff);
    border:1px solid #ddd6fe; border-radius:12px;
    padding:0.9rem 1rem; text-align:center;
}
.wxval { font-size:1.4rem; font-weight:700; color:#2563eb; font-family:'JetBrains Mono',monospace; }
.wxlbl { font-size:0.65rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.2rem; }

/* ── ALERT BANNERS ── */
.abanner {
    border-radius:12px; padding:1rem 1.4rem;
    display:flex; align-items:center; gap:0.9rem;
    margin-bottom:1.5rem; font-weight:600; font-size:0.92rem;
}
.abanner.crit { background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #ef4444; color:#b91c1c; }
.abanner.med  { background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b; color:#92400e; }
.abanner.low  { background:#f0fdf4; border:1px solid #bbf7d0; border-left:4px solid #22c55e; color:#15803d; }

/* ── SECTION LABEL ── */
.slabel {
    font-size:0.68rem; font-weight:700; color:#2563eb;
    text-transform:uppercase; letter-spacing:0.15em;
    margin:1.8rem 0 0.9rem; display:flex; align-items:center; gap:0.6rem;
}
.slabel::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#dbeafe,transparent); }

/* ── INSIGHT ROWS ── */
.irow {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:0.7rem 1rem; margin-bottom:0.45rem;
    font-size:0.83rem; color:#475569;
}
.irow.danger { border-left:3px solid #ef4444; background:#fef2f2; color:#b91c1c; }
.irow.warn   { border-left:3px solid #f59e0b; background:#fffbeb; color:#92400e; }
.irow.safe   { border-left:3px solid #22c55e; background:#f0fdf4; color:#15803d; }

/* ── REC ITEMS ── */
.ritem {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:0.65rem 1rem; margin-bottom:0.4rem; font-size:0.82rem; color:#475569;
}
.ritem.urgent  { border-left:3px solid #ef4444; background:#fef2f2; color:#b91c1c; }
.ritem.caution { border-left:3px solid #f59e0b; background:#fffbeb; color:#92400e; }
.ritem.safe    { border-left:3px solid #22c55e; background:#f0fdf4; color:#15803d; }

/* ── NEWS CARD ── */
.news-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
    padding:1rem 1.2rem; margin-bottom:0.7rem;
    box-shadow:0 1px 6px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.news-card:hover { box-shadow:0 4px 16px rgba(0,0,0,0.08); }
.news-title { font-size:0.88rem; font-weight:600; color:#1e293b; margin-bottom:0.3rem; line-height:1.4; }
.news-meta  { font-size:0.72rem; color:#94a3b8; }
.news-tag {
    display:inline-block; font-size:0.65rem; font-weight:700;
    padding:0.15rem 0.55rem; border-radius:10px; margin-right:0.4rem;
    text-transform:uppercase; letter-spacing:0.05em;
}
.news-tag.alert  { background:#fef2f2; color:#ef4444; border:1px solid #fecaca; }
.news-tag.info   { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }
.news-tag.normal { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }

/* ── HISTORY ROW ── */
.hrow {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
    padding:0.7rem 1.2rem; margin-bottom:0.4rem;
    display:flex; justify-content:space-between; align-items:center;
    font-size:0.82rem; box-shadow:0 1px 4px rgba(0,0,0,0.03);
}
.hbadge { padding:0.18rem 0.65rem; border-radius:20px; font-size:0.7rem; font-weight:700; }
.hbadge.High   { background:#fef2f2; color:#ef4444; border:1px solid #fecaca; }
.hbadge.Medium { background:#fffbeb; color:#f59e0b; border:1px solid #fde68a; }
.hbadge.Low    { background:#f0fdf4; color:#22c55e; border:1px solid #bbf7d0; }

/* ── DIS BADGE ── */
.dbadge {
    display:inline-block; background:#f5f3ff; border:1px solid #ddd6fe;
    border-radius:20px; padding:0.3rem 0.9rem;
    font-size:0.82rem; font-weight:600; color:#7c3aed;
}

/* ── INPUTS ── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background:#ffffff !important; border:1px solid #e2e8f0 !important;
    color:#1e293b !important; border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color:#2563eb !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stTextInput label,.stTextArea label,.stFileUploader label {
    color:#64748b !important; font-size:0.75rem !important;
    font-weight:600 !important; text-transform:uppercase; letter-spacing:0.07em;
}

/* ── BUTTON ── */
.stButton>button {
    background:linear-gradient(135deg,#2563eb,#7c3aed) !important;
    color:#ffffff !important; border:none !important; border-radius:10px !important;
    padding:0.65rem 2rem !important; font-weight:700 !important;
    font-size:0.88rem !important; width:100% !important;
    box-shadow:0 4px 14px rgba(37,99,235,0.3) !important;
    transition:opacity 0.2s,transform 0.15s !important;
}
.stButton>button:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton>button {
    background:#eff6ff !important; color:#2563eb !important;
    border:1px solid #bfdbfe !important; border-radius:10px !important;
    font-weight:600 !important; width:100% !important;
}
.stDownloadButton>button:hover { background:#dbeafe !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background:#ffffff !important; border-radius:12px !important;
    padding:5px !important; gap:4px !important;
    border:1px solid #e2e8f0 !important;
    box-shadow:0 1px 6px rgba(0,0,0,0.05) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:9px !important; color:#94a3b8 !important;
    font-weight:600 !important; font-size:0.84rem !important;
    padding:0.5rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#eff6ff,#f5f3ff) !important;
    color:#2563eb !important; border:1px solid #bfdbfe !important;
}

/* ── FILE UPLOADER ── */
.stFileUploader>div {
    background:#fafafa !important; border:1.5px dashed #cbd5e1 !important;
    border-radius:12px !important;
}

/* ── MISC ── */
hr { border-color:#e2e8f0 !important; margin:1.5rem 0 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#f0f4f8; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
#MainMenu,footer,header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
if "auto_results"     not in st.session_state: st.session_state.auto_results     = None
if "analysis_history" not in st.session_state: st.session_state.analysis_history = []

# ═══════════════════════════════════════════════════════════════
#  TOP NAV
# ═══════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%d %b %Y  %H:%M")
st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🌍</div>
    <div>
      <div class="brand-name">DisasterWatch AI</div>
      <div class="brand-tag">Early Warning Intelligence Platform</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:1.2rem">
    <span style="font-size:0.75rem;color:#94a3b8;font-family:'JetBrains Mono',monospace">{now_str}</span>
    <div class="live-pill"><div class="live-dot"></div>LIVE</div>
  </div>
</div>
<div class="wrap">
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════
def scls(s):  return "danger" if s>=0.66 else "warn" if s>=0.33 else "safe"
def shex(s):  return "#ef4444" if s>=0.66 else "#f59e0b" if s>=0.33 else "#22c55e"

def risk_gauge(score):
    c = shex(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score,3),
        number={"font":{"size":42,"color":c,"family":"JetBrains Mono"}},
        gauge={
            "axis":{"range":[0,1],"tickcolor":"#cbd5e1","tickfont":{"color":"#94a3b8","size":10},"nticks":6},
            "bar":{"color":c,"thickness":0.22},
            "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0,0.33],  "color":"rgba(34,197,94,0.08)"},
                {"range":[0.33,0.66],"color":"rgba(245,158,11,0.08)"},
                {"range":[0.66,1],  "color":"rgba(239,68,68,0.08)"},
            ],
            "threshold":{"line":{"color":c,"width":3},"value":score},
        },
    ))
    fig.update_layout(height=230,margin=dict(l=20,r=20,t=30,b=10),
                      paper_bgcolor="rgba(0,0,0,0)",font_color="#1e293b")
    return fig

def bar_chart(labels, values, colors):
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, marker_line_width=0,
        text=[f"{v:.2f}" for v in values], textposition="outside",
        textfont={"color":"#64748b","size":11},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#64748b", height=240, margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(range=[0,1.15],gridcolor="#f1f5f9",zeroline=False,tickfont=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=11)),
    )
    return fig

def trend_chart(w,c,n,f):
    vals=[w,c,n,f]; lbls=["Weather","Satellite","Social","Final"]
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=lbls,y=vals,mode="lines+markers",
        line=dict(color="#2563eb",width=2.5),
        marker=dict(size=11,color=[shex(v) for v in vals],
                    line=dict(color="#ffffff",width=2)),
        fill="tozeroy",fillcolor="rgba(37,99,235,0.06)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font_color="#64748b",height=240,margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(range=[0,1.15],gridcolor="#f1f5f9",zeroline=False,tickfont=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(size=11)),
    )
    return fig

def prob_chart(w,c,n):
    fl=0.50*w+0.30*c+0.20*n; cy=0.40*w+0.40*c+0.20*n
    tot=(fl+cy) or 1; fp,cp=fl/tot,cy/tot
    fig=go.Figure()
    fig.add_trace(go.Bar(name="🌊 Flood",x=[""],y=[fp],
        marker_color="#3b82f6",text=[f"{fp:.0%}"],textposition="inside",
        textfont={"color":"#fff","size":13,"family":"JetBrains Mono"}))
    fig.add_trace(go.Bar(name="🌀 Cyclone",x=[""],y=[cp],
        marker_color="#8b5cf6",text=[f"{cp:.0%}"],textposition="inside",
        textfont={"color":"#fff","size":13,"family":"JetBrains Mono"}))
    fig.update_layout(
        barmode="stack",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font_color="#64748b",height=240,margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h",y=1.12,x=0.5,xanchor="center",
                    font=dict(size=11),bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[0,1.1],gridcolor="#f1f5f9",zeroline=False,
                   tickformat=".0%",tickfont=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig,fp,cp

def show_alert(score):
    lvl=risk_level(score)
    if lvl=="High":
        st.markdown(f'<div class="abanner crit">🚨 <b>CRITICAL ALERT</b> — Immediate emergency response required &nbsp;·&nbsp; Score: {score:.3f}</div>',unsafe_allow_html=True)
    elif lvl=="Medium":
        st.markdown(f'<div class="abanner med">⚠️ <b>ELEVATED RISK</b> — Monitor situation closely &nbsp;·&nbsp; Score: {score:.3f}</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="abanner low">✅ <b>STABLE CONDITIONS</b> — No immediate threat detected &nbsp;·&nbsp; Score: {score:.3f}</div>',unsafe_allow_html=True)

def model_insights(w,c,n,cls):
    rows=[]
    rows.append(("danger",f"🌧️ Severe atmospheric conditions — score {w:.2f}") if w>0.66
           else ("warn",  f"🌦️ Moderate weather anomaly — score {w:.2f}") if w>0.33
           else ("safe",  f"☀️ Weather appears stable — score {w:.2f}"))
    rows.append(("danger",f"🛰️ Strong {cls} pattern detected — confidence {c:.2f}") if c>0.66
           else ("warn",  f"🛰️ Moderate {cls} indicators — confidence {c:.2f}") if c>0.33
           else ("safe",  f"🛰️ No significant disaster pattern — score {c:.2f}"))
    rows.append(("danger",f"📡 High-risk signals in news feeds — NLP {n:.2f}") if n>0.66
           else ("warn",  f"📡 Moderate public concern signals — score {n:.2f}") if n>0.33
           else ("safe",  f"📡 Low disaster activity in media — score {n:.2f}"))
    return rows

def get_recs(level,fp,cp):
    dom="Flood" if fp>=cp else "Cyclone"
    if level=="High":
        r=[("urgent","🚨 Activate emergency response protocols immediately"),
           ("urgent","📢 Issue public evacuation advisory for at-risk zones"),
           ("urgent","🏥 Put hospitals and emergency services on high alert"),
           ("urgent","🚧 Close flood-prone roads and critical infrastructure")]
        r+=[("urgent","🌊 Deploy flood barriers and pre-position rescue boats")] if dom=="Flood" \
          else[("urgent","🌀 Evacuate coastal areas and secure loose structures")]
    elif level=="Medium":
        r=[("caution","⚠️ Issue weather advisory to local authorities"),
           ("caution","📻 Broadcast warnings via radio and social media"),
           ("caution","🏠 Advise residents to stock emergency supplies"),
           ("caution","🔍 Increase monitoring frequency to every 3 hours")]
        r+=[("caution","🌊 Monitor river levels and drainage systems")] if dom=="Flood" \
          else[("caution","🌀 Track cyclone trajectory via meteorological dept")]
    else:
        r=[("safe","✅ Continue routine monitoring — no immediate action needed"),
           ("safe","📊 Log current readings for trend analysis"),
           ("safe","🔔 Keep alert systems on standby")]
    return r,dom

def wx_cards(info):
    st.markdown('<div class="slabel">🌤️ Live Weather Conditions</div>',unsafe_allow_html=True)
    cols=st.columns(5)
    items=[(f"{info['temp']:.1f}°C","Temperature"),(f"{info['feels_like']:.1f}°C","Feels Like"),
           (f"{info['humidity']}%","Humidity"),(f"{info['wind']:.1f} m/s","Wind Speed"),
           (f"{info['pressure']} hPa","Pressure")]
    for col,(val,lbl) in zip(cols,items):
        with col:
            st.markdown(f'<div class="wxcard"><div class="wxval">{val}</div><div class="wxlbl">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.78rem;margin-top:0.5rem">🌐 Condition: <b style="color:#475569">{info.get("description","N/A")}</b></p>',unsafe_allow_html=True)

def build_report(city,final,level,ws,cs,ns,cnn_class,fp,cp,winfo=None):
    t=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines=["="*56,"   DISASTERWATCH AI — ANALYSIS REPORT","="*56,
           f"Location     : {city}",f"Generated At : {t}","",
           "── RISK SUMMARY ──────────────────────────────────────",
           f"Final Risk Score : {final:.3f}",f"Alert Level      : {level}",
           f"Dominant Disaster: {'Flood' if fp>=cp else 'Cyclone'}",f"CNN Detection    : {cnn_class}","",
           "── MODEL SCORES ──────────────────────────────────────",
           f"Weather Score  : {ws:.3f}",f"Satellite Score: {cs:.3f}",f"Social Score   : {ns:.3f}","",
           "── DISASTER PROBABILITIES ────────────────────────────",
           f"Flood   : {fp:.1%}",f"Cyclone : {cp:.1%}"]
    if winfo:
        lines+=["","── LIVE WEATHER DATA ─────────────────────────────────",
                f"Temperature  : {winfo['temp']:.1f}°C (Feels {winfo['feels_like']:.1f}°C)",
                f"Humidity     : {winfo['humidity']}%",f"Wind Speed   : {winfo['wind']:.1f} m/s",
                f"Pressure     : {winfo['pressure']} hPa",f"Condition    : {winfo['description']}"]
    lines+=["","="*56,"Generated by DisasterWatch AI v2.0"]
    return "\n".join(lines)

def add_hist(city,score,level,cnn_class):
    st.session_state.analysis_history.append(
        {"time":datetime.now().strftime("%H:%M:%S"),"city":city,
         "score":score,"level":level,"type":cnn_class})
    st.session_state.analysis_history=st.session_state.analysis_history[-10:]

# ═══════════════════════════════════════════════════════════════
#  LOCATION INTELLIGENCE  (news + weather articles)
# ═══════════════════════════════════════════════════════════════
def fetch_location_intel(city: str):
    """Fetch news headlines + weather description for the city."""
    articles = []

    # ── Google News RSS ──────────────────────────────────────────
    queries = [
        (f"{city} flood cyclone disaster warning", "alert"),
        (f"{city} weather today", "info"),
    ]
    seen = set()
    for q, tag in queries:
        try:
            url  = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en&gl=IN&ceid=IN:en"
            resp = requests.get(url, timeout=6)
            titles = re.findall(r"<title>(.*?)</title>", resp.text)
            links  = re.findall(r"<link>(.*?)</link>",  resp.text)
            dates  = re.findall(r"<pubDate>(.*?)</pubDate>", resp.text)
            for i, title in enumerate(titles[2:10]):   # skip feed-level titles
                clean = re.sub(r"<[^>]+>","",title).strip()
                if clean and clean not in seen and len(clean) > 15:
                    seen.add(clean)
                    link = links[i+2] if i+2 < len(links) else "#"
                    date = dates[i][:16] if i < len(dates) else ""
                    articles.append({"title":clean,"tag":tag,"link":link,"date":date})
        except Exception:
            pass

    return articles[:12]


def show_location_intel(city: str):
    st.markdown('<div class="slabel">📰 Location Intelligence — News & Weather Articles</div>',
                unsafe_allow_html=True)

    with st.spinner("Fetching latest news and articles..."):
        articles = fetch_location_intel(city)

    if not articles:
        st.markdown("""
        <div style="background:#f8fafc;border:1px dashed #e2e8f0;border-radius:12px;
                    padding:2rem;text-align:center;color:#94a3b8;font-size:0.85rem">
          📭 No recent articles found for this location.
        </div>""", unsafe_allow_html=True)
        return

    # split into 2 columns
    col_a, col_b = st.columns(2)
    for i, art in enumerate(articles):
        tag_cls   = art["tag"]
        tag_label = "⚠️ Alert" if tag_cls=="alert" else "ℹ️ Info"
        card_html = f"""
        <div class="news-card">
          <div style="margin-bottom:0.4rem">
            <span class="news-tag {tag_cls}">{tag_label}</span>
            <span class="news-meta">{art['date']}</span>
          </div>
          <div class="news-title">{art['title']}</div>
          <div style="margin-top:0.5rem">
            <a href="{art['link']}" target="_blank"
               style="font-size:0.72rem;color:#2563eb;text-decoration:none;font-weight:600">
              Read article →
            </a>
          </div>
        </div>"""
        if i % 2 == 0:
            with col_a: st.markdown(card_html, unsafe_allow_html=True)
        else:
            with col_b: st.markdown(card_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SHOW RESULTS
# ═══════════════════════════════════════════════════════════════
def show_results(ws, cs, ns, cnn_class="Unknown",
                 lat=None, lon=None, city="", winfo=None):

    final,ww,wc,wn = fuse_risk(ws,cs,ns)
    level      = risk_level(final)
    confidence = float(max(0, 1-np.std([ws,cs,ns])))
    add_hist(city or "Unknown", final, level, cnn_class)

    show_alert(final)

    # ── Live weather cards ────────────────────────────────────────
    if winfo:
        wx_cards(winfo)

    # ── Gauge + metric tiles ──────────────────────────────────────
    st.markdown('<div class="slabel">📊 Risk Assessment</div>',unsafe_allow_html=True)
    g,m1,m2,m3,m4 = st.columns([2,1,1,1,1])
    with g:
        st.markdown('<div class="card" style="padding:0.4rem">',unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(final),use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    for col,(lbl,val,cls) in zip([m1,m2,m3,m4],[
        ("Weather",ws,scls(ws)),("Satellite",cs,scls(cs)),
        ("Social",ns,scls(ns)),("Confidence",confidence,"blue")]):
        with col:
            disp=f"{val:.2f}" if lbl!="Confidence" else f"{val:.0%}"
            st.markdown(f"""<div class="mtile {cls}">
              <div class="mlabel">{lbl}</div>
              <div class="mval {cls}">{disp}</div>
              <div class="msub">{'Score' if lbl!='Confidence' else 'Agreement'}</div>
            </div>""",unsafe_allow_html=True)

    # ── 3 charts ──────────────────────────────────────────────────
    st.markdown('<div class="slabel">📈 Signal Analysis</div>',unsafe_allow_html=True)
    ch1,ch2,ch3 = st.columns(3)
    for col,title,fig_fn in [
        (ch1,"Risk Trend",         lambda: trend_chart(ws,cs,ns,final)),
        (ch2,"Model Contribution", lambda: bar_chart(["Weather","Satellite","Social"],
                                                     [ww,wc,wn],[shex(ww),shex(wc),shex(wn)])),
        (ch3,"Disaster Probability",None),
    ]:
        with col:
            st.markdown(f'<div class="card" style="padding:1rem 1rem 0.4rem">',unsafe_allow_html=True)
            st.markdown(f'<p style="color:#2563eb;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.2rem">{title}</p>',unsafe_allow_html=True)
            if title=="Disaster Probability":
                pfig,fp,cp = prob_chart(ws,cs,ns)
                st.plotly_chart(pfig,use_container_width=True)
            else:
                st.plotly_chart(fig_fn(),use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

    # ── Intelligence report row ───────────────────────────────────
    st.markdown('<div class="slabel">🔍 Intelligence Report</div>',unsafe_allow_html=True)
    ca,cb,cc = st.columns(3)
    dom_lbl = "🌊 Flood" if fp>=cp else "🌀 Cyclone"

    with ca:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<p style="color:#2563eb;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem">Detection Summary</p>',unsafe_allow_html=True)
        for k,v in [
            ("Final Risk Score",f'<span style="color:{shex(final)};font-family:JetBrains Mono,monospace;font-weight:800;font-size:1.25rem">{final:.3f}</span>'),
            ("Alert Level",     f'<span style="color:{shex(final)};font-weight:700">{level}</span>'),
            ("Confidence",      f'<span style="color:#2563eb;font-weight:600">{confidence:.1%}</span>'),
            ("Likely Disaster", f'<span style="color:#7c3aed;font-weight:600">{dom_lbl}</span>'),
            ("CNN Detection",   f'<span class="dbadge">{cnn_class}</span>'),
        ]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.55rem 0;border-bottom:1px solid #f1f5f9">
              <span style="color:#94a3b8;font-size:0.8rem">{k}</span>{v}</div>""",unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with cb:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<p style="color:#2563eb;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem">Model Interpretability</p>',unsafe_allow_html=True)
        for cls,txt in model_insights(ws,cs,ns,cnn_class):
            st.markdown(f'<div class="irow {cls}">{txt}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with cc:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<p style="color:#2563eb;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.8rem">Recommended Actions</p>',unsafe_allow_html=True)
        recs,_ = get_recs(level,fp,cp)
        for cls,txt in recs:
            st.markdown(f'<div class="ritem {cls}">{txt}</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    # ── Location Intelligence (news) ──────────────────────────────
    if city:
        show_location_intel(city)

    # ── Export ────────────────────────────────────────────────────
    st.markdown("")
    report=build_report(city,final,level,ws,cs,ns,cnn_class,fp,cp,winfo)
    st.download_button("📄 Download Analysis Report",data=report,
        file_name=f"disasterwatch_{city.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain")

    # ── Map ───────────────────────────────────────────────────────
    if lat is not None and lon is not None:
        st.markdown('<div class="slabel">🌍 Geospatial Monitor</div>',unsafe_allow_html=True)
        color="green" if final<0.33 else "orange" if final<0.66 else "red"
        m=folium.Map(location=[lat,lon],zoom_start=7,tiles="OpenStreetMap")
        folium.CircleMarker([lat,lon],radius=20,color=color,fill=True,fill_opacity=0.35,
            popup=f"<b>{level} Risk</b><br>Score: {final:.3f}<br>Type: {cnn_class}").add_to(m)
        folium.Marker([lat,lon],popup=f"Risk: {final:.3f}",
            icon=folium.Icon(color=color,icon="exclamation-sign")).add_to(m)
        st_folium(m,width=None,height=400)

# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "  🔬  Manual Analysis  ",
    "  ⚡  Auto Scan  ",
    "  📋  Mission Log  ",
])

# ── MANUAL ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="slabel">🔬 Manual Multi-Modal Analysis</div>',unsafe_allow_html=True)
    left,right = st.columns(2,gap="large")

    with left:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        city_m  = st.text_input("📍 Location",placeholder="e.g. Chennai, Mumbai",key="m_city")
        text_m  = st.text_area("📢 Social / News Signals",
                               placeholder="Paste news headlines or social media reports here...",
                               height=130,key="m_text")
        st.markdown('</div>',unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        uploaded = st.file_uploader("🛰️ Satellite Image",type=["jpg","jpeg","png"],key="m_img")
        image_path=None
        if uploaded:
            try:
                img=Image.open(uploaded)
                st.image(img,use_container_width=True,caption="Uploaded image")
                tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
                tmp.write(uploaded.getvalue()); image_path=tmp.name
            except Exception:
                st.error("Invalid image file."); st.stop()
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔬 Run Analysis",key="manual_btn"):
        if not city_m or not text_m or image_path is None:
            st.warning("Fill in all three inputs — location, social signals, and satellite image.")
            st.stop()
        with st.spinner("Running multi-modal analysis..."):
            try:    ws,winfo = automatic_weather_risk_with_data(city_m)
            except: ws=automatic_weather_risk(city_m); winfo=None
            cs,cnn_class = predict_cnn_risk(image_path)
            ns = predict_text_risk(text_m)
        try:
            geo=Nominatim(user_agent="disaster-v2"); loc=geo.geocode(city_m)
            lat,lon=(loc.latitude,loc.longitude) if loc else (None,None)
        except: lat,lon=None,None
        show_results(ws,cs,ns,cnn_class,lat,lon,city_m,winfo)


# ── AUTO SCAN ─────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="slabel">⚡ Automatic Risk Scan</div>',unsafe_allow_html=True)
    l2,r2 = st.columns(2,gap="large")
    with l2:
        auto_city=st.text_input("📍 Target Location",placeholder="e.g. Kerala, Odisha",key="auto_city")
    with r2:
        st.markdown("""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:0.9rem 1.1rem;margin-top:1.6rem;font-size:0.82rem;color:#475569;line-height:1.6">
          📡 Auto mode fetches <b style="color:#2563eb">live weather</b> and
          <b style="color:#2563eb">real-time news</b> for the location.
          Satellite CNN mirrors weather signal in this mode.
        </div>""",unsafe_allow_html=True)

    st.markdown("")
    if st.button("⚡ Launch Auto Scan",key="auto_btn"):
        if not auto_city:
            st.warning("Please enter a location."); st.stop()
        with st.spinner("Scanning live data streams..."):
            try:
                geo=Nominatim(user_agent="disaster-v2"); loc=geo.geocode(auto_city)
                if loc is None:
                    st.error(f"Location '{auto_city}' not found."); st.stop()
                lat,lon=loc.latitude,loc.longitude
                try:
                    nr=requests.get(f"https://news.google.com/rss/search?q={auto_city}+weather+today&hl=en&gl=IN&ceid=IN:en",timeout=6)
                    hl=re.findall(r"<title>(.*?)</title>",nr.text)
                    auto_text=" ".join(hl[:10]) if len(hl)>2 else f"weather {auto_city}"
                except: auto_text=f"weather {auto_city}"
                try:    ws,winfo=automatic_weather_risk_with_data(auto_city)
                except: ws=automatic_weather_risk(auto_city); winfo=None
                cs=ws; cnn_class="N/A (Auto Mode)"; ns=predict_text_risk(auto_text)
                st.session_state.auto_results=dict(ws=ws,cs=cs,ns=ns,cnn_class=cnn_class,
                    lat=lat,lon=lon,city=auto_city,winfo=winfo)
            except Exception as e:
                st.error(f"Scan failed: {e}"); st.stop()

    if st.session_state.auto_results:
        r=st.session_state.auto_results
        show_results(r["ws"],r["cs"],r["ns"],r["cnn_class"],
                     r["lat"],r["lon"],r["city"],r.get("winfo"))


# ── MISSION LOG ───────────────────────────────────────────────
with tab3:
    st.markdown('<div class="slabel">📋 Mission Log — Session History</div>',unsafe_allow_html=True)
    hist=st.session_state.analysis_history

    if not hist:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#cbd5e1;
                    border:1.5px dashed #e2e8f0;border-radius:14px;background:#fafafa">
          <div style="font-size:2.5rem;margin-bottom:0.8rem">📡</div>
          <div style="font-size:0.88rem;color:#94a3b8">No analyses logged yet.<br>
          Run a scan to populate the mission log.</div>
        </div>""",unsafe_allow_html=True)
    else:
        scores=[h["score"] for h in hist]
        hc=sum(1 for h in hist if h["level"]=="High")
        mc=sum(1 for h in hist if h["level"]=="Medium")
        lc=sum(1 for h in hist if h["level"]=="Low")

        s1,s2,s3,s4=st.columns(4)
        for col,val,lbl,cls in [(s1,str(len(scores)),"Total Scans","blue"),
                                 (s2,str(hc),"High Risk","danger"),
                                 (s3,str(mc),"Medium Risk","warn"),
                                 (s4,str(lc),"Low Risk","safe")]:
            with col:
                st.markdown(f'<div class="mtile {cls}"><div class="mlabel">{lbl}</div>'
                            f'<div class="mval {cls}">{val}</div></div>',unsafe_allow_html=True)

        if len(scores)>1:
            st.markdown('<div class="slabel">📈 Score Trend</div>',unsafe_allow_html=True)
            cities=[h["city"] for h in hist]
            fig=go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1,len(scores)+1)),y=scores,
                mode="lines+markers+text",
                text=cities,textposition="top center",textfont=dict(size=9,color="#94a3b8"),
                line=dict(color="#2563eb",width=2.5),
                marker=dict(size=10,color=[shex(s) for s in scores],
                            line=dict(color="#ffffff",width=2)),
                fill="tozeroy",fillcolor="rgba(37,99,235,0.05)",
            ))
            fig.add_hline(y=0.66,line_dash="dash",line_color="rgba(239,68,68,0.5)",
                          annotation_text="High",annotation_font_color="#ef4444",annotation_font_size=10)
            fig.add_hline(y=0.33,line_dash="dash",line_color="rgba(245,158,11,0.5)",
                          annotation_text="Medium",annotation_font_color="#f59e0b",annotation_font_size=10)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font_color="#64748b",height=260,margin=dict(l=10,r=10,t=30,b=10),
                yaxis=dict(range=[0,1.2],gridcolor="#f1f5f9",zeroline=False,
                           title="Risk Score",tickfont=dict(size=10)),
                xaxis=dict(gridcolor="rgba(0,0,0,0)",title="Scan #",tickfont=dict(size=10)),
            )
            st.markdown('<div class="card" style="padding:1rem">',unsafe_allow_html=True)
            st.plotly_chart(fig,use_container_width=True)
            st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="slabel">🗂️ All Records</div>',unsafe_allow_html=True)
        for i,h in enumerate(reversed(hist)):
            idx=len(hist)-i
            st.markdown(f"""
            <div class="hrow">
              <span style="color:#cbd5e1;font-family:JetBrains Mono,monospace;font-size:0.72rem">#{idx:02d}</span>
              <span style="color:#94a3b8;font-size:0.75rem">{h['time']}</span>
              <span style="color:#475569;font-weight:600">📍 {h['city']}</span>
              <span style="color:{shex(h['score'])};font-family:JetBrains Mono,monospace;font-weight:700">{h['score']:.3f}</span>
              <span class="hbadge {h['level']}">{h['level'].upper()}</span>
              <span style="color:#94a3b8;font-size:0.78rem">{h['type']}</span>
            </div>""",unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Clear Mission Log",key="clear_hist"):
            st.session_state.analysis_history=[]; st.rerun()

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
</div>
<div style="text-align:center;padding:1.5rem;color:#cbd5e1;font-size:0.72rem;
            border-top:1px solid #e2e8f0;margin-top:2rem;letter-spacing:0.06em">
  DISASTERWATCH AI v2.0 &nbsp;·&nbsp; CNN + WEATHER + NLP FUSION &nbsp;·&nbsp; MULTI-MODAL EARLY WARNING
</div>
""",unsafe_allow_html=True)
