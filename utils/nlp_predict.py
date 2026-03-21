"""
Lightweight keyword-based NLP risk scorer.
No model download required — works instantly on Railway.
"""

# Weighted disaster keywords
_HIGH_RISK = [
    "flood", "flooding", "cyclone", "hurricane", "tornado", "tsunami",
    "earthquake", "landslide", "dam break", "dam burst", "overflow",
    "disaster", "catastrophe", "emergency", "evacuation", "rescue",
    "heavy rainfall", "extreme rain", "storm surge", "waterlogging",
    "inundation", "submerged", "washed away", "collapsed"
]

_MEDIUM_RISK = [
    "storm", "heavy rain", "strong wind", "warning", "alert",
    "rainfall", "thunder", "lightning", "hail", "downpour",
    "rising water", "river level", "high tide", "monsoon",
    "damage", "injured", "casualties", "affected"
]

_LOW_RISK = [
    "rain", "wind", "cloudy", "wet", "drizzle", "overcast",
    "forecast", "weather", "humidity", "temperature"
]


def predict_text_risk(text: str) -> float:
    """
    Returns a risk score between 0.0 and 1.0 based on keyword matching.
    """
    text_lower = text.lower()

    score = 0.0

    for kw in _HIGH_RISK:
        if kw in text_lower:
            score += 0.25

    for kw in _MEDIUM_RISK:
        if kw in text_lower:
            score += 0.10

    for kw in _LOW_RISK:
        if kw in text_lower:
            score += 0.03

    return float(min(1.0, score))
