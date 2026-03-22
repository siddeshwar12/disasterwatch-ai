"""
Keyword-based NLP risk scorer with dampened scoring to avoid false positives.
"""

_HIGH_RISK = [
    "flood", "flooding", "cyclone", "hurricane", "tornado", "tsunami",
    "earthquake", "landslide", "dam break", "dam burst", "overflow",
    "catastrophe", "evacuation", "rescue", "storm surge",
    "inundation", "submerged", "washed away", "collapsed"
]

_MEDIUM_RISK = [
    "disaster", "emergency", "heavy rainfall", "extreme rain",
    "waterlogging", "storm", "heavy rain", "strong wind",
    "warning", "alert", "rising water", "high tide",
    "damage", "casualties", "affected"
]

_LOW_RISK = [
    "rainfall", "thunder", "lightning", "downpour", "monsoon",
    "rain", "wind", "cloudy", "wet", "drizzle", "forecast",
    "weather", "humidity", "temperature", "river level"
]


def predict_text_risk(text: str) -> float:
    """
    Returns a risk score between 0.0 and 1.0.
    Uses unique keyword matching (each keyword counted once)
    with logarithmic dampening to prevent runaway scores.
    """
    text_lower = text.lower()

    high_hits  = sum(1 for kw in _HIGH_RISK   if kw in text_lower)
    med_hits   = sum(1 for kw in _MEDIUM_RISK  if kw in text_lower)
    low_hits   = sum(1 for kw in _LOW_RISK     if kw in text_lower)

    # Dampened scoring: each additional hit contributes less
    import math
    score = 0.0
    if high_hits > 0:
        score += 0.20 * math.log1p(high_hits)   # log(1+n) dampening
    if med_hits > 0:
        score += 0.10 * math.log1p(med_hits)
    if low_hits > 0:
        score += 0.04 * math.log1p(low_hits)

    return float(min(1.0, score))
