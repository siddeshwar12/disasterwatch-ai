import numpy as np


# -------------------------------------------------
# CONFIDENCE SCORE
# -------------------------------------------------

def confidence_score(weather, cnn, nlp):

    scores = np.array([weather, cnn, nlp])

    std = np.std(scores)

    confidence = 1 - std

    return max(0, min(confidence, 1))


# -------------------------------------------------
# DYNAMIC WEIGHT CALCULATION
# -------------------------------------------------

def dynamic_weights(weather_score, cnn_score, nlp_score):

    scores = np.array([weather_score, cnn_score, nlp_score])

    # higher score → stronger signal
    total = np.sum(scores)

    if total == 0:
        return 0.33, 0.33, 0.33

    w_weather = weather_score / total
    w_cnn = cnn_score / total
    w_nlp = nlp_score / total

    return w_weather, w_cnn, w_nlp


# -------------------------------------------------
# FUSION ENGINE
# -------------------------------------------------

def fuse_risk(weather_score, cnn_score, nlp_score):
    # Weather is the primary reliable signal.
    # In auto mode CNN mirrors weather, NLP is from neutral news search.
    W_WEATHER = 0.55
    W_CNN     = 0.25
    W_NLP     = 0.20

    final_risk = (
        W_WEATHER * weather_score +
        W_CNN     * cnn_score +
        W_NLP     * nlp_score
    )

    # weighted contributions for display
    w_weather = W_WEATHER * weather_score
    w_cnn     = W_CNN     * cnn_score
    w_nlp     = W_NLP     * nlp_score

    return float(min(1.0, final_risk)), w_weather, w_cnn, w_nlp


# -------------------------------------------------
# RISK LEVEL
# -------------------------------------------------

def risk_level(score):

    if score < 0.33:
        return "Low"

    elif score < 0.66:
        return "Medium"

    else:
        return "High"


# -------------------------------------------------
# EXPLANATION ENGINE
# -------------------------------------------------

def generate_explanation(weather, cnn, nlp):

    contributions = {
        "Weather Model": weather,
        "Satellite Model": cnn,
        "Social NLP Model": nlp
    }

    dominant = max(contributions, key=contributions.get)

    explanation = f"Risk is primarily influenced by {dominant}. "

    if dominant == "Weather Model":
        explanation += "Weather patterns indicate elevated disaster probability."

    elif dominant == "Satellite Model":
        explanation += "Satellite imagery suggests spatial disaster indicators."

    else:
        explanation += "Social media signals indicate emerging crisis activity."

    return explanation
# -------------------------------------------------
# RISK BREAKDOWN
# -------------------------------------------------

def disaster_breakdown(weather, cnn, nlp):

    # flood influenced more by weather
    flood = (0.5 * weather) + (0.3 * cnn) + (0.2 * nlp)

    # cyclone influenced by weather + satellite
    cyclone = (0.4 * weather) + (0.4 * cnn) + (0.2 * nlp)

    # landslide influenced by rainfall + terrain
   
    return {
        "Flood": flood,
        "Cyclone": cyclone,
        
    }
def predict_disaster_type(weather, cnn, nlp):

    breakdown = disaster_breakdown(weather, cnn, nlp)

    disaster = max(breakdown, key=breakdown.get)

    score = breakdown[disaster]

    # threshold for real disaster
    if score < 0.45:
        return "Normal"

    return disaster