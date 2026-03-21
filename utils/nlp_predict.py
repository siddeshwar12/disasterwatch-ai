from transformers import pipeline

_classifier = None


def _load_model():
    global _classifier
    if _classifier is None:
        # lightweight pipeline — no torch needed, uses onnxruntime backend
        _classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            framework="pt" if _is_torch_available() else "tf"
        )


def _is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def predict_text_risk(text):
    _load_model()

    result = _classifier(text[:512], truncation=True)[0]

    label = result["label"].lower()
    score = result["score"]

    # map sentiment to risk
    if "negative" in label:
        base_score = score
    elif "neutral" in label:
        base_score = score * 0.3
    else:
        base_score = (1 - score) * 0.2

    disaster_keywords = [
        "flood", "flooding", "heavy rainfall", "cyclone",
        "storm", "overflow", "waterlogging", "dam break", "landslide"
    ]

    boost = sum(0.15 for word in disaster_keywords if word in text.lower())

    return float(min(1.0, base_score + boost))
