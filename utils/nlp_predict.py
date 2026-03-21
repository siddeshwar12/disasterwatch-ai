import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()


def predict_text_risk(text):
    _load_model()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    negative = probabilities[0][0].item()

    disaster_keywords = [
        "flood", "flooding", "heavy rainfall", "cyclone",
        "storm", "overflow", "waterlogging", "dam break", "landslide"
    ]

    boost = sum(0.15 for word in disaster_keywords if word in text.lower())

    return float(min(1.0, negative + boost))
