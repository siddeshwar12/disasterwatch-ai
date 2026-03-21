from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = TFAutoModelForSequenceClassification.from_pretrained(_MODEL_NAME, from_pt=True)


def predict_text_risk(text):
    _load_model()

    inputs = _tokenizer(
        text[:512],
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = _model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

    # label order: negative=0, neutral=1, positive=2
    negative = float(probs[0])

    disaster_keywords = [
        "flood", "flooding", "heavy rainfall", "cyclone",
        "storm", "overflow", "waterlogging", "dam break", "landslide"
    ]

    boost = sum(0.15 for word in disaster_keywords if word in text.lower())

    return float(min(1.0, negative + boost))
