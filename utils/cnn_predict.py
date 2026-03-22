import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load 2-class model (cyclone=0, flood=1)
_MODEL_PATH = os.path.join(_BASE, "models", "final_cnn_2class_model.keras")
model = load_model(_MODEL_PATH, compile=False)

# Class labels in training order
CLASS_LABELS = ["Cyclone", "Flood"]


def predict_cnn_risk(img_path: str):
    """
    Returns (risk_score, predicted_class).
    risk_score: 0.0–1.0  (both cyclone and flood are disaster classes,
                           so score = max probability among disaster classes)
    predicted_class: "Cyclone" or "Flood"
    """
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr, verbose=0)[0]   # [p_cyclone, p_flood]

    predicted_idx   = int(np.argmax(prediction))
    predicted_class = CLASS_LABELS[predicted_idx]
    risk_score      = float(np.max(prediction))     # confidence = risk

    return risk_score, predicted_class
