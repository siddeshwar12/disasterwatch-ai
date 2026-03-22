import os
import numpy as np
import onnxruntime as ort
from PIL import Image

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE, "models", "final_cnn_2class_model.onnx")

_session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name

CLASS_LABELS = ["Cyclone", "Flood"]


def predict_cnn_risk(img_path: str):
    """
    Returns (risk_score, predicted_class).
    risk_score: 0.0–1.0
    predicted_class: "Cyclone" or "Flood"
    """
    img = Image.open(img_path).resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    prediction = _session.run(None, {_input_name: arr})[0][0]  # [p_cyclone, p_flood]

    predicted_idx   = int(np.argmax(prediction))
    predicted_class = CLASS_LABELS[predicted_idx]
    risk_score      = float(np.max(prediction))

    return risk_score, predicted_class
