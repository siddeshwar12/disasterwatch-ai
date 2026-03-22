"""
Run this once locally to convert the keras model to ONNX:
    python convert_to_onnx.py
"""
import tensorflow as tf
import tf2onnx
import numpy as np

model = tf.keras.models.load_model("models/final_cnn_2class_model.keras", compile=False)
input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

with open("models/final_cnn_2class_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Done — models/final_cnn_2class_model.onnx created")
