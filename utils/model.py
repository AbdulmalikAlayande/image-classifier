from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "image_classifier.keras"
DECISION_THRESHOLD = 0.5


def load_classifier() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model: tf.keras.Model, image_tensor: np.ndarray) -> tuple[str, float]:
    """Run inference and return (label, confidence in that label).

    Sigmoid output is P(real). If > 0.5 → 'Real', else 'AI-Generated'.
    Confidence is the probability assigned to the predicted class.
    """
    raw = float(model.predict(image_tensor, verbose=0)[0][0])
    if raw > DECISION_THRESHOLD:
        return "Real", raw
    return "AI-Generated", 1.0 - raw
