import numpy as np
from PIL import Image

TARGET_SIZE = (300, 300)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize to 300x300, convert to RGB, normalize to [0, 1], add batch dim."""
    img = pil_image.convert("RGB").resize(TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
