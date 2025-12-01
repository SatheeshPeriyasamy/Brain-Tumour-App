import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Image size used by the model
image_size = 299

# Label order based on your model training
labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]

# Lazy-loaded model (loaded only once)
model = None

def get_model():
    global model
    if model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.h5")
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
    return model


def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    img = cv2.resize(img, (image_size, image_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # shape (1, 299, 299, 3)
    return img


def predict_tumor_class(_, image_path, labels):
    mdl = get_model()  # Loads model once
    image = preprocess_image(image_path, image_size)
    prediction = mdl.predict(image)
    predicted_class = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)
    return predicted_class, confidence
