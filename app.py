from pathlib import Path

import keras
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="PatternSense")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fabric_classifier.h5"
LABELS_PATH = BASE_DIR / "labels.txt"


class CompatibleDense(keras.layers.Dense):
    """Loads models saved by newer Keras versions in Keras 3.12."""

    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


@st.cache_resource(show_spinner="Loading the classifier...")
def load_model():
    return keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"Dense": CompatibleDense},
    )


@st.cache_data
def load_labels():
    return LABELS_PATH.read_text(encoding="utf-8").splitlines()


st.title("PatternSense")
st.subheader("Fabric classification using deep learning")

try:
    model = load_model()
    labels = load_labels()
except Exception as error:
    st.error(f"Unable to load the classifier: {error}")
    st.stop()

uploaded = st.file_uploader(
    "Upload an image of fabric",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    # Display the original upload; only the prediction copy is resized below.
    st.image(image, caption="Uploaded image")

    image_array = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
    prediction = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    index = int(np.argmax(prediction))
    confidence = float(prediction[index]) * 100

    st.success(f"Predicted pattern: {labels[index]}")
    st.write(f"Confidence: {confidence:.2f}%")
