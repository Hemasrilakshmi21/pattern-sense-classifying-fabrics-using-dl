import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fabric_classifier.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.set_page_config(
    page_title="PatternSense",
    page_icon="👗"
)

st.title("👗 PatternSense")
st.subheader("Fabric Pattern Classification using Deep Learning")

uploaded = st.file_uploader(
    "Upload an image of fabric",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fabric Image",
        use_container_width=True
    )

    # Resize image to model input size
    img = image.resize((224, 224))

    # Convert image to array
    img = np.array(img) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Pattern: {labels[index]}")
    st.write(f"Confidence: {confidence:.2f}%")
