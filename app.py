
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fabric_classifier.h5")

with open("labels.txt") as f:
    labels = f.read().splitlines()

st.set_page_config(page_title="PatternSense")

st.title("👗 PatternSense")
st.subheader("Fabrics Classification using Deep Learning")

uploaded = st.file_uploader(
    "Upload a Image of fabric",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = np.max(prediction)*100

    st.success(f"Predicted Pattern: {labels[index]}")
    st.write(f"Confidence: {confidence:.2f}%")
