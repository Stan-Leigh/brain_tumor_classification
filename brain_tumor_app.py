"""
Brain Tumor MRI Classification Streamlit App

Place these files in the same project folder before running:
- brain_tumor_app.py
- models/brain_tumor_mri_model.keras
- models/class_names.json

Run:
    streamlit run brain_tumor_app.py
"""

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

IMG_SIZE = (224, 224)
MODEL_PATH = Path("models/brain_tumor_mri_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")
SAMPLE_IMAGE_PATH = Path("sample_image/sample_mri.jpg")

DEFAULT_CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]

DISPLAY_NAMES = {
    "glioma_tumor": "Glioma tumor",
    "meningioma_tumor": "Meningioma tumor",
    "no_tumor": "No tumor",
    "pituitary_tumor": "Pituitary tumor",
}


# -----------------------------
# Utility functions
# -----------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model_and_classes():
    """Load the saved Keras model and class names once per Streamlit session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the notebook first and copy "
            "brain_tumor_mri_model.keras into the models/ folder."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = DEFAULT_CLASS_NAMES

    return model, class_names


def crop_brain_region(rgb_image: np.ndarray) -> np.ndarray:
    """Crop the visible brain region from an RGB MRI image using contours.

    If contour detection fails, the original image is returned. This keeps the
    app robust for unusual uploads.
    """
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb_image

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return rgb_image

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = rgb_image[y : y + h, x : x + w]

    if cropped.size == 0:
        return rgb_image

    return cropped


def preprocess_image(image: Image.Image, crop: bool = True) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image into the model input format."""
    image = image.convert("RGB")
    rgb_array = np.array(image)

    if crop:
        rgb_array = crop_brain_region(rgb_array)

    resized = cv2.resize(rgb_array, IMG_SIZE, interpolation=cv2.INTER_AREA)
    model_input = np.expand_dims(resized.astype("float32"), axis=0)
    preview = Image.fromarray(resized.astype("uint8"))
    return model_input, preview


def format_label(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


# -----------------------------
# User interface
# -----------------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI image and the model will classify it as glioma tumor, "
    "meningioma tumor, pituitary tumor, or no tumor."
)

st.warning(
    "Educational demo only. This app is not a medical diagnostic tool and should "
    "not be used for clinical decisions."
)

with st.sidebar:
    st.header("Settings")
    crop_image = st.checkbox("Crop visible brain region before prediction", value=True)
    show_probabilities = st.checkbox("Show class probabilities", value=True)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

use_sample_image = st.checkbox("Use default sample MRI image", value=False)

if uploaded_file is None and not use_sample_image:
    st.info("Upload a JPG, JPEG, or PNG brain MRI image, or tick the checkbox to use the default sample image.")
    st.stop()

if use_sample_image and not SAMPLE_IMAGE_PATH.exists():
    st.error(
        f"Sample image not found at {SAMPLE_IMAGE_PATH}. "
        "Create a sample_images folder and add sample_mri.jpg inside it."
    )
    st.stop()

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(str(e))
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_source = "Uploaded image"
else:
    image = Image.open(SAMPLE_IMAGE_PATH)
    image_source = "Default sample image"

model_input, processed_preview = preprocess_image(image, crop=crop_image)

col1, col2 = st.columns(2)
with col1:
    st.subheader(image_source)
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Model input")
    st.image(processed_preview, use_container_width=True)

if st.button("Classify MRI", type="primary"):
    with st.spinner("Running prediction..."):
        probabilities = model.predict(model_input, verbose=0)[0]

    top_index = int(np.argmax(probabilities))
    predicted_class = class_names[top_index]
    confidence = float(probabilities[top_index])

    st.subheader("Prediction")
    st.metric("Predicted class", format_label(predicted_class), f"{confidence:.1%} confidence")

    if predicted_class == "no_tumor":
        st.success("The model predicts: No tumor")
    else:
        st.error(f"The model predicts: {format_label(predicted_class)}")

    if show_probabilities:
        probs_df = pd.DataFrame(
            {
                "Class": [format_label(name) for name in class_names],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        st.subheader("Class probabilities")
        st.dataframe(
            probs_df.assign(Probability=lambda df: df["Probability"].map(lambda x: f"{x:.2%}")),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(probs_df.set_index("Class"))

st.caption(
    "Tip: for best results, upload a clear MRI image similar to the images used during training."
)
