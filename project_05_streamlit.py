import streamlit as st

# ✅ Page config FIRST
st.set_page_config(page_title="SolarGuard - Defect Detection", layout="centered")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# ========== Set Background Image ==========
def set_bg_from_local(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 12px;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Set your background image
set_bg_from_local("D:/Project_Files/Project_05_solar_panel/solar-panel-bg.webp")

# ========== Full Block Wrapper ==========
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ========== Title and Instructions ==========
st.title("☀️ Solar Panel Defect Detection")
st.markdown("Upload an image of a solar panel to detect any possible defect.")
st.markdown("Choose a solar panel image...")

# ========== Load Model ==========
@st.cache_resource
def load_defect_model():
    model = load_model("D:/Project_Files/Project_05_solar_panel/solar_defect_cnn_model (1).h5")
    return model

model = load_defect_model()

# ========== Class Labels ==========
class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# ========== Upload Image ==========
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Result
    st.markdown(f"### 🔍 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")

# ✅ Close styled block
st.markdown('</div>', unsafe_allow_html=True)
