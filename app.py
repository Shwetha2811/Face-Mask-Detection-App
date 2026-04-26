import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ================= LOAD MODEL =================
model = load_model("face_mask_model.keras")

# ================= UI =================
st.title("😷 Face Mask Detection App")
st.write("Upload an image to check if the person is wearing a mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# ================= PREDICTION =================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("❌ Without Mask")
    else:
        st.error("✅ With Mask")