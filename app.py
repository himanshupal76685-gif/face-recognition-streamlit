import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection System", layout="centered")

st.title("😎 Face Detection System")
st.write("Streamlit Cloud compatible face detection app")
st.markdown('''**Design and Developed by: Himanshu pal**''')

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Session state for camera
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.camera_on = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_on = False

# Camera input
if st.session_state.camera_on:
    image = st.camera_input("Take a picture")

    if image is not None:
        img = Image.open(image)
        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(img_np, caption="Detected Face(s)", use_column_width=True)

        if len(faces) > 0:
            st.success(f"✅ Face Detected: {len(faces)}")
        else:
            st.warning("❌ No face detected")
            


