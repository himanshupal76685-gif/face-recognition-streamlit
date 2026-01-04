import streamlit as st
import face_recognition
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Recognition System")
st.title("Face Recognition System")

KNOWN_FACES_DIR = "known_faces"

# ---------------- Load known faces ----------------
known_face_encodings = []
known_face_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img = face_recognition.load_image_file(
            os.path.join(KNOWN_FACES_DIR, file)
        )
        enc = face_recognition.face_encodings(img)
        if enc:
            known_face_encodings.append(enc[0])
            known_face_names.append(os.path.splitext(file)[0])

st.success("Known faces loaded")

# ---------------- Camera buttons ----------------
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

col1, col2 = st.columns(2)

with col1:
    if st.button("📷 Start Camera"):
        st.session_state.camera_on = True

with col2:
    if st.button("🛑 Stop Camera"):
        st.session_state.camera_on = False

# ---------------- Camera Input ----------------
recognized_name = "None"

if st.session_state.camera_on:
    img_file = st.camera_input("Take a photo")

    if img_file:
        image = Image.open(img_file)
        img_np = np.array(image)

        face_locations = face_recognition.face_locations(img_np)
        face_encodings = face_recognition.face_encodings(
            img_np, face_locations
        )

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            if True in matches:
                recognized_name = known_face_names[matches.index(True)]
            else:
                recognized_name = "Unknown"

# ---------------- Result outside box ----------------
st.markdown("## 🧑 Recognized Person")
st.success(recognized_name)
