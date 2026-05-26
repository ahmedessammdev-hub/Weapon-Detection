# app.py
# =========================================================
# Weapon Detection System - Streamlit + YOLO
# Professional UI/UX
# Supports:
# - Image Detection
# - Video Detection
# - Live Camera Detection
# - Alert System
# =========================================================

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import time
from datetime import datetime
import pandas as pd
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0b0f19;
    color: white;
}

/* Main Cards */
.custom-card {
    background: linear-gradient(145deg, #111827, #1f2937);
    padding: 24px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

/* Header */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0;
}

.sub-title {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-top: -10px;
}

/* Detection Box */
.alert-box {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239,68,68,0.4);
    padding: 18px;
    border-radius: 16px;
    color: #fca5a5;
    font-weight: 600;
}

.safe-box {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    padding: 18px;
    border-radius: 16px;
    color: #86efac;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Metric Cards */
.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # your trained model
    return model

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🛡️ Weapon Detection")

source = st.sidebar.selectbox(
    "Select Input Source",
    [
        "Image Upload",
        "Video Upload",
        "Live Camera"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4,
    0.05
)

enable_alert = st.sidebar.toggle("Enable Alert System", True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Supported Classes
- 🔫 Gun
- 🔪 Knife
- ⚠️ Other Weapons
""")

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="custom-card">
    <h1 class="main-title">Weapon Detection System</h1>
    <p class="sub-title">
        Real-time AI surveillance powered by YOLO object detection.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================================================
# ALERT FUNCTION
# =========================================================

def trigger_alert(label):

    if enable_alert:

        st.markdown(f"""
        <div class="alert-box">
            🚨 ALERT: {label.upper()} DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.warning(f"Weapon Detected: {label}")

# =========================================================
# DRAW DETECTIONS
# =========================================================

def process_frame(frame):

    results = model(frame, conf=confidence)

    detected_labels = []

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detected_labels.append(label)

    return annotated_frame, detected_labels

# =========================================================
# IMAGE SECTION
# =========================================================

if source == "Image Upload":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Running Detection..."):

            annotated_image, labels = process_frame(image_np)

        with col2:
            st.markdown("### Detection Result")
            st.image(annotated_image, use_container_width=True)

        if len(labels) > 0:

            unique_labels = list(set(labels))

            for label in unique_labels:
                trigger_alert(label)

        else:

            st.markdown("""
            <div class="safe-box">
                ✅ No weapon detected
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# VIDEO SECTION
# =========================================================

elif source == "Video Upload":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:

        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video.name)

        stframe = st.empty()

        detection_log = []

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            annotated_frame, labels = process_frame(frame)

            annotated_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            stframe.image(
                annotated_frame,
                channels="RGB",
                use_container_width=True
            )

            if len(labels) > 0:

                for label in labels:

                    trigger_alert(label)

                    detection_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Weapon": label
                    })

        cap.release()

        if len(detection_log) > 0:

            st.markdown("## Detection Logs")

            df = pd.DataFrame(detection_log)

            st.dataframe(df, use_container_width=True)

# =========================================================
# LIVE CAMERA SECTION
# =========================================================

elif source == "Live Camera":

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    detection_log = []

    while run:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera Error")
            break

        annotated_frame, labels = process_frame(frame)

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        FRAME_WINDOW.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

        if len(labels) > 0:

            for label in labels:

                trigger_alert(label)

                detection_log.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Weapon": label
                })

        time.sleep(0.03)

    camera.release()

# =========================================================
# FOOTER
# =========================================================

st.write("")
st.markdown("---")

st.markdown("""
<center>
    <p style='color:gray'>
        AI Surveillance • YOLO Object Detection • Real-Time Monitoring
    </p>
</center>
""", unsafe_allow_html=True)