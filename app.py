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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Base Dark Theme & Background */
.stApp {
    background: radial-gradient(circle at top right, #0d1323 0%, #060913 100%);
    color: #e2e8f0;
}

/* Glassmorphism Cards */
.custom-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.custom-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}

/* Headers */
.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(to right, #ffffff, #9ca3af);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.sub-title {
    color: #94a3b8;
    font-size: 1.15rem;
    font-weight: 400;
    margin-top: 0;
}

/* Alerts */
@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.alert-box {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.3));
    border: 1px solid rgba(239, 68, 68, 0.5);
    border-left: 5px solid #ef4444;
    padding: 20px;
    border-radius: 12px;
    color: #fca5a5;
    font-weight: 700;
    font-size: 1.1rem;
    animation: pulse-red 2s infinite;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 15px;
}

.safe-box {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(21, 128, 61, 0.2));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-left: 5px solid #22c55e;
    padding: 18px;
    border-radius: 12px;
    color: #86efac;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.8) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Sidebar text */
.sidebar-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Metrics customization */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #fff !important;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem !important;
    color: #94a3b8 !important;
}
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 15px 20px;
    border-radius: 16px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 54px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5);
    border-color: rgba(255, 255, 255, 0.3);
}

.stButton > button:active {
    transform: translateY(1px);
}

/* Empty State / Upload Container */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading YOLO Weights...")
def load_model():
    try:
        model = YOLO("best.pt")
    except:
        model = YOLO("yolov8n.pt") 
    return model

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("<div class='sidebar-title'>🛡️ WatchTower AI</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size: 0.9rem;'>Advanced Threat Detection</p>", unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Control Panel")
    
    source = st.selectbox(
        "Select Input Source",
        ["Image Upload", "Video Upload", "Live Camera"],
        index=0
    )
    
    confidence = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.45, 0.05,
        help="Minimum confidence score to consider a detection valid."
    )
    
    st.markdown("### 🔔 Notifications")
    enable_alert = st.toggle("Enable Alert System", True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ About System", expanded=True):
        st.markdown("""
        **Supported Classes:**
        - 🔫 Handgun / Rifle
        - 🔪 Knife / Blade
        - ⚠️ Miscellaneous Weapons
        
        *Powered by YOLOv8 Architecture*
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 20px;'>
        Created by <a href='https://linkedin.com/' target='_blank' style='color:#3b82f6; text-decoration:none;'>Ahmed Essam</a><br>
        v2.0 Professional Edition
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# HEADER & METRICS
# =========================================================

st.markdown("""
<div class="custom-card">
    <h1 class="main-title">Real-Time Threat Detection</h1>
    <p class="sub-title">
        Next-generation AI surveillance powered by YOLO deep learning models.
    </p>
</div>
""", unsafe_allow_html=True)

# Top Metrics Row
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("System Status", "🟢 Online", delta="Optimized")
with m2:
    st.metric("Active Source", source)
with m3:
    st.metric("Confidence Filter", f"{int(confidence*100)}%")

st.write("")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}

def trigger_alert(label):
    if enable_alert:
        current_time = time.time()
        last_time = st.session_state.last_alert_time.get(label, 0)
        
        if current_time - last_time > 3:
            st.toast(f"🚨 ALERT: {label.upper()} DETECTED!", icon="⚠️")
            st.session_state.last_alert_time[label] = current_time

        st.markdown(f"""
        <div class="alert-box">
            <span style='font-size: 1.5rem;'>🚨</span>
            <div>
                <div style='font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #ef4444;'>Threat Detected</div>
                <div style='font-size: 1.2rem;'>{label.upper()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def process_frame(frame):
    results = model(frame, conf=confidence)
    detected_labels = []
    
    annotated_frame = results[0].plot(line_width=2)
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_labels.append(label)

    return annotated_frame, detected_labels

# =========================================================
# MAIN CONTENT AREA
# =========================================================

st.markdown("### 📺 Monitor Feed")

if source == "Image Upload":
    
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        with st.spinner("Analyzing image..."):
            annotated_image, labels = process_frame(image_np)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Input", width="stretch")
            
        with col2:
            st.image(annotated_image, caption="AI Detection Result", width="stretch")
            
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")
        
        if len(labels) > 0:
            unique_labels = list(set(labels))
            for label in unique_labels:
                trigger_alert(label)
        else:
            st.markdown("""
            <div class="safe-box">
                <span style='font-size: 1.5rem;'>✅</span>
                Area Secure. No weapons detected in the frame.
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="empty-state">
            <h2 style='color: #475569;'>No Image Uploaded</h2>
            <p>Please upload an image file using the uploader above to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
elif source == "Video Upload":
    
    uploaded_video = st.file_uploader("Upload video footage for analysis", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(temp_video.name)
        
        col_feed, col_logs = st.columns([2, 1])
        
        with col_feed:
            stframe = st.empty()
            
        with col_logs:
            st.markdown("### 🔴 Active Logs")
            log_container = st.empty()
            
        detection_log = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame, labels = process_frame(frame)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(annotated_frame, channels="RGB", width="stretch")
            
            if len(labels) > 0:
                for label in labels:
                    detection_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Threat": label.upper()
                    })
                    
                df = pd.DataFrame(detection_log[-10:])
                log_container.dataframe(df, use_container_width=True, hide_index=True)
                
        cap.release()
        
        if len(detection_log) > 0:
            st.markdown("### 📊 Final Incident Report")
            df_full = pd.DataFrame(detection_log)
            st.dataframe(df_full, use_container_width=True, hide_index=True)
            
    else:
        st.markdown("""
        <div class="empty-state">
            <h2 style='color: #475569;'>No Video Uploaded</h2>
            <p>Please upload a video file to run continuous analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
elif source == "Live Camera":
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        run = st.toggle("🎥 Start Live Camera Feed", value=False)
        FRAME_WINDOW = st.empty()
        
        if not run:
            FRAME_WINDOW.markdown("""
            <div class="empty-state">
                <h2 style='color: #475569;'>Camera Offline</h2>
                <p>Toggle the switch above to activate the live camera feed.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown("### 🔴 Live Alerts")
        alert_container = st.empty()
    
    if run:
        camera = cv2.VideoCapture(0)
        detection_log = []
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera. Please check your permissions.")
                break
                
            annotated_frame, labels = process_frame(frame)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            FRAME_WINDOW.image(annotated_frame, channels="RGB", width="stretch")
            
            if len(labels) > 0:
                with alert_container:
                    unique_labels = list(set(labels))
                    for label in unique_labels:
                        trigger_alert(label)
                        detection_log.append({
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Threat": label.upper()
                        })
            else:
                with alert_container:
                    st.markdown("""
                    <div class="safe-box" style="padding: 10px; font-size: 0.9rem;">
                        ✅ SECURE
                    </div>
                    """, unsafe_allow_html=True)
                    
            time.sleep(0.03)
            
        camera.release()

# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p style='margin-bottom: 5px; font-weight: 600;'>WatchTower AI Surveillance System</p>
    <p style='font-size: 0.85rem; margin-top: 0;'>
        Built with Streamlit & YOLO • Real-Time Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)