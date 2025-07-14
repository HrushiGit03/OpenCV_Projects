import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page Config
st.set_page_config(page_title="🚘 Number Plate Detection", layout="wide")

# 🌙 Dark Mode Toggle
dark_mode = st.sidebar.toggle("🌙 Enable Dark Mode", value=False)

# 🎨 Theme Colors
bg_color = "#121212" if dark_mode else "#f5faff"
fg_color = "#f1f5f9" if dark_mode else "#0f172a"
primary_color = "#38bdf8" if dark_mode else "#007BFF"
accent_color = "#22c55e" if dark_mode else "#10b981"
border_color = "#374151" if dark_mode else "#cbd5e1"

# 🖌️ CSS Styling
st.markdown(f"""
    <style>
        html, body {{
            background-color: {bg_color};
            color: {fg_color};
        }}
        .stApp {{
            background-color: {bg_color};
            color: {fg_color};
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            transition: 0.3s ease-in-out;
        }}
        .stSidebar {{
            background-color: {"#1e293b" if dark_mode else "#e0f2fe"};
        }}
        .css-1cpxqw2 {{
            padding: 2rem;
        }}
        .stFileUploader>label {{
            font-weight: bold;
            color: {primary_color};
        }}
        .stSelectbox>div>div {{
            background-color: {"#334155" if dark_mode else "#ffffff"};
            color: {fg_color};
        }}
    </style>
""", unsafe_allow_html=True)

# Load Haar Cascade
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# 🧠 App Title
st.markdown(f"<h2 style='color:{primary_color};'>🚘 Smart Number Plate Detection App</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{fg_color}; font-size: 16px;'>Using OpenCV Haar Cascades to detect vehicle plates in real time from images or videos.</p>", unsafe_allow_html=True)

# 📌 Sidebar Detection Mode Selector
st.sidebar.markdown("## 🛠️ Choose Detection Mode")
option = st.sidebar.selectbox("🔍 Detection Type:", ("📷 Image Detection", "🎥 Video Detection"))

# ---------- 📷 IMAGE DETECTION ----------
if option == "📷 Image Detection":
    st.markdown(f"<h3 style='color:{accent_color};'>📸 Upload a Vehicle Image</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload JPG / PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in plates:
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 3)

        st.image(img_np, caption="🔍 Detected Number Plate(s)", use_container_width=True)
        st.success(f"✅ Detected Plates: {len(plates)}")

# ---------- 🎥 VIDEO DETECTION ----------
elif option == "🎥 Video Detection":
    st.markdown(f"<h3 style='color:{accent_color};'>🎬 Upload a Vehicle Video</h3>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        st.info("⏳ Processing video... detecting plates frame by frame.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        tfile.close()
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)

        st.success("✅ Video processing completed successfully!")
    else:
        st.warning("📤 Please upload a video to begin detection.")
