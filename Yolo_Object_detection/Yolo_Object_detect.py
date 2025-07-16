import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import time
import os
import base64

# --- Helper Functions for UI Elements ---
def get_base64_image(image_path):
    """Encodes an image to base64 for embedding in HTML/CSS."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# 🌐 Page Config
st.set_page_config(page_title="🔍 YOLOv11 Detection Suite", layout="wide", page_icon="🧠")

# --- Asset Verification ---
# Check for model file and stop if it's missing.
if not os.path.exists("yolo11n.pt"):
    st.error("Error: 'yolo11n.pt' model file not found. Please place the YOLOv11 model file in the same directory as the script.")
    st.stop()

# Check for alert sound and prepare a flag.
ALERT_SOUND_PATH = "alert.mp3"
alert_sound_exists = os.path.exists(ALERT_SOUND_PATH)


# 🌙 Dark Mode Toggle & Logos
st.sidebar.header(":gear: Preferences")
dark_mode = st.sidebar.toggle("Enable Dark Mode :new_moon_with_face:", value=st.session_state.get('dark_mode', False))
st.session_state['dark_mode'] = dark_mode # Persist dark mode state

light_logo_url = "https://img.icons8.com/color/96/000000/artificial-intelligence.png"
dark_logo_url = "https://img.icons8.com/fluency/96/artificial-intelligence.png"
about_image_url = "https://images.unsplash.com/photo-1596558450268-ecc039de4455?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80"

# 🎨 Styling - Dynamic based on dark_mode
bg_color = "#0e1117" if dark_mode else "#ffffff"
font_color = "#fafafa" if dark_mode else "#1a1a1a"
accent_color = "#4CAF50"
secondary_accent = "#FF9800"

st.markdown(
    f"""
    <style>
    body, .stApp {{ background-color: {bg_color}; color: {font_color}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; }}
    [data-testid="stSidebarContent"] {{ background-color: {bg_color}; color: {font_color}; }}
    .stButton>button {{ background-color: {accent_color}; color: white; border-radius: 0.5rem; border: none; padding: 0.6rem 1.2rem; font-weight: bold; transition: background-color 0.3s ease; }}
    .stButton>button:hover {{ background-color: #45a049; }}
    .stDownloadButton>button {{ background-color: {secondary_accent}; color: white; border-radius: 0.5rem; border: none; padding: 0.6rem 1.2rem; font-weight: bold; transition: background-color 0.3s ease; }}
    .stDownloadButton>button:hover {{ background-color: #e68a00; }}
    h1, h2, h3, h4, h5, h6 {{ color: {accent_color}; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image(dark_logo_url if dark_mode else light_logo_url, caption="YOLOv11 Logo", width=120)



# 🎹 Keyboard Shortcut Info
st.sidebar.markdown("---")
with st.sidebar.expander(":keyboard: **Keyboard Shortcuts**"):
    st.markdown("""
    - Press `1` for :frame_with_picture: **Image mode**
    - Press `2` for :movie_camera: **Video mode**
    - Press `3` for :camera: **Webcam mode**

    *(Note: For key presses to trigger, ensure your browser window is focused and active. You might need to click on the Streamlit app area first.)*
    """)

# 🧠 Header
st.markdown(f"<h1 style='color:{accent_color}; text-align:center;'>🧠 YOLOv11 Full Detection Suite</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{font_color};'>Powered by Ultralytics for state-of-the-art object detection.</p>", unsafe_allow_html=True)
st.markdown("---")

# 🧠 Load Model
@st.cache_resource
def load_model():
    with st.spinner("Loading YOLOv11 Model... This may take a moment."):
        model = YOLO("yolo11n.pt")
    st.success("✅ YOLOv11 Model Loaded Successfully!")
    return model

model = load_model()

# 🌍 Sidebar Options
st.sidebar.header(":mag: Detection Settings")
option = st.sidebar.radio(
    "Choose Detection Mode",
    [":frame_with_picture: Image", ":movie_camera: Video", ":camera: Webcam"],
    index=0,
    key="choose-mode-radio"
)

# Populate selected classes
all_classes = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    ":bookmark_tabs: Filter Objects by Class",
    all_classes,
    default=all_classes,
    help="Select specific object classes to detect."
)

st.sidebar.markdown("---")
with st.sidebar.expander(":bar_chart: Model Information"):
    st.write(f"**Model Name:** `yolo11n.pt`")
    st.write(f"**Total Classes:** `{len(all_classes)}`")
    st.write(f"**Active Classes:** `{len(selected_classes)}`")
    st.markdown("---")
    st.write("This app uses a pre-trained YOLOv11 nano model.")


# 🔎 Utilities
def filter_results(results, selected_classes_indices):
    """Filters detection results to include only selected classes."""
    boxes = results[0].boxes
    # Ensure boxes is iterable and has a .cls attribute
    if boxes is not None and hasattr(boxes, 'cls'):
        filtered_indices = [i for i, cls_tensor in enumerate(boxes.cls) if int(cls_tensor) in selected_classes_indices]
        results[0].boxes = boxes[filtered_indices]
    else:
        results[0].boxes = [] # No detections or boxes attribute is missing
    return results

def count_objects(results, model_names):
    """Counts detected objects."""
    counts = {}
    # Ensure results[0].boxes exists and is iterable
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0]) # Assuming box.cls is a tensor and we take the first element
            class_name = model_names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts

selected_cls_indices = [k for k, v in model.names.items() if v in selected_classes]

# --- Main Content Area ---

# 📷 Image Mode
if option == ":frame_with_picture: Image":
    st.header(":framed_picture: Image Object Detection")
    st.markdown("Upload one or more images to detect objects.")

    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Results for `{uploaded_file.name}`")
            col1, col2 = st.columns(2)

            img = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(img)

            with col1:
                st.info("Original Image:")
                st.image(img_np, use_container_width=True)

            with col2:
                with st.spinner("🕵️‍♀️ Running detection..."):
                    results = model.predict(img_np, conf=0.3, verbose=False)
                    results = filter_results(results, selected_cls_indices)
                    annotated_img = results[0].plot() # BGR
                    object_counts = count_objects(results, model.names)

                st.success("✅ Detection Complete!")
                st.info("Annotated Image:")
                st.image(annotated_img, channels="BGR", use_container_width=True)

                st.markdown("#### 📦 Detected Object Counts")
                if object_counts:
                    for obj, count in object_counts.items():
                        st.markdown(f"- **{obj}**: {count}")
                else:
                    st.warning("No objects from the selected classes were detected.")

                # Download button
                img_to_download = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_to_download)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    pil_img.save(tmpfile.name)
                    with open(tmpfile.name, "rb") as f:
                        st.download_button(
                            ":inbox_tray: Download Annotated Image",
                            data=f.read(),
                            file_name=f"detected_{uploaded_file.name}",
                            mime="image/png"
                        )
                os.remove(tmpfile.name) # Clean up the temp file
            st.markdown("---")
    else:
        st.info("No images uploaded yet. Please upload an image to start detection.")
        #st.image("upload_image.jpeg", caption="Image Upload Placeholder", use_container_width=True)


# 🎥 Video Mode
elif option == ":movie_camera: Video":
    st.header(":dvd: Video Object Detection")
    st.markdown("Upload a video to detect objects frame by frame.")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file. It might be corrupt or in an unsupported format.")
            os.remove(tfile.name)
            st.stop()

        else:
            stframe = st.empty()
            progress_bar = st.progress(0)
            count_display = st.empty()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            with st.spinner("Processing video... This may take a while."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=0.3, verbose=False)
                    results = filter_results(results, selected_cls_indices)
                    annotated_frame = results[0].plot()
                    object_counts = count_objects(results, model.names)

                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                    count_str = ", ".join([f"{k}: {v}" for k, v in object_counts.items()])
                    count_display.info(f"**Detected Objects:** {count_str if count_str else 'None'}")

                    frame_idx += 1
                    progress_bar.progress(frame_idx / total_frames)

            cap.release()
            st.success("✅ Video processing complete!")
            # Note: Video download functionality is complex in Streamlit for real-time processing
            # and is omitted here for clarity. The original code's approach is valid but can be slow.
            st.info("Live preview finished. Download functionality for processed videos is a planned feature.")
        os.remove(video_path)


# 📸 Webcam Mode
elif option == ":camera: Webcam":
    st.header(":camera_with_flash: Live Webcam Detection")
    st.markdown("Click the button below to start your webcam and perform real-time detection.")

    run_webcam = st.checkbox("Start Webcam Detection", key="webcam_toggle")

    if run_webcam:
        st.info("Webcam is active. Uncheck the box above to stop.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Please check browser permissions and ensure it's not in use by another app.")
        else:
            stframe = st.empty()
            count_display = st.empty()

            while st.session_state.webcam_toggle: # Loop while checkbox is checked
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame from webcam.")
                    break

                results = model.predict(frame, conf=0.3, verbose=False)
                results = filter_results(results, selected_cls_indices)
                annotated_frame = results[0].plot()
                object_counts = count_objects(results, model.names)

                stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                count_str = ", ".join([f"{k}: {v}" for k, v in object_counts.items()])
                count_display.info(f"**Detected Objects:** {count_str if count_str else 'None'}")

            cap.release()
            st.success("✅ Webcam stream stopped.")
    else:
        st.warning("Webcam is off. Check the box to start detection.")

st.markdown("---")
# --- Footer / About Section ---
with st.expander("💡 About This Application"):
        st.markdown("""
        ### YOLOv11 Detection Suite
        This application demonstrates real-time object detection using the **YOLOv11 model**, a state-of-the-art deep learning model.

        **Features:**
        - 🖼️ **Image Detection:** Upload images for analysis.
        - 🎥 **Video Processing:** Analyze video files frame-by-frame.
        - 📹 **Live Webcam Feed:** Perform detection in real-time.
        - ⚙️ **Customizable Filters:** Select which object classes to detect.

        Built with ❤️ using Streamlit, OpenCV, and Ultralytics.
        """)