import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from huggingface_hub import hf_hub_download
import torch
from pathlib import Path

# ------------------------------------------------
# Initial Session State
# ------------------------------------------------
if "is_webcam_active" not in st.session_state:
    st.session_state.is_webcam_active = False

if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.2

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Smart Garbage Detection",
)

# ------------------------------------------------
# YOLO Classes & Disposal Messages
# ------------------------------------------------
yolo_classes = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

disposal_messages = {
    "battery": "⚡ **Battery detected!** Dispose in the **HAZARDOUS** bin.",
    "biological": "🍃 **Biological waste detected!** Dispose in the **ORGANIC** bin.",
    "cardboard": "📦 **Cardboard detected!** Flatten and dispose in the **RECYCLING** bin.",
    "clothes": "👕 **Clothes detected!** Consider donating, or dispose in the **GENERAL** bin.",
    "glass": "🍶 **Glass detected!** Dispose in the **RECYCLING** bin.",
    "metal": "🔩 **Metal detected!** Dispose in the **RECYCLING** bin.",
    "paper": "📄 **Paper detected!** Dispose in the **RECYCLING** bin.",
    "plastic": "♻️ **Plastic detected!** Dispose in the **RECYCLING** bin.",
    "shoes": "👟 **Shoes detected!** Consider donating, or dispose in the **GENERAL** bin.",
    "trash": "🗑️ **General trash detected!** Dispose in the **GENERAL** bin.",
}

# ------------------------------------------------
# Load YOLO Model
# ------------------------------------------------
@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLO model from a Hugging Face repository and caches it.
    """
    try:
        repo_id = "Numgfsdf/garbage-detection-model"
        filename = "my_model.pt"
        
        # Download the file from Hugging Face
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the YOLO model
        model = YOLO(model_path)
        st.success(f"YOLO Model loaded successfully from {model_path}!")
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Load the model only once
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = load_yolo_model()

# ------------------------------------------------
# Video Processor for Webcam
# ------------------------------------------------
class VideoProcessor(VideoProcessorBase):
    """
    Processes video frames from the webcam for object detection.
    """
    def __init__(self):
        # We access the model from Streamlit's session state
        self.model = st.session_state.yolo_model
        # We access the confidence threshold from Streamlit's session state
        self.confidence = st.session_state.confidence_threshold

    def recv(self, frame):
        # Convert the frame to a format YOLO can use
        image = frame.to_ndarray(format="bgr24")

        # Perform detection using the model
        if self.model:
            results = self.model.predict(source=image, conf=self.confidence)
            # Draw bounding boxes and labels on the frame
            annotated_image = results[0].plot()
            # Return the annotated frame
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
        
        # If model is not loaded, return the original frame
        return frame
        
# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def display_detection_messages(detected_classes):
    """
    Displays the disposal messages for detected objects.
    """
    if detected_classes:
        st.subheader("🎯 Detection Results:")
        unique_classes = list(set(detected_classes))
        
        # Determine the number of columns to use
        cols = st.columns(min(len(unique_classes), 2))
        
        for i, class_name in enumerate(unique_classes):
            with cols[i % 2]:
                message = disposal_messages.get(class_name, "🗑️ **Unknown item:** Dispose in the **GENERAL** bin.")
                
                if class_name == "battery":
                    st.error(f"🟥 {message}")
                elif class_name == "biological":
                    st.success(f"🟢 {message}")
                elif class_name in ["cardboard", "glass", "metal", "paper", "plastic"]:
                    st.warning(f"🟡 {message}")
                elif class_name in ["clothes", "shoes"]:
                    st.info(f"🟦 {message}")
                else:
                    st.error(f"⬛ {message}")

#Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

#Webcam Config
WEBCAM_DEFAULT_ID = 0  # Default webcam (usually the built-in webcam)

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'

#In case of your custom model
#DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

SEGMENTATION_MODEL  = MODEL_DIR/'yolo11n-seg.pt'

POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

    
    # Filter detections based on selected classes
if selected_classes:
        filtered_results = [
            (box, conf, class_id)
            for box, conf, class_id in zip(boxes, confs, class_ids)
            if yolo_classes[class_id] in selected_classes
        ]
        boxes, confs, class_ids = zip(*filtered_results) if filtered_results else ([], [], [])
        detected_classes = [yolo_classes[class_id] for class_id in class_ids]
else:
        detected_classes = [yolo_classes[class_id] for class_id in class_ids]

    # Draw bounding boxes on the image
for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_cv, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

col1, col2 = st.columns([2, 1])
with col1:
        st.image(image_cv, channels="BGR")
with col2:
        display_detection_messages(detected_classes)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.title("Object Detection Settings ⚙️")
    
    # Sliders and selectors for user input
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05)
    st.session_state.confidence_threshold = confidence_threshold

    selected_classes = st.multiselect("Select classes to detect", yolo_classes)

    uploaded_file = st.file_uploader(
        "Upload an image 📤",
        type=["jpg", "png", "jpeg"],
    )

    # Buttons to control detection modes
    webcam_button_text = "Stop Webcam 🛑" if st.session_state.is_webcam_active else "Use Webcam 📷"
    if st.button(webcam_button_text):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        # Reset detection state when switching modes
        if not st.session_state.is_webcam_active:
            st.session_state.is_detecting = False

    detect_button = st.button(
        ("Start Detection ▶️" if not st.session_state.is_detecting else "Stop Detection 🛑"),
        disabled=(not uploaded_file and not st.session_state.is_webcam_active),
    )

    if detect_button:
        st.session_state.is_detecting = not st.session_state.is_detecting
        if not st.session_state.is_webcam_active:
            st.session_state.is_webcam_active = False

    # Disposal Guide
    st.markdown("---")
    st.subheader("📋 Disposal Guide")
    with st.expander("View all disposal instructions"):
        st.markdown("### 🟥 **Hazardous Bin**")
        st.error("⚡ **Battery:** Dispose in the **HAZARDOUS** bin.")
        st.markdown("### 🟢 **Organic Bin**")
        st.success("🍃 **Biological:** Dispose in the **ORGANIC** bin.")
        st.markdown("### 🟡 **Recyclables**")
        st.warning("📦 **Cardboard:** Flatten and dispose in the **RECYCLING** bin.")
        st.warning("🍶 **Glass:** Dispose in the **RECYCLING** bin.")
        st.warning("🔩 **Metal:** Dispose in the **RECYCLING** bin.")
        st.warning("📄 **Paper:** Dispose in the **RECYCLING** bin.")
        st.warning("♻️ **Plastic:** Dispose in the **RECYCLING** bin.")
        st.markdown("### 🟦 **Donate**")
        st.info("👕 **Clothes:** Consider **Donating** or dispose in the **GENERAL** bin.")
        st.info("👟 **Shoes:** Consider **Donating** or dispose in the **GENERAL** bin.")
        st.markdown("### ⬛ **General Waste**")
        st.error("🗑️ **Trash:** Dispose in the **GENERAL** bin.")

# ------------------------------------------------
# Main Content
# ------------------------------------------------
st.title("Smart Garbage Detection & Sorting Assistant")

if st.session_state.is_detecting:
    if st.session_state.is_webcam_active:
        st.info("🔴 Webcam mode active - Detecting objects in real-time.")
        
        # Use streamlit-webrtc for live video stream
        webrtc_streamer(
            key="webrtc",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )
        st.warning("Please allow webcam access in your browser.")
        
    elif uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            st.info("Detecting objects in image...")
            image_detection(uploaded_file, confidence_threshold, selected_classes)
        else:
            st.warning("Only image files are supported in this version.")
else:
    st.info("Upload an image or activate webcam for object detection.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🗂️ Garbage Detection Using YOLO
        This project helps people sort garbage more easily.

        **Features:**
        - Real-time webcam capture
        - Image analysis
        - Smart disposal recommendations
        - Multiple waste categories supported
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:
        1. **Upload** an image or **activate webcam** from the sidebar.
        2. **Adjust** confidence threshold as needed.
        3. **Select** specific classes to detect (optional).
        4. **Start detection** and follow the disposal instructions.

        The system will automatically provide disposal guidance for detected items!
        """)