import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import time
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading

# ------------------------------------------------
# Initial Session State
# ------------------------------------------------
if "is_webcam_active" not in st.session_state:
    st.session_state.is_webcam_active = False

if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False

if "detected_classes" not in st.session_state:
    st.session_state.detected_classes = []

if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.2

if "live_detection_active" not in st.session_state:
    st.session_state.live_detection_active = False

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Real-Time Object Detection",
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
    try:
        repo_id = "Numgfsdf/garbage-detection-model"
        filename = "my_model.pt"

        # Download model from Hugging Face
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # Load YOLO model
        model = YOLO(model_path)
        st.success(f"โหลด YOLO Model สำเร็จจาก {model_path}!")
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด YOLO model: {e}")
        return None

# Load model once
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = load_yolo_model()

# Thread lock for detected classes
lock = threading.Lock()

# ------------------------------------------------
# Real-time Video Processor Class
# ------------------------------------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.2
        self.selected_classes = []
        self.detected_classes = []

    def set_params(self, conf_threshold, selected_classes):
        self.conf_threshold = conf_threshold
        self.selected_classes = selected_classes

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if st.session_state.yolo_model is not None:
            try:
                # Run YOLO detection
                results = st.session_state.yolo_model.predict(
                    source=img, 
                    conf=self.conf_threshold,
                    verbose=False
                )
                detections = results[0]

                # Extract detection data
                if len(detections) > 0:
                    boxes = detections.boxes.xyxy.cpu().numpy()
                    confs = detections.boxes.conf.cpu().numpy()
                    class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                else:
                    boxes, confs, class_ids = [], [], []

                # Filter based on selected classes
                detected_classes = []
                if self.selected_classes:
                    filtered = [
                        (box, conf, class_id)
                        for box, conf, class_id in zip(boxes, confs, class_ids)
                        if yolo_classes[class_id] in self.selected_classes
                    ]
                    if filtered:
                        boxes, confs, class_ids = zip(*filtered)
                        detected_classes = [yolo_classes[class_id] for class_id in class_ids]
                    else:
                        boxes, confs, class_ids = [], [], []
                else:
                    detected_classes = [yolo_classes[class_id] for class_id in class_ids]

                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                    )

                # Update detected classes in session state (thread-safe)
                with lock:
                    st.session_state.detected_classes = detected_classes

            except Exception as e:
                # Draw error message on frame
                cv2.putText(img, f"Detection Error: {str(e)[:50]}...", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def display_detection_messages(detected_classes):
    if detected_classes:
        st.subheader("🎯 Detection Results:")
        unique_classes = list(set(detected_classes))

        if len(unique_classes) <= 2:
            cols = st.columns(len(unique_classes))
        else:
            cols = st.columns(2)

        for i, class_name in enumerate(unique_classes):
            col_index = i if len(unique_classes) <= 2 else i % 2
            with cols[col_index]:
                if class_name == "battery":
                    st.error(f"🟥 {disposal_messages[class_name]}")
                elif class_name == "biological":
                    st.success(f"🟢 {disposal_messages[class_name]}")
                elif class_name in ["cardboard", "glass", "metal", "paper", "plastic"]:
                    st.warning(f"🟡 {disposal_messages[class_name]}")
                elif class_name in ["clothes", "shoes"]:
                    st.info(f"🟦 {disposal_messages[class_name]}")
                else:
                    st.error(f"⬛ {disposal_messages[class_name]}")

def image_detection(uploaded_file, conf_threshold, selected_classes):
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform image detection.")
        return

    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = st.session_state.yolo_model.predict(source=image_cv, conf=conf_threshold)
    detections = results[0]

    boxes = detections.boxes.xyxy.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)

    detected_classes = []
    if selected_classes:
        filtered = [
            (box, conf, class_id)
            for box, conf, class_id in zip(boxes, confs, class_ids)
            if yolo_classes[class_id] in selected_classes
        ]
        if filtered:
            boxes, confs, class_ids = zip(*filtered)
            detected_classes = [yolo_classes[class_id] for class_id in class_ids]
        else:
            boxes, confs, class_ids = [], [], []
    else:
        detected_classes = [yolo_classes[class_id] for class_id in class_ids]

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
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    st.session_state.confidence_threshold = confidence_threshold

    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)

    uploaded_file = st.file_uploader(
        "Upload an image 📤",
        type=["jpg", "png", "jpeg"],
    )

    # Detection mode selection
    st.subheader("Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Single Image", "Real-time Camera"],
        help="Single Image: Upload and analyze one image\nReal-time Camera: Live video stream with detection"
    )

    # Detection control button for single image
    if detection_mode == "Single Image" and uploaded_file:
        if st.button("🔍 Analyze Image"):
            st.session_state.is_detecting = True

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
st.title("🔍 Smart Garbage Detection & Sorting Assistant")

# Handle different detection modes
if detection_mode == "Real-time Camera":
    st.info("📹 Real-time Detection Mode - Live camera stream with object detection")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # WebRTC Configuration
        RTC_CONFIGURATION = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Create video transformer instance
        video_transformer = VideoTransformer()
        video_transformer.set_params(confidence_threshold, selected_classes)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            video_transformer_factory=lambda: video_transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
            async_processing=True,
        )
        
        # Update transformer parameters when settings change
        if webrtc_ctx.video_transformer:
            webrtc_ctx.video_transformer.set_params(confidence_threshold, selected_classes)
    
    with col2:
        # Display detection results (updates automatically)
        st.subheader("🎯 Live Results")
        results_placeholder = st.empty()
        
        # Update results every second
        if webrtc_ctx.state.playing:
            while webrtc_ctx.state.playing:
                with results_placeholder.container():
                    with lock:
                        current_detections = st.session_state.detected_classes.copy()
                    
                    if current_detections:
                        display_detection_messages(current_detections)
                    else:
                        st.info("👀 Looking for objects...")
                
                time.sleep(0.5)  # Update every 500ms

elif detection_mode == "Single Image":
    if st.session_state.is_detecting and uploaded_file:
        # Single image detection
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            st.info("🔍 Detecting objects in uploaded image...")
            image_detection(uploaded_file, confidence_threshold, selected_classes)
            st.session_state.is_detecting = False  # Reset after single image processing
        else:
            st.warning("⚠️ Only image files (JPG, PNG, JPEG) are supported")
    else:
        # Welcome screen
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            ### 🗂️ Garbage Detection Using YOLO
            This project helps people sort garbage more easily using AI-powered object detection.

            **Features:**
            - 📹 **Real-time camera detection**
            - 🖼️ Single image analysis
            - 🎯 Smart disposal recommendations
            - 🗑️ Multiple waste categories supported
            """)

        with col2:
            st.write("""
            ### 📖 How to Use:
            1. **Choose** detection mode in the sidebar
            2. **Upload** an image or use real-time camera
            3. **Adjust** confidence threshold as needed
            4. **Select** specific classes to detect (optional)
            5. **Start detection** and follow the disposal instructions

            The system will automatically provide disposal guidance for detected items!
            """)

        # Display sample images or instructions
        st.info("👈 Choose your detection mode from the sidebar to get started!")

# ------------------------------------------------
# Installation Instructions
# ------------------------------------------------
if st.sidebar.button("📦 Show Installation Instructions"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Installation Required")
    st.sidebar.code("""
pip install streamlit-webrtc
pip install opencv-python
pip install ultralytics
pip install torch
pip install huggingface_hub
    """)
    st.sidebar.info("Install these packages to run real-time detection!")