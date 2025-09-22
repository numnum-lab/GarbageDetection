import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import sys
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import av
import os
import torch
from ultralytics.nn.tasks import DetectionModel 

# Set page config first
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Object Detection",
)

# Debug: Show current working directory and file structure
if "yolo_model" not in st.session_state:
    try:
        # Get the directory where the script is located
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "my_model.pt"

        # Add the trusted class to PyTorch's allowed globals
        with torch.serialization.safe_globals([DetectionModel]):
            st.session_state.yolo_model = YOLO(str(model_path))
        
        st.success(f"YOLO Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

# Set up the correct path for imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))

# YOLO classes and messages
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

# Initialize session state
if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False
if "is_webcam_active" not in st.session_state:
    st.session_state.is_webcam_active = False
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.2

# Load YOLO model using the correct path
if "yolo_model" not in st.session_state:
    try:
        # Get the directory where the script is located
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "my_model.pt"
        
        st.info(f"Trying to load model from: {model_path}")
        st.session_state.yolo_model = YOLO(str(model_path))
        st.success(f"YOLO Model loaded successfully from: {model_path}")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error(f"Attempted path: {script_dir / 'my_model.pt'}")
        st.stop()

def display_detection_messages(detected_classes):
    """Display disposal messages for detected objects with color coding"""
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

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, yolo_model, conf_threshold):
        self.model = yolo_model
        self.conf_threshold = conf_threshold
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = self.model.predict(source=img, conf=self.conf_threshold)
        detections = results[0]
        
        boxes = (detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else [])
        confs = (detections.boxes.conf.cpu().numpy() if len(detections) > 0 else [])
        class_ids = (detections.boxes.cls.cpu().numpy().astype(int) if len(detections) > 0 else [])
        
        detected_classes = [yolo_classes[int(cls_id)] for cls_id in class_ids]
        
       # ✅ แก้ไข: อัปเดต st.session_state แทนการ return
        if detected_classes:
            st.session_state['detected_classes'] = detected_classes
        else:
            st.session_state['detected_classes'] = []
            
        # Draw bounding boxes and labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # ✅ แก้ไข: return แค่ VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def image_detection(uploaded_file, conf_threshold, selected_classes):
    """Process uploaded image"""
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
    
    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(image_cv, channels="BGR")
    with col2:
        display_detection_messages(detected_classes)

# Sidebar controls
with st.sidebar:
    st.title("Object Detection Settings ⚙️")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    st.session_state.confidence_threshold = confidence_threshold
    
    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)
    
    uploaded_file = st.file_uploader(
        "Upload an image or video 📤",
        type=["mp4", "mov", "avi", "m4v", "jpg", "png", "jpeg"],
    )
    
    if st.button("Use Webcam 📷" if not st.session_state.is_webcam_active else "Stop Webcam 🛑"):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        if st.session_state.is_webcam_active:
            st.session_state.is_detecting = True
        else:
            st.session_state.is_detecting = False
    
    detect_button = st.button(
        ("Start Detection ▶️" if not st.session_state.is_detecting else "Stop Detection 🛑"),
        disabled=(not uploaded_file and not st.session_state.is_webcam_active),
    )
    
    if detect_button:
        st.session_state.is_detecting = not st.session_state.is_detecting
    
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

# Main app logic
if st.session_state.is_detecting:
    if st.session_state.is_webcam_active:
        st.info("Detecting objects using webcam...")
        
        # Check if model is loaded before using WebRTC
        if "yolo_model" in st.session_state:
            webrtc_streamer(
                key="yolo-stream",
                # ✅ Pass the arguments using a lambda function
                video_processor_factory=lambda: YOLOProcessor(
                    st.session_state.yolo_model, 
                    st.session_state.confidence_threshold
                ),
                rtc_configuration=ClientSettings(
                    rtc_offer_min_port=10000,
                    rtc_offer_max_port=10200,
                ),
            )
            if "detected_classes" in st.session_state:
                display_detection_messages(st.session_state.detected_classes)
        else:
            st.error("YOLO model is not loaded. Please check the logs for errors.")
            
    elif uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            st.info("Detecting objects in image...")
            image_detection(uploaded_file, confidence_threshold, selected_classes)
        else:
            st.warning("Video processing not implemented in this version")
else:
    st.title("Smart Garbage Detection & Sorting Assistant")
    st.info("Upload an image or video, or start the webcam for object detection.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🗂️ Garbage Detection Using YOLO
        This is a group project by **Num Chakhatanon** and **Dawis Meedech** to help people sort garbage more easily.
        
        **Features:**
        - Real-time object detection via webcam
        - Image analysis
        - Smart disposal recommendations
        - Multiple waste categories supported
        """)
    
    with col2:
        st.write("""
        ### 📖 How to Use:
        1. **Upload** an image or use your **webcam**
        2. **Adjust** confidence threshold as needed
        3. **Select** specific classes to detect (optional)
        4. **Start detection** and follow the disposal instructions
        
        The system will automatically provide disposal guidance for detected items!
        """)
```