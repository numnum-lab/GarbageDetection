import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import sys
import os
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import av

st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Object Detection",
)

# YOLO classes and messages (same as before)
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

# Debug info
st.expander_debug = st.expander("🔧 Debug Info", expanded=False)
with st.expander_debug:
    script_dir = Path(__file__).resolve().parent
    st.write(f"Script directory: {script_dir}")
    st.write(f"Files in script directory: {os.listdir(str(script_dir))}")
    model_path = script_dir / "my_model.pt"
    st.write(f"Expected model path: {model_path}")
    st.write(f"Model file exists: {model_path.exists()}")
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        st.write(f"Model file size: {file_size:.2f} MB")

# Try to load model with better error handling
if "yolo_model" not in st.session_state:
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "my_model.pt"
    
    try:
        with st.spinner("Loading YOLO model..."):
            # First check if file exists and is readable
            if not model_path.exists():
                st.error(f"Model file not found at: {model_path}")
                st.stop()
            
            # Check file size to ensure it's not corrupted
            file_size = model_path.stat().st_size
            if file_size < 1000:  # Less than 1KB indicates corrupted file
                st.error(f"Model file appears to be corrupted (size: {file_size} bytes)")
                st.stop()
            
            # Try loading with different methods
            try:
                # Method 1: Direct path
                st.session_state.yolo_model = YOLO(str(model_path))
                st.success("✅ YOLO Model loaded successfully!")
            except Exception as e1:
                st.error(f"Method 1 failed: {e1}")
                try:
                    # Method 2: Use ultralytics default and reload
                    st.session_state.yolo_model = YOLO()  # Load default YOLOv8n
                    st.session_state.yolo_model = YOLO(str(model_path))  # Then load custom
                    st.success("✅ YOLO Model loaded successfully (Method 2)!")
                except Exception as e2:
                    st.error(f"Method 2 failed: {e2}")
                    # Method 3: Fall back to default YOLOv8n model
                    try:
                        st.warning("Loading default YOLOv8n model as fallback...")
                        st.session_state.yolo_model = YOLO('yolov8n.pt')
                        st.warning("⚠️ Using default YOLOv8n model (COCO classes, not garbage classes)")
                        # Update classes for default model
                        yolo_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"][:10]
                    except Exception as e3:
                        st.error(f"All methods failed: {e3}")
                        st.stop()
                        
    except Exception as e:
        st.error(f"Critical error loading model: {e}")
        st.stop()

def display_detection_messages(detected_classes):
    """Display disposal messages for detected objects"""
    if detected_classes:
        st.subheader("🎯 Detection Results:")
        unique_classes = list(set(detected_classes))
        
        for class_name in unique_classes:
            if class_name in disposal_messages:
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
            else:
                st.info(f"🔍 Detected: {class_name}")

# Global variable to store detected classes for display
if "detected_classes" not in st.session_state:
    st.session_state.detected_classes = []

class YOLOProcessor(VideoProcessorBase):
    def __init__(self, yolo_model):
        self.model = yolo_model
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Get confidence threshold
        conf_threshold = st.session_state.get("confidence_threshold", 0.2)
        
        try:
            # Perform object detection
            results = self.model.predict(source=img, conf=conf_threshold, verbose=False)
            
            if results and len(results) > 0:
                detections = results[0]
                
                if detections.boxes is not None and len(detections.boxes) > 0:
                    boxes = detections.boxes.xyxy.cpu().numpy()
                    confs = detections.boxes.conf.cpu().numpy()
                    class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                    
                    # Store detected classes for display
                    detected_classes = []
                    for class_id in class_ids:
                        if class_id < len(yolo_classes):
                            detected_classes.append(yolo_classes[class_id])
                    
                    st.session_state.detected_classes = detected_classes
                    
                    # Draw bounding boxes
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        if class_ids[i] < len(yolo_classes):
                            label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    st.session_state.detected_classes = []
            else:
                st.session_state.detected_classes = []
                
        except Exception as e:
            st.session_state.detected_classes = []
            # Draw error message on frame
            cv2.putText(img, f"Detection Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def image_detection(uploaded_file, conf_threshold, selected_classes):
    """Process uploaded image"""
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    try:
        results = st.session_state.yolo_model.predict(source=image_cv, conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            detections = results[0]
            
            if detections.boxes is not None and len(detections.boxes) > 0:
                boxes = detections.boxes.xyxy.cpu().numpy()
                confs = detections.boxes.conf.cpu().numpy()
                class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                
                detected_classes = []
                for i, box in enumerate(boxes):
                    if class_ids[i] < len(yolo_classes):
                        detected_classes.append(yolo_classes[class_ids[i]])
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(image_cv, channels="BGR")
                with col2:
                    display_detection_messages(detected_classes)
            else:
                st.image(image_cv, channels="BGR")
                st.info("No objects detected")
        else:
            st.image(image_cv, channels="BGR")
            st.info("No detection results")
            
    except Exception as e:
        st.error(f"Error during image detection: {e}")
        st.image(image_cv, channels="BGR")

# Sidebar controls
with st.sidebar:
    st.title("Object Detection Settings ⚙️")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    st.session_state.confidence_threshold = confidence_threshold
    
    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)
    
    uploaded_file = st.file_uploader(
        "Upload an image 📤",
        type=["jpg", "png", "jpeg"],
    )
    
    if st.button("Use Webcam 📷" if not st.session_state.is_webcam_active else "Stop Webcam 🛑"):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        st.session_state.is_detecting = st.session_state.is_webcam_active
    
    if uploaded_file and st.button("Analyze Image 🔍"):
        st.session_state.is_detecting = True

# Main app logic
if st.session_state.is_webcam_active:
    st.info("🎥 Real-time webcam detection active...")
    
    # Check if model is loaded
    if "yolo_model" in st.session_state:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            webrtc_streamer(
                key="yolo-stream",
                video_processor_factory=YOLOProcessor,
                rtc_configuration=ClientSettings(
                    ice_servers=[{"urls": ["stun:stun.l.google.com:19302"]}],
                    rtc_offer_min_port=10000,
                    rtc_offer_max_port=10200,
                ),
                args=(st.session_state.yolo_model,),
                media_stream_constraints={"video": True, "audio": False},
            )
        
        with col2:
            if st.session_state.detected_classes:
                display_detection_messages(st.session_state.detected_classes)
            else:
                st.info("👁️ Waiting for detection...")
    else:
        st.error("YOLO model not loaded")

elif uploaded_file:
    st.info("📸 Analyzing uploaded image...")
    image_detection(uploaded_file, confidence_threshold, selected_classes)

else:
    st.title("🗑️ Smart Garbage Detection & Sorting Assistant")
    st.info("Upload an image or start webcam detection to begin!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Features:
        - **Real-time webcam detection**
        - **Image analysis**
        - **Smart disposal recommendations**
        - **Multiple waste categories**
        """)
    
    with col2:
        st.markdown("""
        ### 📋 How to use:
        1. **Click "Use Webcam"** for real-time detection
        2. **Upload an image** for static analysis
        3. **Adjust confidence threshold** as needed
        4. **Follow disposal instructions** for detected items
        """)