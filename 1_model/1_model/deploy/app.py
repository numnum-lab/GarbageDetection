import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import os
import torch
from ultralytics.nn.tasks import DetectionModel
import time

# Set page config first
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Real-time Object Detection",
)

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

# Color mapping for different waste types (BGR format for OpenCV)
color_mapping = {
    "battery": (0, 0, 255),      # Red - Hazardous
    "biological": (0, 255, 0),   # Green - Organic
    "cardboard": (0, 165, 255),  # Orange - Recyclable
    "glass": (0, 165, 255),      # Orange - Recyclable
    "metal": (0, 165, 255),      # Orange - Recyclable
    "paper": (0, 165, 255),      # Orange - Recyclable
    "plastic": (0, 165, 255),    # Orange - Recyclable
    "clothes": (255, 0, 0),      # Blue - Donate/General
    "shoes": (255, 0, 0),        # Blue - Donate/General
    "trash": (128, 128, 128),    # Gray - General
}

# Initialize session state
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = 0.25
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = []
if "camera_initialized" not in st.session_state:
    st.session_state.camera_initialized = False

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        # Get the directory where the script is located
        script_dir = Path(__file__).resolve().parent
        model_path = script_dir / "my_model.pt"
        
        # Try different possible paths
        possible_paths = [
            model_path,
            Path("my_model.pt"),
            Path("models/my_model.pt"),
            Path("./my_model.pt")
        ]
        
        model_loaded = False
        for path in possible_paths:
            if path.exists():
                try:
                    with torch.serialization.safe_globals([DetectionModel]):
                        model = YOLO(str(path))
                    st.success(f"✅ YOLO Model loaded successfully from: {path}")
                    model_loaded = True
                    return model
                except Exception as e:
                    st.warning(f"Failed to load from {path}: {e}")
                    continue
        
        if not model_loaded:
            st.error("❌ Could not load YOLO model from any path. Please ensure 'my_model.pt' exists in the correct directory.")
            st.info("Tried the following paths:")
            for path in possible_paths:
                st.info(f"  - {path}")
            return None
            
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

def display_detection_messages(detected_classes):
    """Display disposal messages for detected objects with color coding"""
    if detected_classes:
        st.subheader("🎯 Real-time Detection Results:")
        unique_classes = list(set(detected_classes))
        
        # Create columns based on number of detected classes
        if len(unique_classes) <= 3:
            cols = st.columns(len(unique_classes))
        else:
            cols = st.columns(3)
        
        for i, class_name in enumerate(unique_classes):
            col_index = i % len(cols)
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

def process_frame_with_yolo(frame, model, conf_threshold, selected_classes=None):
    """Process a single frame with YOLO detection"""
    # Run YOLO inference
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    detections = results[0]
    
    detected_classes = []
    
    if len(detections) > 0:
        boxes = detections.boxes.xyxy.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by selected classes if specified
        if selected_classes:
            filtered_indices = [i for i, cls_id in enumerate(class_ids) 
                              if yolo_classes[cls_id] in selected_classes]
            if filtered_indices:
                boxes = boxes[filtered_indices]
                confs = confs[filtered_indices]
                class_ids = class_ids[filtered_indices]
        
        # Draw bounding boxes and collect detected classes
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            class_name = yolo_classes[cls_id]
            detected_classes.append(class_name)
            
            # Get color for the class
            color = color_mapping.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, detected_classes

def initialize_camera():
    """Initialize the camera and return the capture object"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check if your camera is connected and not being used by another application.")
        return None
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap

def run_camera_detection(model, conf_threshold, selected_classes):
    """Run camera detection with auto-refresh"""
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        return
    
    # Create placeholders
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
        fps_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Detection Results")
        detection_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        stop_button = st.button("🛑 Stop Detection", type="secondary", use_container_width=True)
    with col_btn2:
        refresh_button = st.button("🔄 Refresh", type="primary", use_container_width=True)
    
    if stop_button:
        cap.release()
        st.session_state.camera_active = False
        st.rerun()
    
    # Auto-refresh mechanism
    auto_refresh = st.empty()
    
    # Main detection loop
    fps_counter = 0
    start_time = time.time()
    frame_count = 0
    
    try:
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process every few frames to improve performance
            if frame_count % 2 == 0:  # Process every 2nd frame
                # Process frame with YOLO
                processed_frame, detected_classes = process_frame_with_yolo(
                    frame, model, conf_threshold, selected_classes
                )
                
                # Update detected objects in session state
                st.session_state.detected_objects = detected_classes
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 processed frames
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    start_time = current_time
                
                # Add FPS to frame
                fps_text = f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --"
                cv2.putText(processed_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update displays
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                fps_placeholder.info(f"📊 {fps_text} | Frame: {frame_count}")
                
                # Update detection messages
                with detection_placeholder.container():
                    if detected_classes:
                        display_detection_messages(detected_classes)
                    else:
                        st.info("👁️ Looking for objects...")
            else:
                # For non-processed frames, just convert and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            frame_count += 1
            
            # Auto-refresh mechanism to keep the stream active
            if frame_count % 300 == 0:  # Refresh every 300 frames (~10 seconds)
                with auto_refresh.container():
                    st.empty()
                time.sleep(0.01)
            
            # Small delay to prevent overwhelming
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        st.error(f"Camera error: {e}")
    finally:
        cap.release()
        st.session_state.camera_active = False

def image_detection(uploaded_file, model, conf_threshold, selected_classes):
    """Process uploaded image"""
    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    processed_image, detected_classes = process_frame_with_yolo(
        image_cv, model, conf_threshold, selected_classes
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(processed_image, channels="BGR", caption="Detection Results")
    with col2:
        if detected_classes:
            display_detection_messages(detected_classes)
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")

# Load the model
yolo_model = load_yolo_model()

# Sidebar controls
with st.sidebar:
    st.title("🎛️ Detection Settings")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=st.session_state.confidence_threshold,
        step=0.05
    )
    st.session_state.confidence_threshold = confidence_threshold
    
    # Class selection
    selected_classes = st.multiselect(
        "Select classes to detect (leave empty for all)",
        yolo_classes,
        default=[]
    )
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "png", "jpeg"],
        help="Upload an image for detection"
    )
    
    st.markdown("---")
    
    # Camera controls
    st.subheader("📹 Camera Controls")
    
    if not st.session_state.camera_active:
        if st.button("🚀 Start Real-time Detection", type="primary", use_container_width=True):
            if yolo_model is not None:
                st.session_state.camera_active = True
                st.rerun()
            else:
                st.error("Cannot start camera: YOLO model not loaded")
    else:
        st.info("Camera is running... Use stop button in main area to stop.")
    
    # Status indicator
    if st.session_state.camera_active:
        st.success("🟢 Camera Active")
    else:
        st.info("🔴 Camera Inactive")
    
    # Camera troubleshooting
    with st.expander("🔧 Camera Troubleshooting"):
        st.markdown("""
        **If camera doesn't work:**
        1. **Check camera permissions** - Allow browser to access camera
        2. **Close other apps** - Ensure no other apps are using the camera
        3. **Refresh page** - Sometimes helps reset camera state
        4. **Try different browser** - Chrome/Firefox usually work best
        5. **Check camera index** - Some systems may need camera index 1 or 2
        """)
    
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
        
        st.markdown("### 🟦 **Donate/General**")
        st.info("👕 **Clothes:** Consider **Donating** or dispose in the **GENERAL** bin.")
        st.info("👟 **Shoes:** Consider **Donating** or dispose in the **GENERAL** bin.")
        
        st.markdown("### ⬛ **General Waste**")
        st.error("🗑️ **Trash:** Dispose in the **GENERAL** bin.")

# Main content area
st.title("🔍 Smart Garbage Detection & Sorting Assistant")
st.markdown("**Real-time Object Detection using YOLOv11** - *by Num Chakhatanon & Dawis Meedech*")

# Check if model is loaded
if yolo_model is None:
    st.error("❌ YOLO model is not available. Please check the model file.")
    st.stop()

# Main application logic
if st.session_state.camera_active:
    st.info("🎥 **Real-time detection active** - Objects will be detected and classified in real-time!")
    run_camera_detection(yolo_model, confidence_threshold, selected_classes)

elif uploaded_file:
    st.info("📸 **Image Detection Mode** - Analyzing uploaded image...")
    image_detection(uploaded_file, yolo_model, confidence_threshold, selected_classes)

else:
    # Welcome screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Welcome to Smart Garbage Detection!
        
        This advanced AI system helps you sort waste properly using real-time object detection.
        
        ### ✨ Features:
        - **Real-time camera detection** with live video feed
        - **Image analysis** for uploaded photos  
        - **Smart disposal recommendations** for detected items
        - **Color-coded classification** system
        - **High-performance YOLOv11** model
        
        ### 🎯 How to Use:
        1. **Real-time Detection**: Click "Start Real-time Detection" to use your camera
        2. **Image Analysis**: Upload an image using the sidebar
        3. **Adjust Settings**: Fine-tune confidence threshold and select specific classes
        4. **Follow Instructions**: Get instant disposal guidance for detected objects
        """)
        
        # Quick start guide
        st.info("""
        💡 **Quick Start:**
        - Grant camera permissions when prompted
        - Point camera at objects to detect
        - View real-time classifications and disposal instructions
        """)
    
    with col2:
        st.markdown("""
        ### 🗂️ Supported Objects:
        - 🔋 Battery (Hazardous)
        - 🍃 Biological (Organic) 
        - 📦 Cardboard (Recycling)
        - 👕 Clothes (Donate/General)
        - 🍶 Glass (Recycling)
        - 🔩 Metal (Recycling)
        - 📄 Paper (Recycling)
        - ♻️ Plastic (Recycling)
        - 👟 Shoes (Donate/General)
        - 🗑️ General Trash
        
        ### 🎨 Color Coding:
        - 🔴 **Red**: Hazardous waste
        - 🟢 **Green**: Organic waste
        - 🟠 **Orange**: Recyclables
        - 🔵 **Blue**: Donate/General
        - ⚫ **Gray**: General waste
        """)

# Add some custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    div[data-testid="column"] {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)