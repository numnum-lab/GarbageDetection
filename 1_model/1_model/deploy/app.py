import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import time
from huggingface_hub import hf_hub_download

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

if "frame_skip" not in st.session_state:
    st.session_state.frame_skip = 3  # Process every 3rd frame for performance

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Real-time Object Detection",
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

        # ดาวน์โหลดไฟล์จาก Hugging Face
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # โหลดโมเดล YOLO
        model = YOLO(model_path)
        st.success(f"โหลด YOLO Model สำเร็จจาก {model_path}!")
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด YOLO model: {e}")
        return None

# โหลดโมเดลเพียงครั้งเดียว
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = load_yolo_model()

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def display_detection_messages(detected_classes):
    if detected_classes:
        st.subheader("🎯 Real-time Detection Results:")
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

def process_frame(frame, conf_threshold, selected_classes):
    """Process a single frame for object detection"""
    if not st.session_state.yolo_model:
        return frame, []

    # Run YOLO detection
    results = st.session_state.yolo_model.predict(source=frame, conf=conf_threshold, verbose=False)
    detections = results[0]

    if len(detections.boxes) == 0:
        return frame, []

    boxes = detections.boxes.xyxy.cpu().numpy()
    confs = detections.boxes.conf.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)

    detected_classes = []
    
    # Filter by selected classes if any
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

    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
        
        # Choose color based on class type
        class_name = yolo_classes[class_ids[i]]
        if class_name == "battery":
            color = (0, 0, 255)  # Red
        elif class_name == "biological":
            color = (0, 255, 0)  # Green
        elif class_name in ["cardboard", "glass", "metal", "paper", "plastic"]:
            color = (0, 255, 255)  # Yellow
        elif class_name in ["clothes", "shoes"]:
            color = (255, 0, 0)  # Blue
        else:
            color = (128, 128, 128)  # Gray
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, detected_classes

def realtime_detection(conf_threshold, selected_classes):
    """Real-time webcam detection with continuous processing"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera permissions.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create placeholders for video and results
    video_placeholder = st.empty()
    results_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    
    try:
        while st.session_state.is_detecting:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            frame_count += 1
            
            # Process every nth frame for performance
            if frame_count % st.session_state.frame_skip == 0:
                # Process frame for detection
                processed_frame, detected_classes = process_frame(frame, conf_threshold, selected_classes)
                
                # Update session state with latest detections
                st.session_state.detected_classes = detected_classes
                
                # Display processed frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display detection results
                with results_placeholder.container():
                    if detected_classes:
                        display_detection_messages(detected_classes)
                    else:
                        st.info("🔍 Scanning for objects...")
                
                fps_counter += 1
            else:
                # Display unprocessed frame for smooth video
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update stats every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = fps_counter / elapsed_time if elapsed_time > 0 else 0
                stats_placeholder.metric("Processing FPS", f"{fps:.1f}")
            
            # Add small delay to prevent overwhelming the system
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error during real-time detection: {e}")
    finally:
        cap.release()
        video_placeholder.empty()
        st.info("📹 Webcam stopped")

def image_detection(uploaded_file, conf_threshold, selected_classes):
    """Static image detection (original functionality)"""
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform image detection.")
        return

    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    processed_image, detected_classes = process_frame(image_cv, conf_threshold, selected_classes)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(processed_image, channels="BGR")
    with col2:
        display_detection_messages(detected_classes)

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.title("Real-time Detection Settings ⚙️")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    st.session_state.confidence_threshold = confidence_threshold

    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)
    
    # Performance settings
    st.markdown("### Performance Settings")
    frame_skip = st.slider("Frame Skip (1=process all frames)", 1, 10, 3,
                          help="Higher values = better performance, lower accuracy")
    st.session_state.frame_skip = frame_skip

    uploaded_file = st.file_uploader(
        "Upload an image 📤",
        type=["jpg", "png", "jpeg"],
    )

    # Detection mode selection
    st.markdown("### Detection Mode")
    detection_mode = st.radio(
        "Choose detection mode:",
        ["Static Image", "Real-time Webcam"],
        index=1 if st.session_state.is_webcam_active else 0
    )

    if detection_mode == "Real-time Webcam":
        webcam_button = st.button(
            "Start Real-time Detection 📹" if not st.session_state.is_detecting 
            else "Stop Detection 🛑"
        )
        
        if webcam_button:
            st.session_state.is_detecting = not st.session_state.is_detecting
            st.session_state.is_webcam_active = st.session_state.is_detecting
            
    else:
        detect_button = st.button(
            "Analyze Image 🔍",
            disabled=not uploaded_file,
        )
        
        if detect_button and uploaded_file:
            st.session_state.is_detecting = True
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
if st.session_state.is_detecting and st.session_state.is_webcam_active:
    st.title("🔴 Real-time Object Detection")
    st.info("Real-time webcam detection is active. Objects will be detected and classified automatically.")
    
    # Start real-time detection
    realtime_detection(confidence_threshold, selected_classes)
    
elif st.session_state.is_detecting and uploaded_file:
    st.title("📸 Static Image Analysis")
    st.info("Analyzing uploaded image...")
    image_detection(uploaded_file, confidence_threshold, selected_classes)
    st.session_state.is_detecting = False  # Reset after single image analysis
    
else:
    st.title("Smart Garbage Detection & Sorting Assistant")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🔍 Real-time Garbage Detection Using YOLO
        This project helps people sort garbage more easily with real-time detection.

        **Features:**
        - **Real-time webcam detection** with continuous processing
        - Static image analysis
        - Smart disposal recommendations
        - Performance optimization settings
        - Color-coded detection boxes
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:

        **For Real-time Detection:**
        1. Select "Real-time Webcam" mode
        2. Click "Start Real-time Detection"
        3. Point camera at objects
        4. Follow real-time disposal instructions

        **For Image Analysis:**
        1. Select "Static Image" mode
        2. Upload an image
        3. Click "Analyze Image"
        4. View results and disposal guidance
        """)
    
    # Display demo information
    with st.expander("🎯 Detection Classes"):
        st.write("The system can detect and classify these waste types:")
        cols = st.columns(2)
        for i, class_name in enumerate(yolo_classes):
            with cols[i % 2]:
                st.write(f"• **{class_name.title()}**")
    
    # Performance tips
    with st.expander("⚡ Performance Tips"):
        st.write("""
        - **Frame Skip**: Increase to 5-7 for better performance on slower devices
        - **Confidence Threshold**: Lower values detect more objects but may include false positives
        - **Selected Classes**: Choose specific classes to improve detection speed
        - **Camera Resolution**: The app automatically sets optimal resolution for performance
        """)

# ------------------------------------------------
# Status indicator
# ------------------------------------------------
if st.session_state.is_detecting:
    st.sidebar.success("🟢 Detection Active")
else:
    st.sidebar.info("⚪ Detection Inactive")