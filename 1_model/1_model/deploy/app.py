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

if "live_detection_active" not in st.session_state:
    st.session_state.live_detection_active = False

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Object Detection",
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
# Live Streaming Function (Modified for Streamlit)
# ------------------------------------------------
def live_streaming_detection(conf_threshold, selected_classes):
    """
    Live streaming detection using Streamlit's camera_input
    This replaces cv2.VideoCapture which doesn't work in web browsers
    """
    st.info("📸 Live Detection Mode - Take photos to detect objects")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use Streamlit's camera input for live capture
        camera_image = st.camera_input("📷 Take a photo for detection", key="live_camera")
        
        if camera_image is not None:
            # Process the captured image
            try:
                image = Image.open(camera_image)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Run YOLO detection
                results = st.session_state.yolo_model.predict(source=image_cv, conf=conf_threshold)
                detections = results[0]

                # Extract detection data
                boxes = detections.boxes.xyxy.cpu().numpy()
                confs = detections.boxes.conf.cpu().numpy()
                class_ids = detections.boxes.cls.cpu().numpy().astype(int)

                # Filter based on selected classes
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

                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        image_cv,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                # Display processed image
                st.image(image_cv, channels="BGR", caption="Processed Image")
                
                # Update session state with detected classes
                st.session_state.detected_classes = detected_classes

            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
    
    with col2:
        # Display detection messages
        if st.session_state.detected_classes:
            display_detection_messages(st.session_state.detected_classes)
        else:
            st.info("📸 Take a photo to start detecting objects!")
        
        # Auto-refresh option
        if st.checkbox("🔄 Auto-refresh camera", value=False):
            time.sleep(2)
            st.experimental_rerun()

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
        ["Single Image", "Live Camera"],
        help="Single Image: Upload and analyze one image\nLive Camera: Use camera for continuous detection"
    )

    # Update session state based on mode
    if detection_mode == "Live Camera":
        st.session_state.is_webcam_active = True
        st.session_state.live_detection_active = True
    else:
        st.session_state.is_webcam_active = False
        st.session_state.live_detection_active = False

    # Detection control button
    if detection_mode == "Single Image" and uploaded_file:
        detect_button = st.button("🔍 Analyze Image")
        if detect_button:
            st.session_state.is_detecting = True
    elif detection_mode == "Live Camera":
        if st.button("🎥 Start Live Detection" if not st.session_state.is_detecting else "🛑 Stop Live Detection"):
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

# ------------------------------------------------
# Main Content
# ------------------------------------------------
st.title("🔍 Smart Garbage Detection & Sorting Assistant")

# Handle different detection modes
if st.session_state.is_detecting:
    if st.session_state.live_detection_active:
        # Live camera detection
        live_streaming_detection(confidence_threshold, selected_classes)
    elif uploaded_file:
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
        - 📷 Live camera detection
        - 🖼️ Single image analysis
        - 🎯 Smart disposal recommendations
        - 🗑️ Multiple waste categories supported
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:
        1. **Choose** detection mode in the sidebar
        2. **Upload** an image or use live camera
        3. **Adjust** confidence threshold as needed
        4. **Select** specific classes to detect (optional)
        5. **Start detection** and follow the disposal instructions

        The system will automatically provide disposal guidance for detected items!
        """)

    # Display sample images or instructions
    st.info("👈 Choose your detection mode from the sidebar to get started!")