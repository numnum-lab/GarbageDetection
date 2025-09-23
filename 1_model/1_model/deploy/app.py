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

if "webcam_placeholder" not in st.session_state:
    st.session_state.webcam_placeholder = None

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

def process_frame_with_detection(frame, conf_threshold, selected_classes):
    """Process a single frame with YOLO detection"""
    if not st.session_state.yolo_model:
        return frame, []

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

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
        
        # Choose color based on class
        if yolo_classes[class_ids[i]] == "battery":
            color = (0, 0, 255)  # Red
        elif yolo_classes[class_ids[i]] == "biological":
            color = (0, 255, 0)  # Green
        elif yolo_classes[class_ids[i]] in ["cardboard", "glass", "metal", "paper", "plastic"]:
            color = (0, 255, 255)  # Yellow
        elif yolo_classes[class_ids[i]] in ["clothes", "shoes"]:
            color = (255, 255, 0)  # Cyan
        else:
            color = (128, 128, 128)  # Gray
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, detected_classes

def image_detection(uploaded_file, conf_threshold, selected_classes):
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform image detection.")
        return

    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    processed_frame, detected_classes = process_frame_with_detection(image_cv, conf_threshold, selected_classes)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(processed_frame, channels="BGR")
    with col2:
        display_detection_messages(detected_classes)

def real_time_detection(conf_threshold, selected_classes):
    """Real-time webcam detection with YOLO"""
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform real-time detection.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera permissions.")
        return

    # Create placeholders for video and detection results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
    with col2:
        detection_placeholder = st.empty()

    # Control buttons
    stop_button = st.button("Stop Real-time Detection 🛑", key="stop_realtime")
    
    # FPS counter
    fps_placeholder = st.empty()
    frame_count = 0
    start_time = time.time()

    try:
        while st.session_state.is_detecting and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break

            # Process frame with YOLO
            processed_frame, detected_classes = process_frame_with_detection(
                frame, conf_threshold, selected_classes
            )

            # Convert BGR to RGB for Streamlit
            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update video display
            video_placeholder.image(display_frame, channels="RGB", use_column_width=True)
            
            # Update detection results
            with detection_placeholder.container():
                if detected_classes:
                    st.subheader("🎯 Live Detection:")
                    display_detection_messages(detected_classes)
                else:
                    st.info("No objects detected")

            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                fps_placeholder.metric("FPS", f"{fps:.1f}")
                frame_count = 0
                start_time = time.time()

            # Small delay to prevent overwhelming the system
            time.sleep(0.03)  # ~30 FPS max

    except Exception as e:
        st.error(f"Error during real-time detection: {e}")
    finally:
        cap.release()
        st.session_state.is_detecting = False
        st.session_state.is_webcam_active = False

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

    # Real-time webcam button
    if st.button("Start Real-time Detection 📷" if not st.session_state.is_webcam_active else "Stop Real-time Detection 🛑"):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        st.session_state.is_detecting = st.session_state.is_webcam_active

    # Image detection button
    detect_button = st.button(
        "Detect in Image 🖼️",
        disabled=(not uploaded_file),
    )

    if detect_button and uploaded_file:
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
if st.session_state.is_webcam_active and st.session_state.is_detecting:
    st.title("🔴 Real-time Object Detection")
    st.info("Real-time detection is active. The model is running continuously on your webcam feed.")
    real_time_detection(confidence_threshold, selected_classes)
    
elif uploaded_file and st.session_state.is_detecting:
    st.title("🖼️ Image Object Detection")
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension in ["jpg", "jpeg", "png"]:
        st.info("Detecting objects in uploaded image...")
        image_detection(uploaded_file, confidence_threshold, selected_classes)
        st.session_state.is_detecting = False
    else:
        st.warning("Only image files (jpg, jpeg, png) are supported")
        st.session_state.is_detecting = False
        
else:
    st.title("Smart Garbage Detection & Sorting Assistant")
    st.info("Choose an option: Upload an image for detection or start real-time webcam detection.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🗂️ Garbage Detection Using YOLO
        This project helps people sort garbage more easily.

        **Features:**
        - **Real-time webcam detection** with live YOLO processing
        - **Image analysis** for uploaded photos
        - **Smart disposal recommendations**
        - **Multiple waste categories** supported
        - **Adjustable confidence threshold**
        - **Class filtering** options
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:

        **For Real-time Detection:**
        1. Click **"Start Real-time Detection"** 📷
        2. Allow camera access when prompted
        3. Point camera at objects to detect
        4. View live detection results and disposal instructions

        **For Image Detection:**
        1. **Upload** an image using the file uploader
        2. Click **"Detect in Image"** 
        3. View detection results and disposal guidance

        **Settings:**
        - **Adjust confidence threshold** for detection sensitivity
        - **Select specific classes** to focus on certain objects
        """)

    # Display sample images or instructions
    with st.expander("💡 Tips for Better Detection"):
        st.write("""
        - **Good lighting** improves detection accuracy
        - **Clear view** of objects works best
        - **Avoid overlapping** items when possible
        - **Adjust confidence threshold** if getting too many/few detections
        - **Use class filtering** to focus on specific waste types
        """)