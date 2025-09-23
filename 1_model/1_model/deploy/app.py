import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import tempfile
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
        repo_id = "Numgfsdf/garbage-detection-model"  # เปลี่ยนเป็น repo ของคุณ
        filename = "my_model.pt"                      # ตรวจสอบว่ามีไฟล์นี้จริง

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

# Function for live object detection using webcam
def live_streaming(conf_threshold, selected_classes):
    stframe = st.empty()
    message_container = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(
            "Error: Could not access the webcam. Please make sure your webcam is working."
        )
        return

    try:
        while st.session_state.get("is_detecting", False) and st.session_state.get(
            "is_webcam_active", False
        ):
            ret, frame = cap.read()

            if not ret:
                st.warning("Warning: Failed to read frame from the webcam. Retrying...")
                continue

            try:
                results = st.session_state.yolo_model.predict(source=frame, conf=conf_threshold)
                detections = results[0]

                # Extract bounding boxes, confidence scores, and class IDs
                boxes = (
                    detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else []
                )
                confs = (
                    detections.boxes.conf.cpu().numpy() if len(detections) > 0 else []
                )
                class_ids = (
                    detections.boxes.cls.cpu().numpy().astype(int)
                    if len(detections) > 0
                    else []
                )

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

                # Draw bounding boxes and labels on the frame
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                # Display the frame in Streamlit
                stframe.image(frame, channels="BGR")
                
                # Display detection messages
                with message_container.container():
                    display_detection_messages(detected_classes)

            except Exception as e:
                st.error(f"Error during model prediction: {str(e)}")

    finally:
        # Ensure resources are properly released
        cap.release()
        cv2.destroyAllWindows()

def video_streaming(uploaded_file, conf_threshold, selected_classes):
    stframe = st.empty()
    message_container = st.empty()
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    
    # Keep track of all detected classes throughout the video
    all_detected_classes = set()

    while cap.isOpened() and st.session_state.is_detecting:
        ret, frame = cap.read()
        if not ret:
            break

        results = st.session_state.yolo_model.predict(source=frame, conf=conf_threshold)
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

        # Add to cumulative detected classes
        all_detected_classes.update(detected_classes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        stframe.image(frame, channels="BGR")
        
        # Display current detection messages
        with message_container.container():
            display_detection_messages(list(all_detected_classes))

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:
    st.title("Object Detection Settings ⚙️")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.2)
    st.session_state.confidence_threshold = confidence_threshold

    selected_classes = st.multiselect("Select classes for object detection", yolo_classes)

    uploaded_file = st.file_uploader(
        "Upload an image or video 📤",
        type=["jpg", "png", "jpeg", "mp4", "avi", "mov"],
    )

    # Webcam functionality
    if st.button("Use Webcam 📷" if not st.session_state.is_webcam_active else "Stop Webcam 🛑"):
        st.session_state.is_webcam_active = not st.session_state.is_webcam_active
        st.session_state.is_detecting = st.session_state.is_webcam_active

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

# ------------------------------------------------
# Main Content
# ------------------------------------------------
if st.session_state.is_detecting:
    if st.session_state.is_webcam_active:
        st.info("🔴 Live webcam detection active")
        live_streaming(confidence_threshold, selected_classes)
            
    elif uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension in ["jpg", "jpeg", "png"]:
            st.info("Detecting objects in image...")
            image_detection(uploaded_file, confidence_threshold, selected_classes)
        elif file_extension in ["mp4", "avi", "mov"]:
            st.info("Processing video...")
            video_streaming(uploaded_file, confidence_threshold, selected_classes)
        else:
            st.warning("Unsupported file format")
else:
    st.title("Smart Garbage Detection & Sorting Assistant")
    st.info("Upload an image/video or activate webcam for object detection.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🗂️ Garbage Detection Using YOLO
        This project helps people sort garbage more easily.

        **Features:**
        - Real-time webcam capture
        - Image analysis
        - Video processing
        - Smart disposal recommendations
        - Multiple waste categories supported
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:
        1. **Upload** an image/video or **activate webcam**
        2. **Adjust** confidence threshold as needed
        3. **Select** specific classes to detect (optional)
        4. **Start detection** and follow the disposal instructions

        The system will automatically provide disposal guidance for detected items!
        """)