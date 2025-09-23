# Store camera mode in session state
    if 'camera_mode' not in st.session_state:
        st.session_state.camera_mode = camera_mode
    else:
        st.session_state.camera_mode = camera_modeimport cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import time
from huggingface_hub import hf_hub_download
import threading
import queue

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

def camera_input_detection(conf_threshold, selected_classes):
    """Camera detection using Streamlit's camera_input with auto-refresh"""
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform detection.")
        return

    st.info("📷 **Live Camera Detection Active** - Take photos to analyze objects")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use camera_input which properly requests browser permissions
        camera_image = st.camera_input(
            "Take a photo for detection", 
            key="live_camera",
            help="This will request camera permission from your browser"
        )
        
        if camera_image is not None:
            # Process the captured image
            image = Image.open(camera_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process with YOLO
            processed_frame, detected_classes = process_frame_with_detection(
                image_cv, conf_threshold, selected_classes
            )
            
            # Display processed image
            st.image(processed_frame, channels="BGR", caption="Detection Results")
            
            # Store results in session state for the sidebar
            st.session_state.detected_classes = detected_classes
            
        else:
            st.info("📸 Click the camera button above to take a photo and detect objects")
    
    with col2:
        # Display current detection results
        if hasattr(st.session_state, 'detected_classes') and st.session_state.detected_classes:
            st.subheader("🎯 Latest Detection:")
            display_detection_messages(st.session_state.detected_classes)
        else:
            st.info("No recent detections")
            
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh camera", value=False)
        if auto_refresh:
            time.sleep(2)
            st.rerun()

def continuous_camera_detection(conf_threshold, selected_classes):
    """Alternative approach using camera_input with continuous refresh"""
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform detection.")
        return

    st.info("🔴 **Continuous Camera Detection** - Photos will be taken automatically")
    
    # Initialize session state for continuous mode
    if "photo_counter" not in st.session_state:
        st.session_state.photo_counter = 0
    
    # Create placeholders
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use a unique key for each photo to force refresh
        camera_key = f"continuous_camera_{st.session_state.photo_counter}"
        
        camera_image = st.camera_input(
            "Continuous Detection Camera",
            key=camera_key,
            help="Camera will automatically refresh for continuous detection"
        )
        
        if camera_image is not None:
            # Process the image
            image = Image.open(camera_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            processed_frame, detected_classes = process_frame_with_detection(
                image_cv, conf_threshold, selected_classes
            )
            
            st.image(processed_frame, channels="BGR")
            st.session_state.detected_classes = detected_classes
            
            # Auto-increment counter and refresh
            st.session_state.photo_counter += 1
            time.sleep(1)  # Small delay
            st.rerun()
    
    with col2:
        if hasattr(st.session_state, 'detected_classes') and st.session_state.detected_classes:
            st.subheader("🎯 Live Detection:")
            display_detection_messages(st.session_state.detected_classes)
        else:
            st.info("Waiting for detection...")

def webcam_html_detection():
    """HTML/JavaScript based webcam access (experimental)"""
    st.info("🌐 **Browser-based Camera Access**")
    
    html_code = """
    <div id="camera-container">
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #4CAF50;"></video>
        <br><br>
        <button onclick="capturePhoto()" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">📸 Capture Photo</button>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
    </div>
    
    <script>
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    
    // Request camera access
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
            console.log("Camera access granted");
        })
        .catch(function(err) {
            console.error("Camera access denied: ", err);
            alert("Please allow camera access to use this feature");
        });
    
    function capturePhoto() {
        ctx.drawImage(video, 0, 0, 640, 480);
        let imageData = canvas.toDataURL('image/jpeg');
        
        // Send image data to Streamlit (this would need additional implementation)
        console.log("Photo captured");
        alert("Photo captured! (Implementation for sending to Streamlit needed)");
    }
    </script>
    """
    
    st.components.v1.html(html_code, height=600)
    st.warning("⚠️ This is an experimental feature. The captured photos are not yet integrated with YOLO detection.")

# Alternative: Simple periodic camera capture
def periodic_camera_detection(conf_threshold, selected_classes):
    """Periodic camera capture for quasi-real-time detection"""
    if not st.session_state.yolo_model:
        st.error("YOLO model is not loaded. Cannot perform detection.")
        return

    st.info("⏱️ **Periodic Camera Detection** - Camera refreshes every few seconds")
    
    # Auto-refresh mechanism
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Refresh interval control
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 3)
        
        # Check if it's time to refresh
        current_time = time.time()
        if current_time - st.session_state.last_refresh >= refresh_interval:
            st.session_state.last_refresh = current_time
            st.rerun()
        
        # Camera input
        camera_image = st.camera_input("Detection Camera", key=f"periodic_{int(st.session_state.last_refresh)}")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            processed_frame, detected_classes = process_frame_with_detection(
                image_cv, conf_threshold, selected_classes
            )
            
            st.image(processed_frame, channels="BGR")
            st.session_state.detected_classes = detected_classes
    
    with col2:
        # Countdown timer
        time_since_refresh = current_time - st.session_state.last_refresh
        time_until_next = max(0, refresh_interval - time_since_refresh)
        st.metric("Next refresh in", f"{time_until_next:.1f}s")
        
        # Detection results
        if hasattr(st.session_state, 'detected_classes') and st.session_state.detected_classes:
            st.subheader("🎯 Current Detection:")
            display_detection_messages(st.session_state.detected_classes)
        else:
            st.info("No objects detected")
        
        # Progress bar for refresh
        progress = (refresh_interval - time_until_next) / refresh_interval
        st.progress(progress)

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

    # Camera detection mode selection
    st.markdown("### 📷 Camera Detection Modes")
    camera_mode = st.radio(
        "Choose camera detection mode:",
        [
            "📸 Manual Photo Capture",
            "⏱️ Periodic Auto-Capture", 
            "🔄 Continuous Refresh",
            "🌐 Browser Camera (Experimental)"
        ],
        help="Different approaches to camera-based detection"
    )

    # Real-time webcam button (updated text)
    if st.button("Start Camera Detection 📷" if not st.session_state.is_webcam_active else "Stop Camera Detection 🛑"):
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
    # Get camera mode from sidebar
    camera_mode = st.session_state.get('camera_mode', '📸 Manual Photo Capture')
    
    if camera_mode == "📸 Manual Photo Capture":
        st.title("📸 Manual Camera Detection")
        camera_input_detection(confidence_threshold, selected_classes)
    elif camera_mode == "⏱️ Periodic Auto-Capture":
        st.title("⏱️ Periodic Camera Detection") 
        periodic_camera_detection(confidence_threshold, selected_classes)
    elif camera_mode == "🔄 Continuous Refresh":
        st.title("🔄 Continuous Camera Detection")
        continuous_camera_detection(confidence_threshold, selected_classes)
    elif camera_mode == "🌐 Browser Camera (Experimental)":
        st.title("🌐 Browser-based Camera Detection")
        webcam_html_detection()
    
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
        - **Multiple camera detection modes** with browser permission support
        - **Manual photo capture** for precise detection  
        - **Periodic auto-capture** for quasi-real-time detection
        - **Image analysis** for uploaded photos
        - **Smart disposal recommendations**
        - **Multiple waste categories** supported
        - **Adjustable confidence threshold**
        - **Class filtering** options
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:

        **For Camera Detection:**
        1. **Select a camera mode** in the sidebar:
           - 📸 **Manual**: Click to take photos for detection
           - ⏱️ **Periodic**: Automatic photo capture at intervals  
           - 🔄 **Continuous**: Rapid refresh for near real-time
           - 🌐 **Browser**: Direct browser camera access
        2. Click **"Start Camera Detection"** 📷
        3. **Allow camera access** when browser prompts
        4. Point camera at objects and follow the mode instructions

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