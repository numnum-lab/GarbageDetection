import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel
import base64
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
if "capture_key" not in st.session_state:
    st.session_state.capture_key = 0

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
        st.subheader("🎯 Detection Results:")
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

def create_camera_interface():
    """Create HTML/JavaScript interface for camera access"""
    
    camera_html = f"""
    <div style="text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px; margin: 10px 0;">
        <h3>📹 Camera Interface</h3>
        <video id="video" width="640" height="480" autoplay style="border: 2px solid #ddd; border-radius: 10px;"></video>
        <br><br>
        <button id="startBtn" onclick="startCamera()" style="background: #ff4b4b; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
            🚀 Start Camera
        </button>
        <button id="captureBtn" onclick="captureFrame()" style="background: #00cc88; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
            📸 Capture & Analyze
        </button>
        <button id="stopBtn" onclick="stopCamera()" style="background: #888; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
            🛑 Stop Camera
        </button>
        <br><br>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
        <div id="status" style="margin-top: 10px; font-weight: bold;">📱 Click "Start Camera" to begin</div>
    </div>

    <script>
    let video = document.getElementById('video');
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let stream = null;
    let capturing = false;

    async function startCamera() {{
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{ 
                video: {{ 
                    width: 640, 
                    height: 480,
                    facingMode: 'environment'  // Try to use back camera on mobile
                }} 
            }});
            video.srcObject = stream;
            document.getElementById('status').innerHTML = '🟢 Camera Active - Click "Capture & Analyze" to detect objects';
            document.getElementById('startBtn').style.background = '#888';
            document.getElementById('captureBtn').style.background = '#00cc88';
            document.getElementById('stopBtn').style.background = '#ff4b4b';
        }} catch (err) {{
            console.error('Error accessing camera: ', err);
            document.getElementById('status').innerHTML = '❌ Camera access denied. Please allow camera permissions and refresh the page.';
        }}
    }}

    function captureFrame() {{
        if (stream && !capturing) {{
            capturing = true;
            ctx.drawImage(video, 0, 0, 640, 480);
            let imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send image data to Streamlit
            window.parent.postMessage({{
                type: 'captured_frame',
                data: imageData,
                timestamp: Date.now()
            }}, '*');
            
            document.getElementById('status').innerHTML = '🔄 Processing image...';
            
            setTimeout(() => {{
                capturing = false;
                document.getElementById('status').innerHTML = '🟢 Ready for next capture';
            }}, 2000);
        }}
    }}

    function stopCamera() {{
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            video.srcObject = null;
            document.getElementById('status').innerHTML = '🔴 Camera stopped';
            document.getElementById('startBtn').style.background = '#ff4b4b';
            document.getElementById('captureBtn').style.background = '#888';
            document.getElementById('stopBtn').style.background = '#888';
        }}
    }}

    // Auto-capture for continuous detection (optional)
    let autoCaptureInterval = null;
    
    function toggleAutoCaptureMode() {{
        if (autoCaptureInterval) {{
            clearInterval(autoCaptureInterval);
            autoCaptureInterval = null;
            document.getElementById('status').innerHTML = '🟢 Manual mode - Click capture to analyze';
        }} else {{
            autoCaptureInterval = setInterval(() => {{
                if (stream && !capturing) {{
                    captureFrame();
                }}
            }}, 3000); // Capture every 3 seconds
            document.getElementById('status').innerHTML = '🔄 Auto-capture mode - Analyzing every 3 seconds';
        }}
    }}
    </script>
    """
    
    return camera_html

def image_detection(uploaded_file, model, conf_threshold, selected_classes):
    """Process uploaded image"""
    if isinstance(uploaded_file, str) and uploaded_file.startswith('data:image'):
        # Handle base64 image data from camera
        header, data = uploaded_file.split(',', 1)
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data))
    else:
        # Handle uploaded file
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
    
    # Camera mode selection
    st.subheader("📹 Detection Mode")
    detection_mode = st.radio(
        "Choose detection method:",
        ["📸 Image Upload", "🎥 Live Camera"],
        index=1 if st.session_state.camera_active else 0
    )
    
    if detection_mode == "🎥 Live Camera":
        st.session_state.camera_active = True
    else:
        st.session_state.camera_active = False
    
    # Instructions
    with st.expander("📖 How to use Camera"):
        st.markdown("""
        ### 🎯 Camera Instructions:
        1. **Click "Start Camera"** - Browser will ask for camera permission
        2. **Allow camera access** when prompted
        3. **Point camera at objects** you want to detect
        4. **Click "Capture & Analyze"** to detect objects in the current frame
        5. **Use "Stop Camera"** when finished
        
        ### 🔧 Troubleshooting:
        - **No camera prompt?** Try refreshing the page
        - **Camera blocked?** Check browser settings and allow camera access
        - **Blurry image?** Ensure good lighting and stable camera
        - **No detection?** Lower the confidence threshold
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
    st.info("🎥 **Camera Detection Mode** - Use the camera interface below to capture and analyze images!")
    
    # Display camera interface
    camera_interface = create_camera_interface()
    st.components.v1.html(camera_interface, height=650)
    
    # Add JavaScript to handle captured frames
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'captured_frame') {
            // Store the captured image data
            sessionStorage.setItem('captured_frame', event.data.data);
            sessionStorage.setItem('capture_timestamp', event.data.timestamp);
            
            // Trigger a rerun by clicking a hidden button
            const button = document.querySelector('[data-testid="baseButton-secondary"]');
            if (button) button.click();
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Add a hidden button to trigger rerun
    if st.button("🔄 Process Captured Image", key="process_capture", type="secondary"):
        st.rerun()
    
    # Check for captured frame data
    st.markdown("""
    <script>
    const capturedFrame = sessionStorage.getItem('captured_frame');
    if (capturedFrame) {
        // Display processing message
        document.write('<div style="background: #e1f5fe; padding: 10px; border-radius: 5px; margin: 10px 0;"><strong>🔄 Processing captured image...</strong></div>');
    }
    </script>
    """, unsafe_allow_html=True)

elif uploaded_file:
    st.info("📸 **Image Detection Mode** - Analyzing uploaded image...")
    image_detection(uploaded_file, yolo_model, confidence_threshold, selected_classes)

else:
    # Welcome screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 Welcome to Smart Garbage Detection!
        
        This advanced AI system helps you sort waste properly using object detection.
        
        ### ✨ Features:
        - **Live camera detection** with browser camera access
        - **Image analysis** for uploaded photos  
        - **Smart disposal recommendations** for detected items
        - **Color-coded classification** system
        - **High-performance YOLOv11** model
        
        ### 🎯 How to Use:
        1. **Camera Mode**: Select "Live Camera" and use the camera interface
        2. **Upload Mode**: Upload an image using the file uploader
        3. **Adjust Settings**: Fine-tune confidence threshold and select specific classes
        4. **Follow Instructions**: Get instant disposal guidance for detected objects
        """)
        
        # Quick start guide
        st.success("""
        💡 **Quick Start for Camera:**
        1. Select "Live Camera" in the sidebar
        2. Click "Start Camera" in the interface
        3. Allow camera access when prompted by browser
        4. Point camera at objects and click "Capture & Analyze"
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
    
    /* Hide the process button initially */
    button[kind="secondary"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)