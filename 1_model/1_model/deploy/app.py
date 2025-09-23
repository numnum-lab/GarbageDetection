import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
# ลบ streamlit_webrtc และ WebRTC-related imports ออก
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
import torch
# ลบ functools.partial ออก (ใช้แค่สำหรับ webcam)
# from functools import partial
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
# Sidebar
# ------------------------------------------------
    st.set_page_config(layout="wide")
#Config
    cfg = load_yolo_model()
    PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
    WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']



    st.sidebar.title("Settings")



#Create a menu bar
    menu = ["Picture","Webcam"]
    choice = st.sidebar.selectbox("Input type",menu)
    #Put slide to adjust tolerance
    TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.5,0.01)
    st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

#Infomation section 
    st.sidebar.title("trast")

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
# Main Content - เพิ่ม webcam functionality แบบง่าย
# ------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        ### 🗂️ Garbage Detection Using YOLO
        This project helps people sort garbage more easily.

        **Features:**
        - Real-time webcam capture
        - Image analysis
        - Smart disposal recommendations
        - Multiple waste categories supported
        """)

    with col2:
        st.write("""
        ### 📖 How to Use:
        1. **Upload** an image or **activate webcam**
        2. **Adjust** confidence threshold as needed
        3. **Select** specific classes to detect (optional)
        4. **Start detection** and follow the disposal instructions

        The system will automatically provide disposal guidance for detected items!
        """)