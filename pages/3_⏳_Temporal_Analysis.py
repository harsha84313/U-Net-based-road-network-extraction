import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

from inference.predict import RoadExtractor
from utils.preprocessing import validate_uploaded_image


BASE_CSS = """
:root{--glass-bg: rgba(255,255,255,0.04);--glass-border: rgba(255,255,255,0.10);--accent:#4facfe}
.stApp{background: linear-gradient(120deg,#061026 0%, #09213a 50%, #071022 100%);color:#e6eef8}
.glass-card{background:linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));box-shadow: 8px 10px 30px rgba(2,6,23,0.6);border-radius:14px;padding:20px;border:1px solid var(--glass-border);}
"""

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "unet_road_extractor.pth")

@st.cache_resource
def load_extractor():
    return RoadExtractor(MODEL_PATH)

def main():
    st.set_page_config(page_title="Temporal Analysis", page_icon="⏳", layout="wide")
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
    
    st.markdown("# ⏳ Temporal Change Detection")
    st.markdown("Upload 'Before' and 'After' satellite images of the same location. The system will extract both road networks and highlight newly constructed roads (Green) and destroyed/degraded roads (Red).")

    try:
        extractor = load_extractor()
    except Exception as e:
        st.error("Model not found.")
        st.stop()

    st.sidebar.markdown("## ⚙️ Upload Images")
    img_before = st.sidebar.file_uploader("Upload 'Before' Image", type=["jpg","png"])
    img_after = st.sidebar.file_uploader("Upload 'After' Image", type=["jpg","png"])

    if img_before and img_after:
        is_valid1, img1_np, message1 = validate_uploaded_image(img_before)
        if not is_valid1:
            st.error(message1)
            return

        try:
            if hasattr(img_after, "seek"):
                img_after.seek(0)
            with Image.open(img_after) as image_file:
                img2_np = np.array(image_file.convert("RGB"))
        except Exception as e:
            st.error(f"Failed to load 'After' image: {e}")
            return

        image1 = Image.fromarray(img1_np)
        image2 = Image.fromarray(img2_np)
        
        # Ensure same size for simple diff
        image2 = image2.resize(image1.size)
        img2_np = np.array(image2)

        with st.spinner("Extracting networks and computing temporal diff..."):
            mask1 = extractor.predict(img1_np)
            mask2 = extractor.predict(img2_np)
            
            # Binary masks
            bm1 = (mask1 > 128).astype(np.int32)
            bm2 = (mask2 > 128).astype(np.int32)
            
            # Difference
            diff = bm2 - bm1
            
            # newly constructed: diff == 1 (in bm2 but not bm1)
            # destroyed: diff == -1 (in bm1 but not bm2)
            
            new_roads = (diff == 1).astype(np.uint8) * 255
            destroyed_roads = (diff == -1).astype(np.uint8) * 255
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📅 Before")
            st.image(img1_np, use_container_width=True)
        with col2:
            st.markdown("### 📅 After")
            st.image(img2_np, use_container_width=True)
        with col3:
            st.markdown("### 🔍 Change Map")
            # Overlay on After image
            out = img2_np.copy()
            # Green for new
            out[new_roads == 255] = [0, 255, 0]
            # Red for destroyed
            out[destroyed_roads == 255] = [255, 0, 0]
            st.image(out, use_container_width=True)
            st.markdown("Legend: 🟩 New Construction | 🟥 Destroyed/Degraded")
            
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Change Statistics")
        st.write(f"**Newly Constructed Road Pixels:** {np.sum(new_roads==255):,}")
        st.write(f"**Destroyed/Degraded Road Pixels:** {np.sum(destroyed_roads==255):,}")
        st.markdown("</div>", unsafe_allow_html=True)
            
    else:
        st.info("Upload both a 'Before' and an 'After' image in the sidebar.")

if __name__ == "__main__":
    main()
