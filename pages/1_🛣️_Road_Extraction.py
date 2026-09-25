import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

from inference.predict import RoadExtractor
from inference.analytics import estimate_road_width, analyze_connectivity_and_quality, detect_damage
from inference.visualize import visualize_quality, visualize_width, create_overlay
from utils.preprocessing import validate_uploaded_image


BASE_CSS = """
:root{--glass-bg: rgba(255,255,255,0.04);--glass-border: rgba(255,255,255,0.10);--accent:#4facfe}
.stApp{background: linear-gradient(120deg,#061026 0%, #09213a 50%, #071022 100%);color:#e6eef8}
.glass-card{
    background:linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
    box-shadow: 8px 10px 30px rgba(2,6,23,0.6);
    border-radius:14px;
    padding:20px;
    border:1px solid var(--glass-border);
    transition:transform .18s ease; 
    text-align: center;
    height: 100%;
}
.glass-card:hover{transform:translateY(-6px)}
.stat-title{font-size: 14px; font-weight: 700; color: #9fb3d6; letter-spacing: 1px; margin-bottom: 15px; text-transform: uppercase;}
.stat-value{font-size: 38px; font-weight: 800; color: #4facfe;}
.stat-value-poor{font-size: 38px; font-weight: 800; color: #ffd28f;}
.stat-value-good{font-size: 38px; font-weight: 800; color: #8ef5a1;}
.legend{font-size: 16px; font-weight: bold; margin-top: 10px;}
"""

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "unet_road_extractor.pth")

@st.cache_resource
def load_extractor():
    return RoadExtractor(MODEL_PATH)

def main():
    st.set_page_config(page_title="Road Extraction", page_icon="🛣️", layout="wide")
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Controls")
    st.sidebar.markdown("Upload Satellite Image")
    uploaded = st.sidebar.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed")
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.markdown("### Toggle Visualization Layers")
    
    show_mask = st.sidebar.toggle("Show Raw Mask", value=True)
    show_width = st.sidebar.toggle("Show Width Map")
    show_quality = st.sidebar.toggle("Show Quality Map", value=True)
    show_damage = st.sidebar.toggle("Highlight Damage")

    try:
        extractor = load_extractor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    if uploaded is not None:
        is_valid, image_np, message = validate_uploaded_image(uploaded)
        if not is_valid:
            st.error(message)
            return

        with st.spinner("Analyzing image..."):
            mask = extractor.predict(image_np)
            avg_width, dist_map = estimate_road_width(mask)
            quality_data = analyze_connectivity_and_quality(mask)
            damage_mask = detect_damage(mask)

        # Visualization
        out = image_np.copy()
        
        if show_mask and not show_quality:
            out = create_overlay(out, mask, (255, 255, 255), 0.4)
            
        if show_width:
            width_hm = visualize_width(dist_map)
            alpha = 0.6
            road_idx = mask == 255
            out[road_idx] = cv2.addWeighted(out[road_idx], 1-alpha, width_hm[road_idx], alpha, 0)
            
        if show_quality or show_damage:
            conn = quality_data['connected_mask'] if show_quality else np.zeros_like(mask)
            disconn = quality_data['disconnected_mask'] if show_quality else np.zeros_like(mask)
            dmg = damage_mask if show_damage else np.zeros_like(mask)
            out = visualize_quality(out, conn, disconn, dmg)

        # Main Area
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📡 Original Satellite Image")
            st.image(image_np, use_container_width=True)
        with col2:
            st.markdown("### 🎯 Analytical Overlay")
            st.image(out, use_container_width=True)
            st.markdown("<p class='legend'>Legend: 🟩 Good/Connected | 🟧 Moderate/Disconnected | 🟥 Damaged</p>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2>📊 Extraction Statistics</h2>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        
        length = int(quality_data.get('total_length', 0))
        length_str = f"{length:,}<br>px"
        width_str = f"{avg_width:.1f}<br>px"
        qual = quality_data["quality"]
        if qual.lower() == "poor":
            qual_class = "stat-value-poor"
        elif qual.lower() == "good":
            qual_class = "stat-value-good"
        else:
            qual_class = "stat-value"
        cont_str = f"{quality_data['continuity_score'] * 100:.1f}<br>%"
        
        c1.markdown(f"<div class='glass-card'><div class='stat-title'>TOTAL ROAD LENGTH</div><div class='stat-value'>{length_str}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='glass-card'><div class='stat-title'>AVERAGE WIDTH</div><div class='stat-value'>{width_str}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><div class='stat-title'>OVERALL QUALITY</div><div class='{qual_class}' style='margin-top: 15px;'>{qual}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='glass-card'><div class='stat-title'>CONTINUITY SCORE</div><div class='stat-value'>{cont_str}</div></div>", unsafe_allow_html=True)

    else:
        st.markdown("<h2 style='text-align: center; margin-top: 20vh; color: #9fb3d6;'>Please upload an image from the sidebar to begin.</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
