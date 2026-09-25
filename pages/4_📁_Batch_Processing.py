import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd

from inference.predict import RoadExtractor
from inference.analytics import estimate_road_width, analyze_connectivity_and_quality
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
    st.set_page_config(page_title="Batch Analytics", page_icon="📁", layout="wide")
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
    
    st.markdown("# 📁 Enterprise Batch Processing")
    st.markdown("Upload multiple satellite images to run city-scale analytics. The system will process all tiles sequentially and generate a comprehensive CSV report.")

    try:
        extractor = load_extractor()
    except Exception as e:
        st.error("Model not found.")
        st.stop()

    uploaded_files = st.file_uploader("Upload Satellite Images", type=["jpg","png"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Start Batch Processing"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                is_valid, img_np, message = validate_uploaded_image(file)
                if not is_valid:
                    st.error(f"{message} (File: {file.name})")
                    return
                
                mask = extractor.predict(img_np)
                avg_width, _ = estimate_road_width(mask)
                quality_data = analyze_connectivity_and_quality(mask)
                
                results.append({
                    "Filename": file.name,
                    "Total Length (px)": quality_data.get('total_length', 0),
                    "Average Width (px)": round(avg_width, 2),
                    "Continuity Score (%)": round(quality_data['continuity_score'] * 100, 2),
                    "Overall Quality": quality_data['quality']
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Batch processing complete!")
            
            df = pd.DataFrame(results)
            st.markdown("### Batch Results")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV Report",
                csv,
                "batch_analytics_report.csv",
                "text/csv"
            )
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Aggregate Summary")
            col1, col2 = st.columns(2)
            col1.metric("Total Analyzed Length", f"{df['Total Length (px)'].sum():,} px")
            col2.metric("Mean Continuity Score", f"{df['Continuity Score (%)'].mean():.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
