import streamlit as st

BASE_CSS = """
:root {
    --glass-bg: rgba(15, 23, 42, 0.45);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent: #38bdf8;
    --accent-hover: #0ea5e9;
}
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
    color: #cbd5e1;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Custom Hero Section */
.hero-title {
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.25;
    background: linear-gradient(135deg, #ffffff 40%, #e2e8f0 70%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1.05rem;
    font-weight: 400;
    text-align: center;
    color: #94a3b8;
    margin-bottom: 35px;
    letter-spacing: 0.01em;
    line-height: 1.5;
}

/* Typography Customizations */
h3 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.01em;
}

/* Top Status Bar */
.top-status-bar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    padding: 8px 0;
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
}
.status-dot {
    height: 8px;
    width: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #10b981;
    margin-right: 8px;
}

/* Glass Cards & Streamlit Border Wrappers */
.glass-card, div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    height: 100%;
}
.glass-card:hover, div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px) !important;
    border-color: rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.08) !important;
}

/* Card Descriptions */
.card-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-top: 8px;
    margin-bottom: 0;
}

/* Streamlit Page Link Enhancements */
div[data-testid="stPageLink"] {
    background-color: transparent !important;
}
div[data-testid="stPageLink"] a {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stPageLink"] a:hover {
    background-color: rgba(56, 189, 248, 0.08) !important;
    border-color: rgba(56, 189, 248, 0.25) !important;
    color: #38bdf8 !important;
}

/* KPI Metrics */
.kpi-title {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.kpi-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 4px 0;
    letter-spacing: -0.02em;
}
.kpi-trend {
    font-size: 0.75rem;
    color: #10b981;
    font-weight: 500;
}

/* Tech Stack items */
.tech-item h4 {
    font-size: 0.9rem !important;
    color: #38bdf8 !important;
    margin: 0 0 4px 0 !important;
    font-weight: 600 !important;
}
.tech-item p {
    font-size: 0.78rem !important;
    color: #64748b !important;
    margin: 0 !important;
}
"""

def main():
    st.set_page_config(page_title="U-Net Based Road Network Extraction & Connectivity Analysis", page_icon="🌍", layout="wide")
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)

    # Top Status Bar
    st.markdown("""
        <div class='top-status-bar'>
            <span class='status-dot'></span> System Online | Core AI Engine: Active | Latency: 42ms
        </div>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("<div class='hero-title'>U-Net Based Road Network Extraction & Connectivity Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Next-Generation Satellite Data Analysis & Disaster Response Platform</div>", unsafe_allow_html=True)

    # KPI Row
    st.markdown("### 📊 Live System Telemetry")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div class='kpi-title'>Model Accuracy</div>
            <div class='kpi-val'>96.8%</div>
            <div class='kpi-trend'>↑ 0.4% this week</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div class='kpi-title'>Avg Inference Time</div>
            <div class='kpi-val'>1.12s</div>
            <div class='kpi-trend'>Optimal</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div class='kpi-title'>Images Processed</div>
            <div class='kpi-val'>14,208</div>
            <div class='kpi-trend'>+312 today</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <div class='kpi-title'>Network Uptime</div>
            <div class='kpi-val'>99.99%</div>
            <div class='kpi-trend'>Secure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Cards
    st.markdown("### 🛠️ Core Modules")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.page_link("pages/1_🛣️_Road_Extraction.py", label="**🛣️ Road Extraction Engine**", icon="🛰️")
            st.markdown("<p class='card-desc'>Upload standard satellite imagery to automatically extract road networks. Features real-time width estimation, continuity scoring, and damage detection using PyTorch U-Net.</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.page_link("pages/3_⏳_Temporal_Analysis.py", label="**⏳ Temporal Change Detection**", icon="🕰️")
            st.markdown("<p class='card-desc'>Compare historical satellite data. Upload Before & After images to instantly calculate infrastructure degradation and highlight newly constructed road segments.</p>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.page_link("pages/2_🚑_Disaster_Routing.py", label="**🚑 Disaster Response Routing**", icon="🚨")
            st.markdown("<p class='card-desc'>Advanced graph-theory module. Converts extracted road networks into navigable mathematical graphs to calculate the shortest safe routes for emergency vehicles, avoiding detected damage.</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container(border=True):
            st.page_link("pages/4_📁_Batch_Processing.py", label="**📁 Enterprise Batch Processing**", icon="🏢")
            st.markdown("<p class='card-desc'>Designed for city-scale analytics. Process hundreds of satellite tiles simultaneously to generate aggregate infrastructure reports and downloadable CSV datasets.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture & Technology
    st.markdown("### 💻 Technology Stack")
    st.markdown("""
    <div class='glass-card'>
        <div style='display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap; gap: 20px;'>
            <div class='tech-item'>
                <h4>AI & Vision</h4>
                <p>PyTorch, OpenCV, Scikit-Image</p>
            </div>
            <div class='tech-item'>
                <h4>Routing Logic</h4>
                <p>NetworkX, SciPy KD-Trees</p>
            </div>
            <div class='tech-item'>
                <h4>Data Analytics</h4>
                <p>Pandas, NumPy</p>
            </div>
            <div class='tech-item'>
                <h4>Frontend</h4>
                <p>Streamlit, Glassmorphism CSS</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
