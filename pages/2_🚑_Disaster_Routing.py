import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

from inference.predict import RoadExtractor
from inference.analytics import detect_damage
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

def build_graph_from_skeleton(skeleton, damage_mask):
    # Find all road pixels
    y, x = np.where(skeleton > 0)
    points = list(zip(x, y))
    
    if not points:
        return nx.Graph(), None

    G = nx.Graph()
    # Add nodes
    for i, p in enumerate(points):
        G.add_node(i, pos=p)

    # KDTree for fast neighbor lookup (connect adjacent pixels)
    tree = KDTree(points)
    # Distance 1.5 connects 8-connected neighbors
    pairs = tree.query_pairs(1.5)
    
    for (i, j) in pairs:
        p1 = points[i]
        p2 = points[j]
        # Check if either point is in the damage mask
        # If it is, give it a massive weight to avoid it
        if damage_mask[p1[1], p1[0]] > 0 or damage_mask[p2[1], p2[0]] > 0:
            weight = 999999 # Impassable
        else:
            weight = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        G.add_edge(i, j, weight=weight)
        
    return G, points

def main():
    st.set_page_config(page_title="Disaster Routing", page_icon="🚑", layout="wide")
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
    
    st.markdown("# 🚑 Disaster Response Routing Planner")
    st.markdown("This unique module proves that our road extraction isn't just about pixels. We convert the extracted roads into a mathematical graph and calculate the **shortest safe route** for emergency vehicles, strictly avoiding roads identified as damaged.")

    try:
        extractor = load_extractor()
    except Exception as e:
        st.error("Model not found.")
        st.stop()

    st.sidebar.markdown("## ⚙️ Controls")
    uploaded = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg","png"])
    generate_btn = st.sidebar.button("Calculate Random Safe Route")

    if uploaded is not None:
        is_valid, image_np, message = validate_uploaded_image(uploaded)
        if not is_valid:
            st.error(message)
            return

        with st.spinner("Processing image and building road graph..."):
            mask = extractor.predict(image_np)
            damage_mask = detect_damage(mask)
            
            # Skeletonize for routing
            binary_mask = (mask > 128).astype(np.uint8)
            skeleton = skeletonize(binary_mask).astype(np.uint8) * 255
            
            G, points = build_graph_from_skeleton(skeleton, damage_mask)

        if G is None or G.number_of_nodes() == 0:
            st.error("No roads detected to build a graph.")
            return

        out_img = image_np.copy()
        
        # Dim the background image for better overlay visibility
        out_img = (out_img * 0.5).astype(np.uint8)

        # Draw all roads in faint gray
        for (i, j) in G.edges():
            p1 = points[i]
            p2 = points[j]
            cv2.line(out_img, p1, p2, (100, 100, 100), 1)
            
        # Draw damaged roads in red
        damage_y, damage_x = np.where(damage_mask > 0)
        for dx, dy in zip(damage_x, damage_y):
            cv2.circle(out_img, (dx, dy), 2, (255, 0, 0), -1)

        route_found = False
        if generate_btn:
            # Construct a safe subgraph containing only edges with weights < 999999 (avoiding damaged areas)
            safe_G = nx.Graph()
            for u, v, data in G.edges(data=True):
                if data.get('weight', 0) < 999999:
                    safe_G.add_edge(u, v, weight=data['weight'])
            
            # Find connected components in the safe subgraph that have at least 2 nodes
            components = [c for c in nx.connected_components(safe_G) if len(c) >= 2]
            
            # Fallback to full graph components if no safe component is large enough
            if not components:
                components = [c for c in nx.connected_components(G) if len(c) >= 2]
            
            if components:
                # Pick the largest connected component to find the most significant route
                largest_component = max(components, key=len)
                component_nodes = list(largest_component)
                
                # Choose start node randomly from the component
                start_node = np.random.choice(component_nodes)
                start_pos = np.array(points[start_node])
                
                # Choose end node as the furthest node in the same component
                distances = [np.linalg.norm(np.array(points[n]) - start_pos) for n in component_nodes]
                end_node = component_nodes[np.argmax(distances)]
                
                try:
                    path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
                    
                    # Check if path is valid (doesn't contain impassable weights)
                    path_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
                    if path_weight >= 999999:
                        st.warning("No safe path exists between the chosen points that avoids all damage.")
                    else:
                        # Draw path
                        for i in range(len(path)-1):
                            p1 = points[path[i]]
                            p2 = points[path[i+1]]
                            cv2.line(out_img, p1, p2, (0, 255, 0), 3) # Green route
                            
                        # Draw Start and End
                        cv2.circle(out_img, points[start_node], 8, (255, 255, 0), -1) # Cyan start
                        cv2.circle(out_img, points[end_node], 8, (255, 0, 255), -1)   # Magenta end
                        route_found = True
                except nx.NetworkXNoPath:
                    st.warning("No path exists between the points.")
            else:
                st.warning("The road network graph is too small or fragmented to calculate a route.")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(out_img, use_container_width=True)
            st.markdown("Legend: ⚪ Available Roads | 🔴 Damaged Areas | 🟢 Safe Route | 🟡 Start | 🟣 End")
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### Routing Analytics")
            st.metric("Graph Nodes (Intersections)", G.number_of_nodes())
            st.metric("Graph Edges (Road Segments)", G.number_of_edges())
            if route_found:
                st.success("Safe route successfully calculated!")
            else:
                st.info("Click 'Calculate Random Safe Route' in the sidebar to generate a path.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please upload an image to begin.")

if __name__ == "__main__":
    main()
