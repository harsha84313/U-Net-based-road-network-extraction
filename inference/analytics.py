import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

def estimate_road_width(mask):
    """
    Estimates the width of the roads using Distance Transform.
    mask: binary image (0 or 255)
    returns: average width in pixels, distance_map
    """
    if mask.max() == 255:
        mask = mask // 255
        
    dist_transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    
    # Consider only pixels inside the road (mask == 1)
    road_pixels = dist_transform[mask == 1]
    
    if len(road_pixels) == 0:
        return 0, dist_transform
        
    # Average width is roughly 2 * average distance to nearest background pixel
    avg_width = np.mean(road_pixels) * 2
    return avg_width, dist_transform

def analyze_connectivity_and_quality(mask):
    """
    Converts road mask into a graph to find connected vs disconnected components,
    and returns metrics to classify quality.
    """
    if mask.max() == 255:
        binary_mask = mask.astype(np.uint8) // 255
    else:
        binary_mask = mask.astype(np.uint8)
        
    # Skeletonize the road to 1-pixel width branches
    skeleton = skeletonize(binary_mask).astype(np.uint8)
    
    # Build graph using NetworkX based on neighbor adjacency 
    # To keep it fast, we will analyze connected components via OpenCV first
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter out very small components (noise)
    min_area = 50
    valid_components = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
    
    connected_roads_mask = np.zeros_like(binary_mask)
    disconnected_roads_mask = np.zeros_like(binary_mask)
    
    # If there is a massive main road component, consider it connected
    if len(valid_components) > 0:
        main_component_idx = max(valid_components, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        connected_roads_mask[labels == main_component_idx] = 1
        
        for i in valid_components:
            if i != main_component_idx:
                disconnected_roads_mask[labels == i] = 1
                
    # Quality Heuristic
    # Ratio of main component area to total road area
    total_area = sum(stats[i, cv2.CC_STAT_AREA] for i in valid_components) if valid_components else 0
    main_area = stats[main_component_idx, cv2.CC_STAT_AREA] if valid_components else 0
    continuity_score = (main_area / total_area) if total_area > 0 else 0
    
    if continuity_score > 0.85:
        quality = "Good"
    elif continuity_score > 0.50:
        quality = "Moderate"
    else:
        quality = "Poor"
        
    total_length = np.sum(skeleton) # rough approximation in pixels
        
    return {
        "connected_mask": connected_roads_mask * 255,
        "disconnected_mask": disconnected_roads_mask * 255,
        "quality": quality,
        "continuity_score": continuity_score,
        "num_segments": len(valid_components),
        "total_length": total_length
    }

def detect_damage(mask):
    """
    Finds sharp endings or gaps that likely indicate road damage using morphological operations.
    Returns a mask with highlighted damage points.
    """
    if mask.max() == 255:
        binary_mask = mask.astype(np.uint8) // 255
    else:
        binary_mask = mask.astype(np.uint8)

    skeleton = skeletonize(binary_mask).astype(np.uint8) * 255
    
    # Use Harris Corner detection on the skeleton to find endpoints
    dst = cv2.cornerHarris(np.float32(skeleton), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    damage_mask = np.zeros_like(mask)
    # Threshold for an endpoint
    damage_mask[dst > 0.1 * dst.max()] = 255 
    
    return damage_mask
