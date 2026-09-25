import cv2
import numpy as np

def create_overlay(image, mask, color, alpha=0.5):
    """
    Overlays a binary mask onto an RGB image with a specified color.
    image: RGB array
    mask: Binary array (0 or 255)
    color: tuple (R, G, B) e.g., (0, 255, 0)
    """
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    
    # Apply alpha blending
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Where mask is 0, preserve original image
    result = np.where(mask[..., None] == 255, overlay, image)
    return result

def visualize_quality(image, connected_mask, disconnected_mask, damage_mask):
    """
    Creates a combined visualization where:
    Green: Connected/Good roads
    Yellow: Disconnected/Moderate roads
    Red: Damaged/Broken points
    """
    viz_img = image.copy()
    
    viz_img = create_overlay(viz_img, connected_mask, (0, 255, 0), alpha=0.6) # Green
    viz_img = create_overlay(viz_img, disconnected_mask, (255, 255, 0), alpha=0.6) # Yellow
    
    # Dilate damage mask slightly so red points are visible
    if damage_mask.sum() > 0:
        kernel = np.ones((5,5),np.uint8)
        visible_damage = cv2.dilate(damage_mask, kernel, iterations=1)
        viz_img = create_overlay(viz_img, visible_damage, (255, 0, 0), alpha=0.9) # Red
        
    return viz_img

def visualize_width(dist_map):
    """
    Converts a distance map to a beautiful heatmap.
    """
    dist_map_norm = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX)
    colormap = cv2.applyColorMap(dist_map_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Black out background where distance is 0
    colormap[dist_map == 0] = 0
    
    return cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
