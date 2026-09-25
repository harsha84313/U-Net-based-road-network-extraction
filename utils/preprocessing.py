"""
Image preprocessing utilities.

Keep preprocessing centralized so it is easy to change normalization,
resize strategy, or to add remote preprocessing pipelines.
"""

import cv2
import numpy as np
from PIL import Image


def load_and_resize_pil(pil_img, size=(256,256)):
    arr = np.array(pil_img.convert('RGB'))
    return cv2.resize(arr, size[::-1], interpolation=cv2.INTER_AREA)


def normalize_image(image_np):
    # Standard ImageNet-like normalization (placeholder)
    img = image_np.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


def binary_mask_to_bool(mask):
    """Convert 0/255 masks to boolean arrays."""
    return (mask > 127).astype(np.uint8)


def is_valid_satellite_image(image_np):
    """
    Validates if the input image is a valid satellite image based on:
    1. Size and dimensions
    2. Structural complexity (block Laplacian variance for non-homogeneous regions)
    3. Color diversity (unique colors, color variance)
    4. Edge density (structural content)
    
    Note: Satellite imagery can legitimately have large regions of earth/vegetation tones
    that match skin-tone color ranges. We rely instead on structural properties:
    - Portraits have sharp focus transitions (high block variance CV ratio)
    - Ground photos are taken close-up with uneven depth (high max/min block variance)
    - Satellite images show consistent terrain patterns (low block variance CV ratio)
    """
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        return False, "Invalid image shape. Must be a 3-channel RGB image."
    
    h, w, c = image_np.shape
    if h < 128 or w < 128:
        return False, f"Resolution too low ({w}x{h}). Minimum resolution is 128x128."
        
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 3.0:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Image is too stretched."

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Check detail / blur / flat regions (check early)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        return False, f"Low detail (Laplacian Var = {lap_var:.2f})."
    if lap_var > 15000:
        return False, f"Unnaturally high contrast/noise (Laplacian Var = {lap_var:.2f})."

    # Check structural edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    if edge_density < 0.015:
        return False, f"Too few edges ({edge_density:.4f})."
    if edge_density > 0.35:
        return False, f"Too many edges ({edge_density:.4f})."

    # Reduce color space to handle noise/compression artifacts
    reduced = (image_np // 8).astype(np.int32)
    flat_colors = reduced[:,:,0] * 1000000 + reduced[:,:,1] * 1000 + reduced[:,:,2]
    unique_colors = len(np.unique(flat_colors))
    if unique_colors < 150:
        return False, f"Too few colors ({unique_colors})."

    color_std = np.mean(np.std(image_np, axis=2))
    if color_std < 2.0:
        return False, f"Low color variance ({color_std:.2f})."

    # Reject highly non-homogeneous ground-level photos such as portraits, close-ups, or selfies
    # by detecting unnatural spikes in local Laplacian variance (sharp focus variations in faces).
    # Satellite imagery has relatively consistent local structure across the image.
    bh, bw = max(1, h // 4), max(1, w // 4)
    block_vars = []
    for i in range(4):
        for j in range(4):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            if block.size > 0:
                block_vars.append(cv2.Laplacian(block, cv2.CV_64F).var())
    if len(block_vars) >= 2:
        block_vars = np.array(block_vars)
        mean_var = np.mean(block_vars)
        std_var = np.std(block_vars)
        cv_ratio = std_var / (mean_var + 1e-5)
        max_to_min = np.max(block_vars) / (np.min(block_vars) + 1e-5)
        # These thresholds detect portraits/close-ups (high variance in local detail due to focus/depth)
        # Relaxed to accommodate natural terrain variation in satellite imagery
        if cv_ratio > 2.0 or max_to_min > 30.0:
            return False, "Invalid image. Please upload a satellite image."

    return True, "Valid satellite image."


def validate_uploaded_image(uploaded_file, expected_message="Invalid image. Please upload a satellite image."):
    """Validate an uploaded image and return RGB numpy data for downstream prediction."""
    if uploaded_file is None:
        return False, None, expected_message

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if hasattr(uploaded_file, "convert"):
            image_np = np.array(uploaded_file.convert("RGB"))
        else:
            with Image.open(uploaded_file) as image_file:
                image_np = np.array(image_file.convert("RGB"))
    except Exception:
        return False, None, expected_message

    is_valid, _ = is_valid_satellite_image(image_np)
    if not is_valid:
        return False, None, expected_message

    return True, image_np, "Valid satellite image."

