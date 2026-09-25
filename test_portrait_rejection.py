import cv2
import numpy as np

def is_valid_satellite_image(image_np):
    if len(image_np.shape) != 3 or image_np.shape[2] != 3:
        return False, "Invalid image shape. Must be a 3-channel RGB image."
    
    h, w, c = image_np.shape
    if h < 128 or w < 128:
        return False, f"Resolution too low ({w}x{h}). Minimum resolution is 128x128."
        
    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > 3.0:
        return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Image is too stretched."

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 1. Face detection (Frontal and Profile)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0 or len(profile_faces) > 0:
        return False, "Contains human face (portrait/selfie)."

    # 2. Local texture homogeneity (to reject ground-level photos with large objects/foregrounds)
    bh, bw = h // 4, w // 4
    block_vars = []
    for i in range(4):
        for j in range(4):
            block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_vars.append(cv2.Laplacian(block, cv2.CV_64F).var())
            
    block_vars = np.array(block_vars)
    mean_var = np.mean(block_vars)
    std_var = np.std(block_vars)
    cv = std_var / (mean_var + 1e-5)
    max_to_min = np.max(block_vars) / (np.min(block_vars) + 1e-5)
    
    # We reject if texture is highly non-homogeneous
    if cv > 0.95 or max_to_min > 100.0:
        return False, f"Non-homogeneous texture (CV = {cv:.2f}, Max/Min Ratio = {max_to_min:.1f}). likely a ground-level photograph."

    # 3. Overall Laplacian Variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 50:
        return False, f"Low detail (Laplacian Var = {lap_var:.2f})."
    if lap_var > 15000:
        return False, f"Unnaturally high contrast/noise (Laplacian Var = {lap_var:.2f})."

    # 4. Color space check
    reduced = (image_np // 8).astype(np.int32)
    flat_colors = reduced[:,:,0] * 1000000 + reduced[:,:,1] * 1000 + reduced[:,:,2]
    unique_colors = len(np.unique(flat_colors))
    if unique_colors < 150:
        return False, f"Too few colors ({unique_colors})."

    color_std = np.mean(np.std(image_np, axis=2))
    if color_std < 2.0:
        return False, f"Low color variance ({color_std:.2f})."

    # 5. Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    if edge_density < 0.015:
        return False, f"Too few edges ({edge_density:.4f})."
    if edge_density > 0.35:
        return False, f"Too many edges ({edge_density:.4f})."

    return True, "Valid satellite image."

# Generate simulated portrait
portrait = np.zeros((1024, 1024, 3), dtype=np.uint8)
bg = np.linspace(100, 200, 1024, dtype=np.uint8)
for i in range(1024):
    portrait[i, :, 0] = bg[i]
    portrait[i, :, 1] = bg[i] // 2
    portrait[i, :, 2] = 50
for i in range(300):
    for j in range(300):
        val = np.random.randint(50, 150)
        portrait[i, j] = [val // 2, val, val // 3]
        portrait[i, 1023 - j] = [val // 2, val, val // 3]
cv2.circle(portrait, (512, 512), 300, (180, 220, 250), -1)
cv2.circle(portrait, (400, 450), 30, (50, 50, 50), -1)
cv2.circle(portrait, (624, 450), 30, (50, 50, 50), -1)
mask = np.zeros((1024, 1024), dtype=np.uint8)
cv2.circle(mask, (512, 512), 350, 255, -1)
blurred = cv2.GaussianBlur(portrait, (35, 35), 0)
portrait = np.where(mask[:, :, None] == 255, blurred, portrait)

is_valid, msg = is_valid_satellite_image(portrait)
print(f"Validation Result: {is_valid} ({msg})")
