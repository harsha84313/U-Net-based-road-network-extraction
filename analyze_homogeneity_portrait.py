import os
import cv2
import numpy as np

def analyze_homogeneity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
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
    
    return mean_var, std_var, cv, max_to_min

# Generate a portrait simulation
portrait = np.zeros((1024, 1024, 3), dtype=np.uint8)
# blurred background (very low variance)
bg = np.linspace(100, 200, 1024, dtype=np.uint8)
for i in range(1024):
    portrait[i, :, 0] = bg[i]
    portrait[i, :, 1] = bg[i] // 2
    portrait[i, :, 2] = 50

# background foliage in the top corners (highly textured)
for i in range(300):
    for j in range(300):
        # random leaf pattern
        val = np.random.randint(50, 150)
        portrait[i, j] = [val // 2, val, val // 3]
        portrait[i, 1023 - j] = [val // 2, val, val // 3]

# large skin tone circle representing face in the center
cv2.circle(portrait, (512, 512), 300, (180, 220, 250), -1)
# smooth face details
cv2.circle(portrait, (400, 450), 30, (50, 50, 50), -1)
cv2.circle(portrait, (624, 450), 30, (50, 50, 50), -1)

# smooth out central face but keep leaves textured
mask = np.zeros((1024, 1024), dtype=np.uint8)
cv2.circle(mask, (512, 512), 350, 255, -1)
blurred = cv2.GaussianBlur(portrait, (35, 35), 0)
portrait = np.where(mask[:, :, None] == 255, blurred, portrait)

# Let's print metrics for the simulated portrait
mean_var, std_var, cv, max_to_min = analyze_homogeneity(portrait)
print(f"Simulated Portrait -> Mean Var: {mean_var:.1f}, Std Var: {std_var:.1f}, CV: {cv:.2f}, Max/Min Ratio: {max_to_min:.1f}")
