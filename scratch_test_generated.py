import cv2
import numpy as np
from utils.preprocessing import is_valid_satellite_image

def test_on_generated():
    img_paths = {
        "Selfie": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\selfie_test_image_1783497733018.png",
        "Street View": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\street_test_image_1783497745593.png"
    }
    
    for name, path in img_paths.items():
        img = cv2.imread(path)
        if img is None:
            print(f"Could not load {name} at {path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        is_valid, msg = is_valid_satellite_image(img_rgb)
        print(f"Image: {name} -> is_valid={is_valid}, message='{msg}'")
        
        # Print homogeneity details
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
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
        cv_ratio = std_var / (mean_var + 1e-5)
        max_to_min = np.max(block_vars) / (np.min(block_vars) + 1e-5)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"  Details: cv_ratio={cv_ratio:.4f}, max_to_min={max_to_min:.4f}, lap_var={lap_var:.4f}")

if __name__ == "__main__":
    test_on_generated()
