import os
import cv2
import numpy as np

def test_flat_blocks():
    valid_dir = r"c:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT\dataset\valid"
    valid_files = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith("_sat.jpg")][:50]
    
    other_files = {
        "Selfie": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\selfie_test_image_1783497733018.png",
        "Street View": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\street_test_image_1783497745593.png",
        "Cat": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\cat_test_image_1783497809676.png",
        "Car": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\car_test_image_1783497823444.png"
    }
    
    def get_flat_block_fraction(gray, grid_size=8, threshold=15.0):
        h, w = gray.shape
        bh, bw = h // grid_size, w // grid_size
        flat_count = 0
        total_blocks = grid_size * grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                if block.size > 0:
                    var = cv2.Laplacian(block, cv2.CV_64F).var()
                    if var < threshold:
                        flat_count += 1
        return flat_count / total_blocks

    print("--- Valid Satellite Images ---")
    fractions = []
    for path in valid_files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frac = get_flat_block_fraction(gray)
        fractions.append(frac)
        
    print(f"Valid Satellite flat block fraction: min={min(fractions):.4f}, max={max(fractions):.4f}, mean={np.mean(fractions):.4f}")
    
    print("\n--- Non-Satellite Images ---")
    for name, path in other_files.items():
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frac = get_flat_block_fraction(gray)
        print(f"{name}: flat block fraction = {frac:.4f}")

if __name__ == "__main__":
    test_flat_blocks()
