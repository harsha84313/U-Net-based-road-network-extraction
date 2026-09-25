import os
import cv2
import numpy as np

def analyze_top_vs_bottom():
    valid_dir = r"c:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT\dataset\valid"
    valid_files = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith("_sat.jpg")][:20]
    
    other_files = {
        "Selfie": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\selfie_test_image_1783497733018.png",
        "Street View": r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\street_test_image_1783497745593.png"
    }
    
    print("--- Valid Satellite Images ---")
    for path in valid_files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split into top 25% and bottom 75%
        top_part = gray[:h//4, :]
        bottom_part = gray[h//4:, :]
        
        top_var = cv2.Laplacian(top_part, cv2.CV_64F).var()
        bottom_var = cv2.Laplacian(bottom_part, cv2.CV_64F).var()
        ratio = top_var / (bottom_var + 1e-5)
        
        print(f"File: {os.path.basename(path)} -> top_var={top_var:.2f}, bottom_var={bottom_var:.2f}, ratio={ratio:.4f}")
        
    print("\n--- Non-Satellite Images ---")
    for name, path in other_files.items():
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        top_part = gray[:h//4, :]
        bottom_part = gray[h//4:, :]
        
        top_var = cv2.Laplacian(top_part, cv2.CV_64F).var()
        bottom_var = cv2.Laplacian(bottom_part, cv2.CV_64F).var()
        ratio = top_var / (bottom_var + 1e-5)
        
        print(f"Name: {name} -> top_var={top_var:.2f}, bottom_var={bottom_var:.2f}, ratio={ratio:.4f}")

if __name__ == "__main__":
    analyze_top_vs_bottom()
