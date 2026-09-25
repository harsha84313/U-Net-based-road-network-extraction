import os
import cv2
import numpy as np

def test_on_valid_images():
    valid_dir = r"c:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT\dataset\valid"
    files = [f for f in os.listdir(valid_dir) if f.endswith("_sat.jpg")][:100]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    false_faces = 0
    failures = 0

    for f in files:
        path = os.path.join(valid_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0 or len(profile_faces) > 0:
            false_faces += 1
            print(f"False Face Detection in: {f}")
            
        # Check CV / Homogeneity
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
        
        if cv > 0.95 or max_to_min > 100.0:
            failures += 1
            print(f"Homogeneity failure in: {f} (CV = {cv:.2f}, Max/Min Ratio = {max_to_min:.1f})")

    print(f"Tested {len(files)} valid images:")
    print(f"  False Face Detections: {false_faces}")
    print(f"  Homogeneity Failures: {failures}")

if __name__ == "__main__":
    test_on_valid_images()
