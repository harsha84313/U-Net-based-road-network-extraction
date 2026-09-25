import os
import cv2
import numpy as np

def test_face_detection_params():
    valid_dir = r"c:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT\dataset\valid"
    valid_files = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith("_sat.jpg")][:20]
    
    selfie_path = r"C:\Users\RAKSHITHADAS\.gemini\antigravity\brain\d1a164af-4836-4b77-a77e-eaf3a088102d\selfie_test_image_1783497733018.png"
    selfie_img = cv2.imread(selfie_path)
    selfie_gray = cv2.cvtColor(selfie_img, cv2.COLOR_BGR2GRAY) if selfie_img is not None else None
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    for min_neighbors in [4, 6, 8, 10, 12]:
        false_positives = 0
        for path in valid_files:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, min_neighbors)
            profile_faces = profile_cascade.detectMultiScale(gray, 1.1, min_neighbors)
            if len(faces) > 0 or len(profile_faces) > 0:
                false_positives += 1
                
        # Test on selfie
        detected_selfie = False
        if selfie_gray is not None:
            faces_selfie = face_cascade.detectMultiScale(selfie_gray, 1.1, min_neighbors)
            profile_selfie = profile_cascade.detectMultiScale(selfie_gray, 1.1, min_neighbors)
            if len(faces_selfie) > 0 or len(profile_selfie) > 0:
                detected_selfie = True
                
        print(f"minNeighbors={min_neighbors}:")
        print(f"  False Positives on 20 Satellite Images: {false_positives}")
        print(f"  Detected Selfie: {detected_selfie}")

if __name__ == "__main__":
    test_face_detection_params()
