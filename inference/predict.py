import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet import UNet

class RoadExtractor:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print("WARNING: Model path not found. Initializing with random weights for demonstration.")
        self.model.eval()
        
        # Inference transforms
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, image_np, threshold=0.05):
        """
        image_np: RGB image array
        returns: binary mask (0 or 255) array of shape (H, W)
        """
        original_shape = image_np.shape[:2]
        
        # Preprocess
        augmented = self.transform(image=image_np)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            prob_mask = output.squeeze().cpu().numpy()
            
        # Postprocess
        binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
        
        # Resize back to original
        binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return binary_mask
