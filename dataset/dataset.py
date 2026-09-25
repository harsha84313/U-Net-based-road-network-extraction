import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepGlobeDataset(Dataset):
    """
    Dataset loader for DeepGlobe Road Extraction.
    Assumes images are named as [ID]_sat.jpg and masks as [ID]_mask.png
    """
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith("_sat.jpg")]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")
        
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.images_dir, mask_name)
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image could not be read: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Handles inference cases where mask might not exist
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            # Binarize mask
            mask = (mask > 127).astype(np.float32)

        # Apply transformations (albumentations expects image, mask as kwargs)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask has channel dimension [1, H, W] for BCE loss compatibility
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        return image, mask

def get_transforms(is_train=True):
    """
    Returns albumentations transforms for training or validation.
    Resizes images to 256x256 for U-Net, normalizes the input, and adds augmentations during training.
    """
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

if __name__ == "__main__":
    # Smoke test
    test_dir = r"c:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT\dataset\train"
    if os.path.exists(test_dir):
        ds = DeepGlobeDataset(test_dir, transform=get_transforms(is_train=True))
        print(f"Dataset length: {len(ds)}")
        img, mask = ds[0]
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Mask shape: {mask.shape}, max: {mask.max()}, min: {mask.min()}")
