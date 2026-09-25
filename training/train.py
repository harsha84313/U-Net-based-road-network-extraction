import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import DeepGlobeDataset, get_transforms
from model.unet import UNet
from utils.metrics import BCEDiceLoss, calculate_iou, calculate_accuracy

def train_model(train_dir, valid_dir, epochs=10, batch_size=8, learning_rate=1e-3, device="cuda", quick_test=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Datasets and DataLoaders
    train_ds = DeepGlobeDataset(train_dir, transform=get_transforms(is_train=True))
    valid_ds = DeepGlobeDataset(valid_dir, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Metrics history
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    
    best_val_loss = float("inf")
    save_path = os.path.join(os.path.dirname(__file__), "unet_road_extractor.pth")

    for epoch in range(epochs):
        model.train()
        train_loss, train_iou = 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
        for i, (images, masks) in enumerate(loop):
            if quick_test and i >= 2:
                break
            
            images = images.to(device)
            masks = masks.to(device).float()

            # Forward pass
            preds = model(images)
            loss = criterion(preds, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_iou += calculate_iou(preds, masks)
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss, val_iou = 0, 0
        with torch.no_grad():
            loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [VALID]")
            for i, (images, masks) in enumerate(loop):
                if quick_test and i >= 2:
                    break
                
                images = images.to(device)
                masks = masks.to(device).float()

                preds = model(images)
                loss = criterion(preds, masks)

                val_loss += loss.item()
                val_iou += calculate_iou(preds, masks)

        val_loss /= len(valid_loader)
        val_iou /= len(valid_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

    # Plot metrics
    plot_training_history(history)
    print("Training Complete!")

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # IoU plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_iou"], label="Train IoU")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.title("IoU Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.legend()

    plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "training_history.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training history plot saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train U-Net for Road Extraction")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick_test", action="store_true", help="Run only 2 batches per epoch for rapid testing")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "dataset", "train")
    valid_dir = os.path.join(base_dir, "dataset", "valid")

    train_model(train_dir, valid_dir, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, quick_test=args.quick_test)
