import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_iou(preds, labels, threshold=0.5, smooth=1e-6):
    """
    Calculate the Intersection over Union (IoU) metric.
    preds: model predictions (probabilities)
    labels: ground truth masks
    threshold: threshold to binarize predictions
    """
    preds = (preds > threshold).float()
    
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_accuracy(preds, labels, threshold=0.5):
    """
    Calculate pixel-wise accuracy.
    """
    preds = (preds > threshold).float()
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return (correct / total).item()
