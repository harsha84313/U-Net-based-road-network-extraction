"""
Visualization helpers shared across pages.

Provides Plotly wrappers and small utilities for rendering confusion matrices,
segmentation previews, and heatmaps. Designed to be small and easily extended.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_confusion_matrix(cm, labels=['bg','road']):
    cm = np.array(cm)
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues'))
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
    return fig


def segmentation_preview(image_np, mask_np, alpha=0.6):
    """Create an RGB overlay preview. image and mask as numpy arrays."""
    import cv2
    overlay = image_np.copy()
    color = (0,255,0)
    mask_bool = mask_np > 127
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 1-alpha, np.array(color,dtype=np.uint8), alpha, 0)
    return overlay


def plot_timeseries(history_dict):
    import plotly.graph_objects as go
    fig = go.Figure()
    if 'train_loss' in history_dict and 'val_loss' in history_dict:
        fig.add_trace(go.Scatter(x=history_dict['epoch'], y=history_dict['train_loss'], name='train_loss'))
        fig.add_trace(go.Scatter(x=history_dict['epoch'], y=history_dict['val_loss'], name='val_loss'))
    if 'iou' in history_dict:
        fig.add_trace(go.Scatter(x=history_dict['epoch'], y=history_dict['iou'], name='IoU', yaxis='y2'))
    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Loss', yaxis2=dict(overlaying='y', side='right', title='IoU'))
    return fig


def plot_ai_confidence_distribution(confidences):
    fig = px.histogram(confidences, nbins=40, title='AI Confidence Distribution')
    return fig
