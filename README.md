# U-Net Based Road Network Extraction & Connectivity Analysis

![Road Extraction Dashboard Header](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## Overview
This project performs end-to-end Road Network Extraction from Satellite Images using a custom **U-Net architecture**. Once the road is segmented, advanced Computer Vision heuristics estimate the **road width**, classify the **overall quality**, and build a connected graph to analyze **connectivity** and identify **damaged/broken** road segments.

The entire pipeline is presented through an interactive and beautifully designed **Streamlit Dashboard** utilizing custom Glassmorphism CSS.

## Key Features
- **DeepGlobe Dataset Handling**: PyTorch `Dataset` with **Albumentations** (resizing, jitter, affine transforms).
- **Core U-Net Segmentation**: Implemented entirely from scratch in PyTorch. Trained using a custom `BCEDiceLoss()`.
- **Advanced Post-Processing Analytics**:
  - `estimate_road_width()`: Euclidean Distance Transforms.
  - `analyze_connectivity()`: Skimage skeletonization and OpenCV Connected Components to identify broken segments.
  - `detect_damage()`: Corner detection heuristics mapping sharp breaks.
- **Glassmorphism UI**: Interactive Streamlit app with multiple overlay toggles (Show width, show damage, raw mask, quality).

---

## 🚀 Setup Instructions

### 1. Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Configuration
Ensure the extracted DeepGlobe data resides in:
```text
project_root/
  └─ dataset/
      ├─ train/      <-- ([ID]_sat.jpg and [ID]_mask.png)
      └─ valid/
```

---

## 💻 How to Run

### 1. Train the Model
To commence training and generate the necessary `.pth` weight file:
```bash
python training/train.py --epochs 10 --batch_size 8 --lr 0.001
```
*Outputs: `training/unet_road_extractor.pth` and `utils/training_history.png`.*

### 2. Launch the Interactive Dashboard
To fire up the modern inference UI:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501`, upload an image from the dataset, and toggle the different analytical overlays to see the magic!

---

## ☁️ Google Colab Guide

If you want to train your model using **Google Colab's Free GPU**, follow this step-by-step guide to get your project correctly loaded and running.

### 1. Upload your Project
Because Colab runs in a cloud environment (`/content/`), it doesn't automatically have access to your local files. 

1. On your local computer, open your project folder (`C:\Users\RAKSHITHADAS\OneDrive\Documents\MAJOR PROJECT`).
2. Select everything inside the folder, right-click, and select **Compress to ZIP file**. Name it `major_project.zip`.
3. Open your Google Colab Notebook. Click on the **Folder Icon (📁)** in the left sidebar.
4. Click the **Upload File icon (⬆️)** and upload `major_project.zip`. Wait for the upload wheel at the bottom left to finish.

### 2. Extract and Set Up
In your Colab Notebook, create a new code cell and run the following commands to unzip the files and enter the directory:

```bash
# Unzip the uploaded files into a folder
!unzip -q /content/major_project.zip -d /content/

# Change the working directory to your project folder
# Note: If your zip file was created by right-clicking the folder, it will create a folder named 'MAJOR PROJECT'
%cd "/content/MAJOR PROJECT"

# Install dependencies
!pip install -r requirements.txt
```

### 3. Train the Model
Make sure your Colab runtime is set to **GPU** (Runtime -> Change runtime type -> T4 GPU). Then, create a new cell and run your training script:

```bash
!python training/train.py --epochs 10 --batch_size 8 --lr 0.001
```

### 4. Running the Streamlit Dashboard in Colab (Optional)
Streamlit apps run on a local server, which means Colab's notebook can't show it directly. To view your Streamlit app in Colab, use `localtunnel` to create a public web link. 

Run this in a new cell:
```bash
!npm install localtunnel
!streamlit run app.py & npx localtunnel --port 8501
```
*(This will generate a link `https://xxxx.loca.lt/`. Click it, and if it asks for a "Tunnel Password", your password is the external IP of the Colab instance, which you can find by running `!curl ipv4.icanhazip.com` in another cell).*
