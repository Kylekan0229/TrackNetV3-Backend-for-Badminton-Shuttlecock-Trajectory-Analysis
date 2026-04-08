# TrackNetV3 Backend for Badminton Shuttlecock Trajectory Analysis

This repository contains the modified TrackNetV3-based backend used in the Final Year Project:

> **An iPhone App for Badminton Shuttlecock Trajectory Analysis**

The backend performs shuttlecock detection from badminton rally videos and exports CSV tracking results for import into the iPhone analysis application.

---

# Overview

This backend is responsible for:

- Shuttlecock detection using TrackNetV3
- Optional trajectory repair using InpaintNet
- Dual-camera timestamp alignment
- CSV export for Camera A / Camera B
- Combined stereo CSV generation for later analysis
- Optional predicted video rendering

This repository is the **PC-side processing component** of the full badminton analysis system.

---

# Repository Structure

```text
new_TrackNetV3/
├── ckpts/                      # Model checkpoints (.pt files)
│   ├── TrackNet_best.pt
│   └── InpaintNet_best.pt
│
├── predict_dual_modified.py   # Main dual-camera inference script
├── predict_modified.py        # Single-video inference
├── predict.py                 # Original TrackNetV3 inference
│
├── dataset.py
├── model.py
├── test.py
├── train.py
│
├── utils/
├── figure/
│
├── requirements.txt
└── README.md
```

---

# Environment Setup

Recommended Python version:

```text
Python 3.10+
```

Create environment:

## Conda

```bash
conda create -n tracknet python=3.10
conda activate tracknet
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch installation is required separately, please follow the official PyTorch installation guide.

---

# Download Pretrained Model Weights

Due to GitHub file size limitations, the `.pt` checkpoint files are **not included** in this repository.

Download the pretrained weights here:

### Google Drive Link
https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view

---

# Place Model Weights in ckpts Folder

After downloading, place the `.pt` files into:

```text
ckpts/
├── TrackNet_best.pt
└── InpaintNet_best.pt
```

> **Important:**  
> The inference scripts assume the checkpoint files are located inside the `ckpts/` folder.

---

# Input Video Requirements

Prepare two badminton rally videos:

### Camera A
- Main review camera
- Used as primary calibrated review path in iPhone app

### Camera B
- Assistant / ultra-wide camera
- Used for additional geometric reasoning

Both videos should:

- Record the same rally
- Be approximately time-synchronised
- Preferably use same FPS
- Preferably use same resolution

Example:

```text
camA.mp4
camB.mp4
```

---

# How to Run Dual-Camera Prediction

Run:

```bash
python predict_dual_modified.py \
  --video_file_a camA.mp4 \
  --video_file_b camB.mp4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir pred_result_dual
```

---

# What This Command Does

The script will:

1. Load TrackNetV3 checkpoint  
2. Load InpaintNet checkpoint  
3. Run shuttle detection on Camera A  
4. Run shuttle detection on Camera B  
5. Export Camera A CSV  
6. Export Camera B CSV  
7. Generate combined stereo-aligned CSV  
8. Save metadata JSON files  

---

# Output Files

After execution, the output folder will contain:

```text
pred_result_dual/
├── camA_ball.csv
├── camB_ball.csv
├── camA_meta.json
├── camB_meta.json
├── camA__camB_stereo.csv
└── camA__camB_stereo_meta.json
```

---

# Import into iPhone App

The generated CSV files can be imported into the iPhone application for:

- Court calibration
- Shuttle trajectory overlay
- Landing analysis
- Shot segmentation
- Rule-based shot classification
- Interactive rally review

---

# Optional Arguments

## Output Prediction Videos

```bash
python predict_dual_modified.py \
  --video_file_a camA.mp4 \
  --video_file_b camB.mp4 \
  --tracknet_file ckpts/TrackNet_best.pt \
  --inpaintnet_file ckpts/InpaintNet_best.pt \
  --save_dir pred_result_dual \
  --output_video
```

## Force CPU Inference

```bash
--device cpu
```

# Reference

TrackNetV3: https://github.com/qaz812345/TrackNetV3
