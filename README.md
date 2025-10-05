# improved_YOLOv11s

# Improved YOLOv11s for Fire and Smoke Detection 🔥

This repository contains the implementation of **an improved YOLOv11s architecture** optimized for fire and smoke detection on the D-Fire dataset.

## 🚀 Features
- Adaptive backbone structure for lightweight efficiency
- Enhanced detection head for small object recognition
- Optimized anchor configuration

## 🧠 Model
The architecture is defined in [`yolo11.yaml`](./yolo11.yaml).  
You can train or evaluate the model as follows:

```bash
# Train
python train.py --data data/DFire.yaml --cfg models/yolo11s_custom.yaml --weights yolov11s.pt --epochs 250

# Detect
python detect.py --weights weights/best.pt --source data/sample_images/
