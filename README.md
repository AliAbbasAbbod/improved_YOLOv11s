# improved_YOLOv11s

# 🔥 Improved YOLOv11s for Fire and Smoke Detection

This repository contains the implementation of an **improved YOLOv11s architecture** optimized for **fire and smoke detection** on the D-Fire dataset.

---

## 🚀 Features
- Adaptive backbone structure for lightweight efficiency  
- Enhanced detection head for small object recognition  
- Optimized anchor configuration for fire and smoke detection  
- Evaluated on the **D-Fire dataset**

---

## 📊 Results
| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|------------|---------|----------|---------------|
| Improved YOLOv11s | 0.94 | 0.91 | 0.95 | 0.71 |

---

## 📁 Dataset
The **D-Fire dataset** used in this project is publicly available here:  
👉 [https://github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)

Download and organize it as follows:
data/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/

kotlin
Copy code

Then create the file `data/DFire.yaml`:
```yaml
train: ./data/train/images
val: ./data/val/images
test: ./data/test/images

nc: 2
names: ['fire', 'smoke']
⚙️ Installation
Clone this repository and install dependencies:

bash
Copy code
git clone https://github.com/AliAbbasAbbod/improved_YOLOv11s.git
cd improved_YOLOv11s

# Install Python libraries
pip install -r requirements.txt
requirements.txt

shell
Copy code
ultralytics>=8.0.200
torch>=2.1.0
torchvision>=0.16.0
opencv-python
numpy
pandas
matplotlib
🧠 Model Architecture
The YOLOv11s structure (improved) is defined in yolo11.yaml.
You can modify this file to adjust layers, channels, or detection heads.

🏋️ Training
You can train the model using the built-in Ultralytics interface
or by running the provided training script.

train.py

python
Copy code
from ultralytics import YOLO

# Load model structure
model = YOLO("yolo11.yaml")

# Train the model on D-Fire dataset
model.train(
    data="data/DFire.yaml",
    epochs=250,
    imgsz=640,
    batch=16,
    name="improved_yolov11s"
)
Run the training:

bash
Copy code
python train.py
Results (weights, metrics, logs) will be saved automatically under:

bash
Copy code
runs/detect/improved_yolov11s/
🔍 Detection / Inference
After training completes, test the model on new images:

detect.py

python
Copy code
from ultralytics import YOLO

# Load best weights
model = YOLO("runs/detect/improved_yolov11s/weights/best.pt")

# Run detection on test images
model.predict(
    source="data/test/images",
    conf=0.25,
    save=True
)
Run:

bash
Copy code
python detect.py
Output images with bounding boxes will appear in:

bash
Copy code
runs/predict/
📄 License
This project is released under the MIT License.


