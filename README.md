# improved_YOLOv11s

🔥 Improved YOLOv11s for Fire and Smoke Detection

This repository contains the implementation of an **improved YOLOv11s architecture** optimized for **fire and smoke detection** on the D-Fire dataset.

---

🚀 Features
- Optimized anchor configuration for fire and smoke detection  
- Evaluated on the **D-Fire dataset**

---

📊 Results
| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|--------|------------|---------|----------|---------------|
| Improved YOLOv11s | 0.94 | 0.91 | 0.95 | 0.71 |

---

📁 Dataset
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
pip install ultralytics
requirements.txt

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


📄 License
This project is released under the MIT License.


