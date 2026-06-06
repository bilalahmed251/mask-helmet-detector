# Face Mask / Helmet Detection System
## Complete A-to-Z Deep Learning Project

### Project Overview
Binary classification using Transfer Learning (MobileNetV2)
- **Task 1**: Face Mask Detection (Mask / No Mask)
- **Task 2**: Helmet Detection (Helmet / No Helmet)
- **Real-time**: Webcam / Video detection using OpenCV
- **Deployment**: Streamlit Web App

---

## Project Structure
```
mask_helmet_detection/
├── dataset/                  # Put your dataset here
│   ├── mask/                 # For mask project
│   │   ├── train/
│   │   │   ├── with_mask/
│   │   │   └── without_mask/
│   │   └── val/
│   │       ├── with_mask/
│   │       └── without_mask/
│   └── helmet/               # For helmet project
│       ├── train/
│       │   ├── helmet/
│       │   └── no_helmet/
│       └── val/
│           ├── helmet/
│           └── no_helmet/
├── models/                   # Saved trained models
├── utils/
│   ├── data_loader.py        # Dataset loading + augmentation
│   ├── model_builder.py      # MobileNetV2 architecture
│   └── visualizer.py         # Plotting utilities
├── notebooks/
│   └── full_pipeline.ipynb   # Complete Jupyter Notebook
├── app/
│   └── streamlit_app.py      # Web deployment app
├── train.py                  # Main training script
├── evaluate.py               # Model evaluation script
├── realtime_detect.py        # Live webcam detection
├── predict.py                # Single image prediction
└── requirements.txt          # All dependencies
```

---

## Dataset Links (Kaggle)
- **Face Mask**: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- **Helmet**: https://www.kaggle.com/datasets/andrewmvd/helmet-detection

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Prepare dataset
Download from Kaggle and place in `dataset/` folder as shown above.

### Step 3 — Train the model
```bash
# Train mask detection
python train.py --task mask --epochs 20 --batch_size 32

# Train helmet detection
python train.py --task helmet --epochs 20 --batch_size 32
```

### Step 4 — Evaluate model
```bash
python evaluate.py --task mask
```

### Step 5 — Real-time webcam detection
```bash
python realtime_detect.py --task mask
```

### Step 6 — Launch web app
```bash
streamlit run app/streamlit_app.py
```

---

## Expected Results
- Validation Accuracy: **95–98%** using MobileNetV2
- Real-time FPS: **20–30 FPS** on standard CPU
