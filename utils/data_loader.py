"""
utils/data_loader.py
====================
Dataset loading, augmentation, and preprocessing for
Face Mask / Helmet Detection project.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # MobileNetV2 standard input
BATCH_SIZE  = 32


# ─────────────────────────────────────────────
# Task config  (class names differ per task)
# ─────────────────────────────────────────────
TASK_CONFIG = {
    "mask": {
        "train_dir"  : "dataset/mask/train",
        "val_dir"    : "dataset/mask/val",
        "class_names": ["with_mask", "without_mask"],
        "model_path" : "models/mask_model.h5",
    },
    "helmet": {
        "train_dir"  : "dataset/helmet/train",
        "val_dir"    : "dataset/helmet/val",
        "class_names": ["helmet", "no_helmet"],
        "model_path" : "models/helmet_model.h5",
    },
}


# ─────────────────────────────────────────────
# 1. Augmented training generator
# ─────────────────────────────────────────────
def get_train_generator(train_dir: str, batch_size: int = BATCH_SIZE):
    """
    Returns an augmented ImageDataGenerator for training.
    Augmentations applied:
        - Random horizontal flip
        - Zoom ±20 %
        - Rotation ±15 °
        - Width / height shift ±10 %
        - Shear ±10 %
        - MobileNetV2 preprocessing (scale to -1 … 1)
    """
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
    )

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",   # binary → sigmoid output
        shuffle=True,
    )
    return generator


# ─────────────────────────────────────────────
# 2. Validation generator  (no augmentation)
# ─────────────────────────────────────────────
def get_val_generator(val_dir: str, batch_size: int = BATCH_SIZE):
    """
    Returns a plain ImageDataGenerator for validation/test.
    Only preprocessing is applied — no augmentation.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,    # False → class order preserved for metrics
    )
    return generator


# ─────────────────────────────────────────────
# 3. Load a single image for inference
# ─────────────────────────────────────────────
def load_image_for_inference(image_path: str) -> np.ndarray:
    """
    Loads one image from disk, resizes and preprocesses it
    so it can be passed directly to model.predict().

    Returns:
        numpy array of shape (1, 224, 224, 3)
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)   # add batch dim
    return img_array


# ─────────────────────────────────────────────
# 4. Dataset stats helper
# ─────────────────────────────────────────────
def print_dataset_info(task: str):
    """Prints class distribution for a given task."""
    config = TASK_CONFIG[task]
    print(f"\n{'='*50}")
    print(f"  Dataset Info — Task: {task.upper()}")
    print(f"{'='*50}")

    for split in ["train_dir", "val_dir"]:
        split_path = config[split]
        split_name = "TRAIN" if "train" in split else "VAL"
        print(f"\n  [{split_name}]  {split_path}")

        if not os.path.exists(split_path):
            print(f"    ⚠  Path not found — create it first!")
            continue

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"    {cls:20s} →  {count} images")

    print()
