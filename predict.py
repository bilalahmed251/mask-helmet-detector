"""
predict.py
==========
Predict mask/helmet status for a single image.

Usage:
    python predict.py --task mask   --image path/to/face.jpg
    python predict.py --task helmet --image path/to/rider.jpg

Author : Bilal Ahmed (231980028) — GIFT University
"""

import argparse
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from utils.data_loader import load_image_for_inference, TASK_CONFIG


# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Predict on a single image")
    parser.add_argument("--task",  type=str, default="mask",
                        choices=["mask", "helmet"])
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    return parser.parse_args()


# ─────────────────────────────────────────────
def predict_image(model_path: str, image_path: str, class_names: list):
    """
    Loads model + image, runs inference, returns result dict.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess
    img_array = load_image_for_inference(image_path)

    # Predict
    prob = model.predict(img_array, verbose=0)[0][0]
    pred_idx   = int(prob >= 0.5)
    pred_label = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else 1.0 - prob

    return {
        "label"     : pred_label,
        "confidence": float(confidence),
        "raw_prob"  : float(prob),
        "class_idx" : pred_idx,
    }


# ─────────────────────────────────────────────
def show_result(image_path: str, result: dict, task: str):
    """Display image with prediction overlaid."""
    img = Image.open(image_path).convert("RGB")

    # Color: green = safe/compliant, red = violation
    is_good = (task == "mask"   and result["label"] == "with_mask") or \
              (task == "helmet" and result["label"] == "helmet")

    color = "green" if is_good else "red"
    icon  = "✅" if is_good else "❌"

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"{icon}  {result['label'].replace('_', ' ').upper()}\n"
        f"Confidence: {result['confidence']*100:.1f}%",
        fontsize=14,
        fontweight="bold",
        color=color,
        pad=12,
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    config = TASK_CONFIG[args.task]

    if not os.path.exists(args.image):
        print(f"❌  Image not found: {args.image}")
        return

    if not os.path.exists(config["model_path"]):
        print(f"❌  Model not found: {config['model_path']}")
        print("   Run train.py first.")
        return

    result = predict_image(config["model_path"], args.image, config["class_names"])

    print(f"\n{'='*45}")
    print(f"  Task      : {args.task.upper()}")
    print(f"  Image     : {args.image}")
    print(f"  Prediction: {result['label']}")
    print(f"  Confidence: {result['confidence']*100:.2f}%")
    print(f"  Raw prob  : {result['raw_prob']:.4f}")
    print(f"{'='*45}\n")

    show_result(args.image, result, args.task)


if __name__ == "__main__":
    main()
