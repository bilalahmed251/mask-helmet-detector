"""
evaluate.py
===========
Comprehensive model evaluation — confusion matrix, classification
report, and sample prediction grid.

Usage:
    python evaluate.py --task mask
    python evaluate.py --task helmet

Author : Bilal Ahmed (231980028) — GIFT University
"""

import argparse
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from utils.data_loader import get_val_generator, TASK_CONFIG
from utils.visualizer  import (plot_confusion_matrix, plot_sample_predictions)


# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained detector")
    parser.add_argument("--task", type=str, default="mask",
                        choices=["mask", "helmet"])
    return parser.parse_args()


# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    config = TASK_CONFIG[args.task]

    print(f"\n{'='*55}")
    print(f"  EVALUATION  →  Task: {args.task.upper()}")
    print(f"{'='*55}")

    # ── 1. Load model ────────────────────────
    if not os.path.exists(config["model_path"]):
        print(f"❌  No saved model found at '{config['model_path']}'")
        print("   Run train.py first.")
        return

    print(f"Loading model from {config['model_path']} ...")
    model = tf.keras.models.load_model(config["model_path"])

    # ── 2. Validation generator ───────────────
    val_gen = get_val_generator(config["val_dir"])
    val_gen.reset()

    class_names = list(val_gen.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Validation samples: {val_gen.samples}\n")

    # ── 3. Predict all validation images ─────
    print("Running predictions on validation set...")
    preds_prob = model.predict(val_gen, verbose=1)
    preds_prob = preds_prob.flatten()

    y_true = val_gen.classes
    y_pred = (preds_prob >= 0.5).astype(int)

    # ── 4. Overall accuracy ───────────────────
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"\n✅  Validation Accuracy: {accuracy:.2f}%")

    # ── 5. Confusion matrix ───────────────────
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=f"models/{args.task}_confusion_matrix.png"
    )

    # ── 6. Sample predictions ─────────────────
    print("\nPreparing sample prediction grid...")
    val_gen.reset()
    sample_images, sample_labels = next(val_gen)

    sample_preds_prob = model.predict(sample_images, verbose=0).flatten()
    sample_preds = (sample_preds_prob >= 0.5).astype(int)

    plot_sample_predictions(
        sample_images, sample_labels.astype(int), sample_preds, class_names,
        n=12,
        save_path=f"models/{args.task}_sample_predictions.png"
    )

    # ── 7. Per-class accuracy ─────────────────
    print("\nPer-Class Accuracy:")
    print("─" * 30)
    for i, cls in enumerate(class_names):
        mask = y_true == i
        cls_acc = np.mean(y_pred[mask] == y_true[mask]) * 100
        print(f"  {cls:20s}: {cls_acc:.2f}%")
    print()


if __name__ == "__main__":
    main()
