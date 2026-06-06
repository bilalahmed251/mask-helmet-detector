"""
utils/visualizer.py
===================
All plotting and visualization utilities.

Functions:
    - plot_training_history()  → accuracy & loss curves
    - plot_confusion_matrix()  → seaborn heatmap
    - plot_sample_predictions() → grid of images with labels
    - plot_class_distribution() → bar chart

Author : Bilal Ahmed (231980028) — GIFT University
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


# ─────────────────────────────────────────────
# 1. Training History Curves
# ─────────────────────────────────────────────
def plot_training_history(history, save_path="models/training_curves.png"):
    """
    Plots accuracy and loss curves for train and validation sets.

    Args:
        history   : Keras History object returned by model.fit()
        save_path : Where to save the PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # ── Accuracy ────────────────────────────
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",  color="#2196F3", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",    color="#4CAF50", linewidth=2)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ── Loss ────────────────────────────────
    axes[1].plot(history.history["loss"],     label="Train Loss",  color="#F44336", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",    color="#FF9800", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# 2. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path="models/confusion_matrix.png"):
    """
    Plots a seaborn confusion matrix heatmap.

    Args:
        y_true      : Ground truth labels (0/1)
        y_pred      : Predicted labels (0/1)
        class_names : List like ['with_mask', 'without_mask']
        save_path   : Where to save the PNG
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label",    fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved → {save_path}")

    # Also print classification report
    print("\nClassification Report:")
    print("─" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))


# ─────────────────────────────────────────────
# 3. Sample Predictions Grid
# ─────────────────────────────────────────────
def plot_sample_predictions(images, y_true, y_pred, class_names,
                            n=12, save_path="models/sample_predictions.png"):
    """
    Shows a grid of images with predicted vs actual labels.
    Green title = correct, Red title = wrong.

    Args:
        images      : Numpy array of images (already preprocessed)
        y_true      : Ground truth array
        y_pred      : Predicted array
        class_names : Class label list
        n           : Number of images to show (default 12)
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Sample Predictions  (Green = Correct | Red = Wrong)",
                 fontsize=13, fontweight="bold")

    axes = axes.flatten() if rows > 1 else [axes] * cols

    for i in range(n):
        # De-normalize for display (MobileNetV2 uses -1 to 1)
        img = images[i].copy()
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)

        pred_label   = class_names[int(y_pred[i])]
        actual_label = class_names[int(y_true[i])]
        correct      = y_pred[i] == y_true[i]

        axes[i].imshow(img)
        axes[i].set_title(
            f"Pred: {pred_label}\nActual: {actual_label}",
            color="green" if correct else "red",
            fontsize=9,
        )
        axes[i].axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Sample predictions saved → {save_path}")


# ─────────────────────────────────────────────
# 4. Class Distribution Bar Chart
# ─────────────────────────────────────────────
def plot_class_distribution(generator, class_names,
                             save_path="models/class_distribution.png"):
    """
    Plots bar chart showing class-wise image counts from a generator.
    """
    labels = generator.classes
    counts = [np.sum(labels == i) for i in range(len(class_names))]

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"][:len(class_names)]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(class_names, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Class distribution saved → {save_path}")
