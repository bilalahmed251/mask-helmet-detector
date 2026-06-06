"""
train.py
========
Main training script for Face Mask / Helmet Detection.

Usage:
    python train.py --task mask    --epochs 20 --batch_size 32
    python train.py --task helmet  --epochs 20 --batch_size 32

Author : Bilal Ahmed (231980028) — GIFT University
"""

import argparse
import os
import sys
import numpy as np

# ── Suppress TF info logs ─────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from utils.data_loader   import (get_train_generator, get_val_generator,
                                  print_dataset_info, TASK_CONFIG)
from utils.model_builder import (build_model, fine_tune_model,
                                  get_callbacks, print_model_summary)
from utils.visualizer    import (plot_training_history, plot_class_distribution)


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask / Helmet Detector")
    parser.add_argument("--task",        type=str,   default="mask",
                        choices=["mask", "helmet"],
                        help="Which task to train (mask or helmet)")
    parser.add_argument("--epochs",      type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--fine_tune",   action="store_true",
                        help="Also run fine-tuning phase after initial training")
    parser.add_argument("--fine_epochs", type=int,   default=10,
                        help="Epochs for fine-tuning phase")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    config = TASK_CONFIG[args.task]

    print("\n" + "=" * 60)
    print(f"  TRAINING  →  Task: {args.task.upper()}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print("=" * 60)

    # ── 0. Check dataset exists ──────────────
    if not os.path.exists(config["train_dir"]):
        print(f"\n❌  Dataset not found at '{config['train_dir']}'")
        print("   Please download and place images as shown in README.md")
        sys.exit(1)

    print_dataset_info(args.task)

    # ── 1. Data generators ───────────────────
    print("Loading data generators...")
    train_gen = get_train_generator(config["train_dir"], args.batch_size)
    val_gen   = get_val_generator(config["val_dir"],   args.batch_size)

    class_names = list(train_gen.class_indices.keys())
    print(f"Classes detected: {class_names}")
    print(f"Train samples: {train_gen.samples}")
    print(f"Val   samples: {val_gen.samples}")

    # ── 2. Plot class distribution ───────────
    plot_class_distribution(
        train_gen, class_names,
        save_path=f"models/{args.task}_class_distribution.png"
    )

    # ── 3. Build model ───────────────────────
    print("\nBuilding MobileNetV2 model...")
    model = build_model(learning_rate=args.lr)
    print_model_summary(model)

    # ── 4. Phase 1 — Train head only ─────────
    print(f"\n{'─'*50}")
    print("PHASE 1 — Training classification head (base model frozen)")
    print(f"{'─'*50}\n")

    os.makedirs("models", exist_ok=True)
    callbacks = get_callbacks(config["model_path"])

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 5. Phase 2 — Fine-tune (optional) ────
    if args.fine_tune:
        print(f"\n{'─'*50}")
        print("PHASE 2 — Fine-tuning top layers of MobileNetV2")
        print(f"{'─'*50}\n")

        model = fine_tune_model(model, unfreeze_from_layer=100, learning_rate=1e-5)

        history_ft = model.fit(
            train_gen,
            epochs=args.fine_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )

        # Merge histories for plotting
        for key in history.history:
            history.history[key].extend(history_ft.history[key])

    # ── 6. Save model ────────────────────────
    model.save(config["model_path"])
    print(f"\n✅  Model saved → {config['model_path']}")

    # ── 7. Plot training curves ───────────────
    plot_training_history(
        history,
        save_path=f"models/{args.task}_training_curves.png"
    )

    # ── 8. Final metrics ─────────────────────
    print("\nEvaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Final Val Loss     : {val_loss:.4f}")
    print(f"  Final Val Accuracy : {val_acc*100:.2f}%")
    print(f"{'='*40}")
    print("\nTraining complete! Run evaluate.py for full metrics.\n")


if __name__ == "__main__":
    main()
