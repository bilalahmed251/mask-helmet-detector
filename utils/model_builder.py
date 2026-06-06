"""
utils/model_builder.py
======================
MobileNetV2 Transfer Learning model for binary classification.

Architecture:
    MobileNetV2 (pretrained on ImageNet, frozen)
        ↓
    GlobalAveragePooling2D
        ↓
    Dense(128, relu) + Dropout(0.3)
        ↓
    Dense(1, sigmoid)   ←── Binary output

Author : Bilal Ahmed (231980028) — GIFT University
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)


# ─────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────
def build_model(input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Builds a transfer-learning model using MobileNetV2.

    Steps:
        1. Load MobileNetV2 pretrained on ImageNet (include_top=False)
        2. Freeze all base-model layers
        3. Add custom classification head
        4. Compile with Adam + binary_crossentropy

    Args:
        input_shape   : (H, W, C) — default (224, 224, 3)
        learning_rate : Adam optimizer LR — default 1e-4

    Returns:
        Compiled Keras model
    """

    # ── 1. Base model (frozen) ───────────────
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,          # remove ImageNet head
        weights="imagenet",         # pretrained weights
    )
    base_model.trainable = False    # freeze all layers

    # ── 2. Custom head ───────────────────────
    inputs = tf.keras.Input(shape=input_shape)

    x = base_model(inputs, training=False)   # run base in inference mode
    x = layers.GlobalAveragePooling2D()(x)   # flatten spatial dims
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)               # regularization
    outputs = layers.Dense(1, activation="sigmoid")(x)   # binary output

    model = models.Model(inputs, outputs)

    # ── 3. Compile ───────────────────────────
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────
# Fine-tune: unfreeze top N layers of base model
# ─────────────────────────────────────────────
def fine_tune_model(model, unfreeze_from_layer=100, learning_rate=1e-5):
    """
    Phase 2 — Fine-tuning.
    Unfreeze the top layers of MobileNetV2 and retrain
    with a much lower learning rate.

    Args:
        model             : Already-trained Keras model
        unfreeze_from_layer: Layer index from which to unfreeze
        learning_rate     : Lower LR for fine-tuning

    Returns:
        Model ready for fine-tune training
    """
    # The base model is the 2nd layer (index 1) in our model
    base_model = model.layers[1]
    base_model.trainable = True

    # Freeze everything before `unfreeze_from_layer`
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nFine-tune: Unfreezing layers from index {unfreeze_from_layer}")
    print(f"Trainable layers: {sum(1 for l in model.layers if l.trainable)}")
    return model


# ─────────────────────────────────────────────
# Training callbacks
# ─────────────────────────────────────────────
def get_callbacks(model_save_path: str):
    """
    Returns a list of Keras callbacks:
        - EarlyStopping (patience=5, restore best weights)
        - ModelCheckpoint (save best val_accuracy)
        - ReduceLROnPlateau (halve LR if val_loss stalls)
    """
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────
def print_model_summary(model):
    model.summary()
    total     = sum(tf.size(w).numpy() for w in model.weights)
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {total - trainable:,}")
