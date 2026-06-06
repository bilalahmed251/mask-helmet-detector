"""
realtime_detect.py
==================
Live webcam / video file detection using OpenCV.

How it works:
    1. Capture frame from webcam
    2. Detect faces using OpenCV Haar Cascade
    3. Crop each face → resize → preprocess
    4. Run model.predict()
    5. Draw bounding box + label on frame
    6. Show live window

Usage:
    # Webcam (default camera)
    python realtime_detect.py --task mask

    # Video file
    python realtime_detect.py --task mask --source path/to/video.mp4

    # Helmet detection (full frame — no face crop needed)
    python realtime_detect.py --task helmet

Controls:
    Q  →  Quit
    S  →  Save screenshot

Author : Bilal Ahmed (231980028) — GIFT University
"""

import argparse
import os
import time
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import tensorflow as tf

from utils.data_loader import TASK_CONFIG


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE       = (224, 224)
CONF_THRESHOLD = 0.6     # only show prediction if confidence > 60 %

# Color palette (BGR for OpenCV)
COLOR_GREEN = (50,  200,  50)
COLOR_RED   = (50,   50, 220)
COLOR_GRAY  = (180, 180, 180)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0,     0,   0)


# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time detection")
    parser.add_argument("--task",   type=str, default="mask",
                        choices=["mask", "helmet"])
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (0) or video file path")
    return parser.parse_args()


# ─────────────────────────────────────────────
def preprocess_face(face_img):
    """
    Preprocess a face crop for MobileNetV2 inference.
    Returns numpy array of shape (1, 224, 224, 3).
    """
    face_resized = cv2.resize(face_img, IMG_SIZE)
    face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_array   = face_rgb.astype(np.float32)
    face_array   = tf.keras.applications.mobilenet_v2.preprocess_input(face_array)
    return np.expand_dims(face_array, axis=0)


# ─────────────────────────────────────────────
def draw_label_box(frame, x, y, w, h, label, confidence, color):
    """
    Draws a bounding box + filled label on the frame.
    """
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label background
    label_text  = f"{label}  {confidence*100:.0f}%"
    font_scale  = 0.65
    thickness   = 2
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)

    # Label text
    cv2.putText(frame, label_text,
                (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, COLOR_WHITE, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────
def get_label_color(task, label):
    """Returns green for compliant, red for violation."""
    if task == "mask"   and label == "with_mask":   return COLOR_GREEN
    if task == "helmet" and label == "helmet":       return COLOR_GREEN
    return COLOR_RED


# ─────────────────────────────────────────────
def main():
    args   = parse_args()
    config = TASK_CONFIG[args.task]

    # ── Load model ────────────────────────────
    if not os.path.exists(config["model_path"]):
        print(f"❌  Model not found: {config['model_path']}")
        print("   Run train.py first.")
        return

    print(f"Loading model: {config['model_path']}")
    model = tf.keras.models.load_model(config["model_path"])
    class_names = config["class_names"]

    # ── Load Haar Cascade for face detection ──
    # (only for mask task — helmet works on whole image region)
    face_cascade = None
    if args.task == "mask":
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        print("Face cascade loaded ✓")

    # ── Open camera / video ───────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌  Cannot open source: {args.source}")
        return

    print(f"\nReal-time detection started  (task={args.task.upper()})")
    print("Controls:  Q = Quit  |  S = Save screenshot\n")

    fps_display  = 0
    frame_count  = 0
    start_time   = time.time()
    screenshot_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        frame_count += 1
        display = frame.copy()

        # ── FPS calculation ───────────────────
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_display = frame_count / elapsed

        # ─────────────────────────────────────
        # MASK task → detect faces, classify each
        # ─────────────────────────────────────
        if args.task == "mask" and face_cascade is not None:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:
                # Add some padding around the face crop
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                inp  = preprocess_face(face_crop)
                prob = model.predict(inp, verbose=0)[0][0]

                pred_idx   = int(prob >= 0.5)
                label      = class_names[pred_idx]
                confidence = prob if pred_idx == 1 else 1.0 - prob
                color      = get_label_color(args.task, label)

                if confidence >= CONF_THRESHOLD:
                    draw_label_box(display, x1, y1,
                                   x2 - x1, y2 - y1,
                                   label, confidence, color)

        # ─────────────────────────────────────
        # HELMET task → classify whole frame
        # (use for top-half crop or full frame)
        # ─────────────────────────────────────
        elif args.task == "helmet":
            # Use top half of frame (head region)
            h_frame = frame.shape[0]
            top_half = frame[:h_frame // 2, :]

            inp  = preprocess_face(top_half)
            prob = model.predict(inp, verbose=0)[0][0]

            pred_idx   = int(prob >= 0.5)
            label      = class_names[pred_idx]
            confidence = prob if pred_idx == 1 else 1.0 - prob
            color      = get_label_color(args.task, label)

            if confidence >= CONF_THRESHOLD:
                # Draw on top portion
                draw_label_box(display, 10, 10,
                               frame.shape[1] - 20, h_frame // 2 - 10,
                               label, confidence, color)

        # ── Overlay: FPS + task label ─────────
        cv2.putText(display,
                    f"FPS: {fps_display:.1f}  |  Task: {args.task.upper()}",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_GRAY, 1, cv2.LINE_AA)

        # ── Show frame ────────────────────────
        cv2.imshow(f"Detection — {args.task.upper()}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_n += 1
            fname = f"screenshot_{args.task}_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
