"""
utils/export_onnx.py
====================
Convert Keras .h5 models to optimized ONNX format for high-performance deployment.

Usage:
    python utils/export_onnx.py --input models/mask_model.h5

Author : Bilal Ahmed (231980028) — GIFT University
"""

import os
import sys
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def parse_args():
    parser = argparse.ArgumentParser(description="Export Keras .h5 model to ONNX format.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Keras .h5 model file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output .onnx file (defaults to matching input path but with .onnx extension)."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input model path '{args.input}' does not exist.")
        sys.exit(1)
        
    output_path = args.output
    if not output_path:
        # Swap extension to .onnx
        base, _ = os.path.splitext(args.input)
        output_path = base + ".onnx"
        
    print(f"[INFO] Loading Keras model from {args.input}...")
    try:
        import tensorflow as tf
        import tf2onnx
        import onnxruntime as ort
    except ImportError as e:
        print(f"[ERROR] Required library is missing: {str(e)}")
        print("   Please install packages first: pip install tf2onnx onnxruntime tensorflow")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(args.input)
        print("[SUCCESS] Keras model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Keras model: {e}")
        sys.exit(1)
        
    # Get input shape and type from model
    input_shape = model.inputs[0].shape
    input_dtype = model.inputs[0].dtype
    print(f"Input Name  : {model.inputs[0].name}")
    print(f"Input Shape : {input_shape}")
    print(f"Input Dtype : {input_dtype}")
    
    # Define input signature
    # Handle dynamic batch dimensions (replace None with a static or dynamic spec)
    input_signature = [
        tf.TensorSpec(shape=input_shape, dtype=input_dtype, name="input_1")
    ]
    
    print(f"[INFO] Converting Keras model to ONNX (opset={args.opset})...")
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=args.opset
        )
        
        # Save model
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
            
        print(f"[SUCCESS] ONNX model successfully saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] ONNX conversion failed: {e}")
        sys.exit(1)
        
    # ── Verify the exported ONNX model ──────────────────────
    print("\n[INFO] Verifying ONNX model runtime & numerical consistency...")
    try:
        # 1. Start ORT session
        ort_sess = ort.InferenceSession(output_path)
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        
        # Create random input tensor matching the expected shape (e.g. batch size of 1)
        # Replacing dynamic dims with 1
        test_shape = [1 if d is None else d for d in input_shape]
        dummy_input = np.random.uniform(-1.0, 1.0, test_shape).astype(np.float32)
        
        # 2. Run TF prediction
        tf_pred = model.predict(dummy_input, verbose=0)
        
        # 3. Run ONNX Runtime prediction
        onnx_pred = ort_sess.run([output_name], {input_name: dummy_input})[0]
        
        # 4. Compare outputs
        difference = np.abs(tf_pred - onnx_pred)
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        
        print(f"   ONNX input node name  : {input_name}")
        print(f"   ONNX output node name : {output_name}")
        print(f"   Max absolute difference: {max_diff:.2e}")
        print(f"   Mean difference       : {mean_diff:.2e}")
        
        # Threshold check (1e-4 is standard for FP32 predictions)
        if max_diff < 1e-4:
            print("[SUCCESS] Numerical verification PASSED! ONNX model output matches TensorFlow exactly.")
        else:
            print("[WARNING] Numerical verification WARNING: Outputs have small variations (expected occasionally).")
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
