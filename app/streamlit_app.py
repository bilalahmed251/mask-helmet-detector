"""
app/streamlit_app.py
====================
Beautiful Streamlit web application for
Face Mask & Helmet Detection.

Run:
    streamlit run app/streamlit_app.py

Author : Bilal Ahmed (231980028) — GIFT University
"""

import os
import sys
import numpy as np
from PIL import Image
import io

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf

from utils.data_loader import load_image_for_inference, TASK_CONFIG


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mask & Helmet Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .safe {
        background-color: #E8F5E9;
        color: #2E7D32;
        border: 2px solid #4CAF50;
    }
    .danger {
        background-color: #FFEBEE;
        color: #C62828;
        border: 2px solid #F44336;
    }
    .info-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loader (cached — loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(task: str):
    config     = TASK_CONFIG[task]
    model_path = config["model_path"]

    if not os.path.exists(model_path):
        return None

    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────
def predict(model, img: Image.Image, class_names: list):
    """Run inference on a PIL image."""
    # Save to buffer, load via our utility
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    img_array = np.array(img.resize((224, 224)), dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prob       = model.predict(img_array, verbose=0)[0][0]
    pred_idx   = int(prob >= 0.5)
    pred_label = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else 1.0 - prob

    return pred_label, float(confidence), float(prob)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/face-id.png", width=80)
    st.title("Settings")

    task = st.selectbox(
        "Detection Task",
        options=["mask", "helmet"],
        format_func=lambda x: "😷 Face Mask Detection" if x == "mask"
                               else "⛑️ Helmet Detection",
    )

    config      = TASK_CONFIG[task]
    class_names = config["class_names"]

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **Model:** MobileNetV2 (Transfer Learning)
    **Input:** 224 × 224 RGB
    **Output:** Binary classification
    **Expected Accuracy:** 95–98%
    """)

    st.divider()
    st.markdown("**Bilal Ahmed** | GIFT University | 231980028")


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ Safety Compliance Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image to detect Face Mask / Helmet compliance</div>',
    unsafe_allow_html=True
)

# ── Task info card ────────────────────────────
task_emoji = "😷" if task == "mask" else "⛑️"
st.markdown(f"""
<div class="info-card">
  <strong>{task_emoji} Active Task:</strong>
  {"Face Mask Detection — classifies whether a person is wearing a mask or not."
   if task == "mask"
   else "Helmet Detection — classifies whether a rider is wearing a helmet or not."}
  <br><br>
  <strong>Classes:</strong> {" / ".join(c.replace("_", " ").title() for c in class_names)}
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────
model = load_model(task)

if model is None:
    st.error(
        f"❌ Model not found at `{config['model_path']}`\n\n"
        "Please train the model first:\n"
        f"```bash\npython train.py --task {task}\n```"
    )
    st.stop()

st.success("✅ Model loaded successfully!")

# ── Upload section ────────────────────────────
st.divider()
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a clear image of a person's face / upper body",
)

# ── Demo mode (no upload) ─────────────────────
if uploaded_file is None:
    st.info("👆 Upload an image above to get started.")

    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Upload**\nChoose any JPG/PNG image")
    with col2:
        st.markdown("**2️⃣ Analyze**\nModel runs in < 1 second")
    with col3:
        st.markdown("**3️⃣ Result**\nSee prediction + confidence")

else:
    # ── Display uploaded image ─────────────────
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Uploaded Image")
        st.image(img, use_container_width=True)

    # ── Run prediction ─────────────────────────
    with st.spinner("Running detection..."):
        label, confidence, raw_prob = predict(model, img, class_names)

    # ── Determine if compliant ─────────────────
    is_safe = (task == "mask"   and label == "with_mask") or \
              (task == "helmet" and label == "helmet")

    with col2:
        st.markdown("#### Prediction Result")

        icon  = "✅" if is_safe else "❌"
        cls   = "safe" if is_safe else "danger"
        label_display = label.replace("_", " ").upper()

        st.markdown(
            f'<div class="result-box {cls}">'
            f'{icon} {label_display}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence metrics
        st.metric("Confidence",  f"{confidence*100:.1f}%")
        st.metric("Raw Sigmoid", f"{raw_prob:.4f}")
        st.metric("Status",
                  "✅ Compliant" if is_safe else "❌ Violation",
                  delta="Safe"   if is_safe else "Unsafe",
                  delta_color="normal" if is_safe else "inverse")

    # ── Confidence bar ─────────────────────────
    st.divider()
    st.markdown("#### Confidence Breakdown")

    prob_class0 = 1.0 - raw_prob
    prob_class1 = raw_prob

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{class_names[0].replace('_', ' ').title()}**")
        st.progress(float(prob_class0))
        st.caption(f"{prob_class0*100:.1f}%")
    with col_b:
        st.markdown(f"**{class_names[1].replace('_', ' ').title()}**")
        st.progress(float(prob_class1))
        st.caption(f"{prob_class1*100:.1f}%")

    # ── Recommendation ─────────────────────────
    st.divider()
    if is_safe:
        st.success(
            "✅ **Compliant!** The person is following safety guidelines."
        )
    else:
        st.error(
            "❌ **Violation Detected!** "
            + ("Please wear a face mask." if task == "mask"
               else "Please wear a helmet for safety.")
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built with ❤️ by Bilal Ahmed | "
    "GIFT University | Deep Learning Project | "
    "MobileNetV2 Transfer Learning</small></center>",
    unsafe_allow_html=True,
)
