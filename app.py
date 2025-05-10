import streamlit as st
import numpy as np
import cv2
from PIL import Image   
import pickle
from typing import Tuple, Dict, List, Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable
from tensorflow.keras.layers import Layer, Dense, Conv2D, Multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
import tensorflow as tf
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import os
import tempfile
import pandas as pd 


# -----------------------------
# Step 1: Define Custom Objects
# -----------------------------

class CBAMLayer(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.shared_dense_one = Dense(self.channels // self.ratio, activation='relu', use_bias=True)
        self.shared_dense_two_avg = Dense(self.channels, activation='sigmoid', use_bias=True)
        self.shared_dense_two_max = Dense(self.channels, activation='sigmoid', use_bias=True)
        self.conv_spatial = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        super(CBAMLayer, self).build(input_shape)

    def call(self, inputs):
        channel_avg = GlobalAveragePooling2D()(inputs)
        channel_max = GlobalMaxPooling2D()(inputs)
        channel_avg = self.shared_dense_one(channel_avg)
        channel_max = self.shared_dense_one(channel_max)
        channel_avg = self.shared_dense_two_avg(channel_avg)
        channel_max = self.shared_dense_two_max(channel_max)
        channel_attention = Add()([channel_avg, channel_max])
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1)
        channel_refined = Multiply()([inputs, channel_attention])
        spatial_avg = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_concat = tf.concat([spatial_avg, spatial_max], axis=-1)
        spatial_attention = self.conv_spatial(spatial_concat)
        spatial_refined = Multiply()([channel_refined, spatial_attention])
        return spatial_refined

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


@register_keras_serializable()
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_sum(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt), axis=-1)
    return focal_loss_fixed

# Register custom objects
get_custom_objects().update({
    'CBAMLayer': CBAMLayer,
    'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)
})


# Load model
MODEL_PATH = r'liver_tumor_classifier.keras'  # Update path if needed
METADATA_PATH = r'D:\liver-tumor-app\model_metadata (2).pkl'  # Update path if needed

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam',
              loss=focal_loss(gamma=2., alpha=0.25),
              metrics=['accuracy'])

with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
thresholds = metadata.get('thresholds', {0: 0.36178, 1: 0.48146927, 2: 0.5123631})
class_names = ['Normal Liver', 'Hemangioma Liver Tumor', 'Hepatocellular Carcinoma']


# -----------------------------
# Step 2: Image Preprocessing
# -----------------------------

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    if img is None:
        raise ValueError("Image could not be decoded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension


# -----------------------------
# Step 3: Prediction Function
# -----------------------------

def predict_image(model, image_bytes: bytes, thresholds: dict, class_names: list) -> Tuple[str, dict]:
    try:
        processed_img = preprocess_image(image_bytes)
        probabilities = model.predict(processed_img)[0]
        results = []
        for i, prob in enumerate(probabilities):
            if prob >= thresholds[i]:
                results.append((class_names[i], float(prob)))
        if not results:
            return "Uncertain", dict(zip(class_names, probabilities))
        elif len(results) == 1:
            return results[0][0], dict(zip(class_names, probabilities))
        else:
            results.sort(key=lambda x: x[1], reverse=True)
            return results[0][0], dict(zip(class_names, probabilities))
    except Exception as e:
        return f"Error: {str(e)}", {}


# -----------------------------
# Step 4: PDF Export Function with Image Preview
# -----------------------------

def download_pdf(data: dict, image_bytes: bytes, filename: str = "liver_prediction_report.pdf", key: str = "default_key") -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(0, 10, txt="Liver Tumor Classification Report", ln=True, align='C')
    pdf.ln(10)

    # Timestamp
    pdf.cell(0, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(10)

    # Add image
    if image_bytes:
        try:
            temp_dir = tempfile.mkdtemp()
            img_path = os.path.join(temp_dir, "uploaded_image.jpg")
            img = Image.open(BytesIO(image_bytes))
            img.save(img_path)
            pdf.image(img_path, x=50, w=110)
            pdf.ln(10)
        except Exception as e:
            st.warning("⚠️ Could not embed image in PDF")

    # Prediction result
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Prediction:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"- {data['prediction']}", ln=True)
    pdf.ln(5)

    # Confidence scores
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Confidence Scores:", ln=True)
    pdf.set_font("Arial", '', 12)
    for cls, score in data.get("scores", {}).items():
        pdf.cell(0, 10, txt=f"- {cls}: {score:.4f}", ln=True)

    # Output to bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output = BytesIO(pdf_bytes)

    # Provide download button
    st.download_button(
        label="📄 Download Result (PDF)",
        data=pdf_output,
        file_name=filename,
        mime="application/pdf",
        key=key
    )


# -----------------------------
# Step 5: Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="🧬 Liver Tumor Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for modern UI
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f8f9fa;
        color: #2c3e50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 15px;
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align:center;'>🧬 Liver Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:16px;'>Upload an ultrasound scan of the liver for classification.</p>", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("""
    <div style="background:#007bff; color:white; padding:20px; border-radius:10px;">
    <h3>🧠 About This Model</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Goal
    This AI model assists in identifying and classifying common types of liver tumors from ultrasound images to support early diagnosis.
    ### 🔍 Classes Detected
    - ✅ **Normal Liver**
    - ⚠️ **Hemangioma (Benign)**
    - ❗ **Hepatocellular Carcinoma (Malignant)**
    ### 📈 Why Use This?
    - Fast and consistent predictions
    - Optimized for clinical use
    - Interpretable output
    """)

# Tabs
tab1, tab2 = st.tabs(["🖼️ Single Image Upload", "📂 Batch Upload"])

# ---- Tab 1: Single Upload ----
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img_rgb, caption='Uploaded Image', use_container_width=True)
        with col2:
            predicted_class, confidence_dict = predict_image(model, file_bytes, thresholds, class_names)
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            if predicted_class.startswith("Error"):
                st.error(predicted_class)
            elif predicted_class == "Uncertain":
                st.warning("⚠️ Prediction Uncertain")
            else:
                st.success(f"🎯 **Prediction:** {predicted_class}")
            st.subheader("📊 Confidence Scores:")
            probabilities = [confidence_dict.get(cls, 0.0) for cls in class_names]
            for cls, score in zip(class_names, probabilities):
                st.progress(float(score))
                st.text(f"{cls}: {score:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            result_data = {
                "prediction": predicted_class,
                "scores": {c: float(s) for c, s in zip(class_names, probabilities)}
            }
            # Generate PDF with image preview
            download_pdf(result_data, image_bytes=file_bytes, filename="liver_prediction_report.pdf", key="single_report")

# ---- Tab 2: Batch Upload ----
with tab2:
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} images...")
        results_list = []
        cols = st.columns(3)
        idx = 0
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            try:
                img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (128, 128)) / 255.0
                input_tensor = np.expand_dims(img_resized, axis=0)
                probabilities = model.predict(input_tensor)[0]
                predicted_class = max(zip(class_names, probabilities), key=lambda x: x[1])[0]
                results_list.append({
                    "filename": uploaded_file.name,
                    "prediction": predicted_class,
                    "scores": {c: float(s) for c, s in zip(class_names, probabilities)},
                    "image_bytes": file_bytes
                })
                with cols[idx % 3]:
                    st.image(img_rgb, caption=uploaded_file.name, use_container_width=True)
                    # Display prediction badge
                    if predicted_class == "Normal Liver":
                        st.markdown(f"<span style='background-color:#d4edda; color:#155724; padding:5px; border-radius:5px;'>✅ Normal Liver</span>", unsafe_allow_html=True)
                    elif predicted_class == "Hemangioma Liver Tumor":
                        st.markdown(f"<span style='background-color:#fff3cd; color:#856404; padding:5px; border-radius:5px;'>⚠️ Hemangioma</span>", unsafe_allow_html=True)
                    elif predicted_class == "Hepatocellular Carcinoma":
                        st.markdown(f"<span style='background-color:#f8d7da; color:#721c24; padding:5px; border-radius:5px;'>❗ Hepatocellular Carcinoma</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='background-color:#d1d3d4; color:#3b3d3f; padding:5px; border-radius:5px;'>❓ Uncertain</span>", unsafe_allow_html=True)
                    # Confidence scores
                    for cls, score in zip(class_names, probabilities):
                        st.markdown(f"**{cls}**: <span style='color:green;'>{score:.2f}</span>", unsafe_allow_html=True)
                        st.progress(float(score))
                    # Download button per image
                    download_pdf({
                        "prediction": predicted_class,
                        "scores": {c: float(s) for c, s in zip(class_names, probabilities)}
                    }, image_bytes=file_bytes, filename=f"report_{uploaded_file.name}.pdf", key=f"batch_report_{uploaded_file.name}")
                idx += 1
            except Exception as e:
                st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        # Show summary table (collapsible)
        with st.expander("📋 View All Predictions"):
            df = pd.DataFrame([
                {
                    "Filename": item["filename"],
                    "Prediction": item["prediction"],
                    **item["scores"]
                } for item in results_list
            ])
            df_styled = df.style.applymap(lambda x: 'background-color: #d4edda;color:#155724;' if x == "Normal Liver" else '') \
                             .applymap(lambda x: 'background-color: #fff3cd;color:#856404;' if x == "Hemangioma Liver Tumor" else '') \
                             .applymap(lambda x: 'background-color: #f8d7da;color:#721c24;' if x == "Hepatocellular Carcinoma" else '')
            st.dataframe(df_styled, use_container_width=True)
        # Option to download all at once
        if st.button("📄 Download All Reports as PDF"):
            for item in results_list:
                download_pdf({
                    "prediction": item["prediction"],
                    "scores": item["scores"]
                }, image_bytes=item["image_bytes"], filename=f"report_{item['filename']}.pdf", key=f"all_reports_{item['filename']}")
            st.success("✅ All PDF reports generated!")

# Footer
st.markdown("""
<footer style="text-align:center; margin-top:30px; font-size:14px; color:#888;">
    © 2025 Liver Tumor Classifier | Powered by Deep Learning & Streamlit
</footer>
""", unsafe_allow_html=True)