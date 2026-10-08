
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Vehicle Classification App", page_icon="🚘", layout="centered")

CLASSES = ["Car", "Motorcycle", "Bus", "Truck", "Van"]
MODEL_PATH = Path("model/vehicle_classifier.pkl")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.16),transparent 30%),#07111f;color:#fff}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:960px;padding-top:2rem}
.hero{border:1px solid #203653;background:#0b1424;border-radius:24px;padding:28px;margin-bottom:20px}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.badge{background:#10243a;border:1px solid #24496f;color:#22d3ee;padding:7px 12px;border-radius:9px;font-size:.76rem;font-weight:800}
.hero h1{color:#fff;margin:0 0 8px;font-size:2.2rem}
.hero p{color:#cbd5e1;line-height:1.65;margin:0}
.info-card{background:#0d1728;border:1px solid #223652;border-radius:18px;padding:18px;margin:10px 0}
.metric-label{color:#9fb4cf;font-size:.82rem}
.metric-value{color:#fff;font-size:1.55rem;font-weight:800}
[data-testid="stMarkdownContainer"] *{color:#fff!important}
[data-testid="stFileUploader"]{background:#0d1728;border:1px dashed #35527d;border-radius:16px;padding:10px}
.stButton>button{width:100%;border:0;border-radius:12px;font-weight:800;background:linear-gradient(135deg,#2563eb,#06b6d4);color:#fff!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badges">
    <span class="badge">MACHINE LEARNING</span>
    <span class="badge">STREAMLIT</span>
    <span class="badge">PYTHON</span>
  </div>
  <h1>Vehicle Classification App</h1>
  <p>Upload a vehicle image and classify it as Car, Motorcycle, Bus, Truck, or Van.</p>
</div>
""", unsafe_allow_html=True)

def extract_features(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2)-arr.min(axis=2)).mean())
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_strength = float(gx + gy)
    top = gray[:112].mean()
    bottom = gray[112:].mean()
    left = gray[:, :112].mean()
    right = gray[:, 112:].mean()
    return np.array([
        *means.tolist(), *stds.tolist(),
        brightness, contrast, saturation, edge_strength,
        float(top-bottom), float(left-right)
    ], dtype=np.float32).reshape(1, -1)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def demo_predict(features):
    x = features.flatten()
    seed = int(abs(float(np.dot(x, np.arange(1, len(x)+1)))) * 100000) % 100000
    rng = np.random.default_rng(seed)
    raw = rng.uniform(.15, 1.0, len(CLASSES))
    return raw / raw.sum()

model = load_model()

if model is None:
    st.info("Demo mode is active. Add `model/vehicle_classifier.pkl` to use a trained ML model.")
else:
    st.success("Trained model loaded successfully.")

uploaded = st.file_uploader("Upload vehicle image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded vehicle image", use_container_width=True)

    if st.button("Classify Vehicle"):
        features = extract_features(image)

        if model is not None and hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            names = list(getattr(model, "classes_", CLASSES))
            pairs = sorted(zip(names, probs), key=lambda x:x[1], reverse=True)
        else:
            probs = demo_predict(features)
            pairs = sorted(zip(CLASSES, probs), key=lambda x:x[1], reverse=True)

        best, conf = pairs[0]
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f'<div class="info-card"><div class="metric-label">Predicted Vehicle</div><div class="metric-value">{best}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="info-card"><div class="metric-label">Confidence</div><div class="metric-value">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)

        for name, prob in pairs:
            st.write(f"**{name}** — {prob*100:.1f}%")
            st.progress(float(prob))

        if model is None:
            st.warning("Demo prediction only. Train the included model for meaningful results.")

st.caption("Vehicle Classification App • Python • Streamlit • Machine Learning")
