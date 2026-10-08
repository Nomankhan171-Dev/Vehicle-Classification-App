
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "vehicle_classifier.pkl"

def extract_features(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224,224))
    arr = np.asarray(img).astype(np.float32)/255.0
    gray = arr.mean(axis=2)
    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2)-arr.min(axis=2)).mean())
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    top = gray[:112].mean()
    bottom = gray[112:].mean()
    left = gray[:, :112].mean()
    right = gray[:, 112:].mean()
    return np.array([
        *means.tolist(), *stds.tolist(),
        brightness, contrast, saturation, float(gx+gy),
        float(top-bottom), float(left-right)
    ], dtype=np.float32)

X, y = [], []

for class_dir in sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()]):
    for file in class_dir.iterdir():
        if file.suffix.lower() not in {".jpg",".jpeg",".png"}:
            continue
        X.append(extract_features(Image.open(file)))
        y.append(class_dir.name)

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Saved:", MODEL_PATH)
