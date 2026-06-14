"""
Flask Web Application for CIFAR-10 Image Classification
Pages: Home (Upload & Classify), Comparison, Dataset, Source Code, ML Pipeline
"""

import os
import pickle
import urllib.request
import tarfile
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import joblib


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_FILE = "static/data/cifar-10-python.tar.gz"
DATASET_DIR = "static/data/cifar-10-batches-py"

# Global model cache
models = {}
scaler = None
pca = None


def ensure_dataset():
    """Download CIFAR-10 if not present."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs('static/data', exist_ok=True)
        # urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        with tarfile.open(DATASET_FILE, 'r:gz') as tar:
            tar.extractall("static/data")


def load_models():
    """Load trained models and preprocessing objects."""
    global models, scaler, pca
    ensure_dataset()
    model_files = {
        "Logistic Regression": "models/logistic_regression.joblib",
        "Random Forest": "models/random_forest.joblib",
        "Support Vector Machine": "models/support_vector_machine.joblib",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if os.path.exists("models/scaler.joblib"):
        scaler = joblib.load("models/scaler.joblib")
    if os.path.exists("models/pca.joblib"):
        pca = joblib.load("models/pca.joblib")
    if len(models) == 0:
    raise Exception("No models loaded. Check models folder.")


def preprocess_uploaded_image(image_path):
    """Resize, flatten, normalize, standardize, and apply PCA."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    flat = img_array.reshape(1, -1)
    if scaler is None or pca is None:
        raise Exception("Scaler or PCA not loaded")
      
   scaled = scaler.transform(flat)
   reduced = pca.transform(scaled)
   return reduced


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    try:
        features = preprocess_uploaded_image(filename)
        predictions = {}
        for name, model in models.items():
            pred_class = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                  proba = model.predict_proba(features)[0]
            else:
                  proba = np.ones(len(CLASS_NAMES))
            predictions[name] = {
                "class": CLASS_NAMES[pred_class],
                "confidence": float(np.max(proba) * 100)
            }

        # Use best performing model (SVM) as final answer
        final_model = "Support Vector Machine"
        final_class = predictions[final_model]["class"]
        final_confidence = predictions[final_model]["confidence"]

        return jsonify({
            "success": True,
            "prediction": final_class,
            "confidence": final_confidence,
            "all_models": predictions,
            "image_url": "/" + filename.replace("\\", "/")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/comparison')
def comparison():
    return render_template('comparison.html')


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


@app.route('/sourcecode')
def sourcecode():
    try:
        with open('train_models.py', 'r') as f:
            source_code = f.read()
        with open('app.py', 'r') as f:
            flask_code = f.read()
    except Exception:
        source_code = "Source code not available."
        flask_code = "Flask code not available."
    return render_template('sourcecode.html', source_code=source_code, flask_code=flask_code)


@app.route('/pipeline')
def pipeline():
    return render_template('pipeline.html')


@app.route('/about')
def about():
    return render_template('about.html')


# Auto-load models when imported (for production/gunicorn)
load_models()
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
