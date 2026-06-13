"""
Streamlit deployment file for CIFAR-10 Image Classification.
Deploy this to Streamlit Cloud to get a URL like:
https://yourname-cifar10-classifier.streamlit.app/
"""

import os
import urllib.request
import tarfile
import numpy as np
from PIL import Image
import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="CIFAR-10 Image Classification",
    page_icon="ML",
    layout="wide"
)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_FILE = "static/data/cifar-10-python.tar.gz"
DATASET_DIR = "static/data/cifar-10-batches-py"

# Custom CSS for better appearance
custom_css = """
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    h1, h2, h3 {
        color: #f59e0b !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #1e3a8a);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(59,130,246,0.4);
    }
    .result-box {
        background: rgba(30, 58, 138, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .code-box {
        background: #0d1117;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


def ensure_dataset():
    """Download dataset if not present."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs('static/data', exist_ok=True)
        with st.spinner('Downloading CIFAR-10 dataset (one-time)...'):
            urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
            with tarfile.open(DATASET_FILE, 'r:gz') as tar:
                tar.extractall("static/data")


@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models and preprocessing objects."""
    ensure_dataset()
    model_files = {
        "Logistic Regression": "models/logistic_regression.joblib",
        "Random Forest": "models/random_forest.joblib",
        "Support Vector Machine": "models/support_vector_machine.joblib",
    }
    loaded_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
    
    scaler = joblib.load("models/scaler.joblib") if os.path.exists("models/scaler.joblib") else None
    pca = joblib.load("models/pca.joblib") if os.path.exists("models/pca.joblib") else None
    return loaded_models, scaler, pca


def preprocess_image(image, scaler, pca):
    """Preprocess uploaded image for prediction."""
    img = image.convert('RGB').resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    flat = img_array.reshape(1, -1)
    scaled = scaler.transform(flat)
    reduced = pca.transform(scaled)
    return reduced


# Load models
models, scaler, pca = load_models()

# Sidebar navigation
st.sidebar.title("ML Project")
page = st.sidebar.radio("Go to", [
    "Home",
    "Model Comparison",
    "Dataset",
    "ML Pipeline",
    "Source Code",
    "About"
])

# Check if models loaded correctly
models_loaded = len(models) == 3 and scaler is not None and pca is not None

if not models_loaded:
    st.error("""
    Models not found! Please upload the `models/` folder to GitHub/Streamlit Cloud.
    
    The folder should contain:
    - logistic_regression.joblib
    - random_forest.joblib
    - support_vector_machine.joblib
    - scaler.joblib
    - pca.joblib
    """)

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Image Classification System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>CIFAR-10 Dataset | Logistic Regression, Random Forest, SVM</p>", unsafe_allow_html=True)
    
    if models_loaded:
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'webp'])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file)

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                with st.spinner("Classifying image..."):
                    try:
                        features = preprocess_image(image, scaler, pca)
                        predictions = {}
                        for name, model in models.items():
                            pred_class = model.predict(features)[0]
                            proba = model.predict_proba(features)[0]
                            predictions[name] = {
                                "class": CLASS_NAMES[pred_class],
                                "confidence": float(np.max(proba) * 100)
                            }

                        final = predictions["Support Vector Machine"]
                        
                        st.markdown(f"""
                        <div class='result-box'>
                            <h2>Predicted Class: {final['class'].upper()}</h2>
                            <h3 style='color: #3b82f6;'>Confidence: {final['confidence']:.2f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        st.subheader("All Models Predictions")
                        for name, result in predictions.items():
                            st.write(f"**{name}**: {result['class']} ({result['confidence']:.2f}%)")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

# ---------------- COMPARISON ----------------
elif page == "Model Comparison":
    st.markdown("<h1 style='text-align: center;'>Model Comparison & Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Three easy-to-explain ML models compared</p>", unsafe_allow_html=True)

    results = {
        "Logistic Regression": {"Accuracy": "39.33%", "Precision": "38.86%", "Recall": "39.33%", "F1-Score": "38.95%"},
        "Random Forest": {"Accuracy": "38.63%", "Precision": "38.03%", "Recall": "38.63%", "F1-Score": "37.88%"},
        "Support Vector Machine": {"Accuracy": "48.90%", "Precision": "48.72%", "Recall": "48.90%", "F1-Score": "48.62%"}
    }

    st.markdown("""
    <div class='result-box'>
        <h3>Best Model: Support Vector Machine (SVM)</h3>
        <p>SVM performed best because it finds the optimal decision boundary between classes in the reduced PCA feature space.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Results Table")
    st.table(results)

    if os.path.exists("static/charts/accuracy_comparison.png"):
        st.subheader("Accuracy Comparison Chart")
        st.image("static/charts/accuracy_comparison.png", use_container_width=True)

    if os.path.exists("static/charts/confusion_matrices.png"):
        st.subheader("Confusion Matrices")
        st.image("static/charts/confusion_matrices.png", use_container_width=True)

# ---------------- DATASET ----------------
elif page == "Dataset":
    st.markdown("<h1 style='text-align: center;'>Dataset Used: CIFAR-10</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Source**: [CIFAR-10 Official](https://www.cs.toronto.edu/~kriz/cifar.html)
        
        **Kaggle**: [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python)
        
        **Total Images**: 60,000
        
        **Training Images**: 50,000
        
        **Test Images**: 10,000
        
        **Image Size**: 32 x 32 pixels, RGB
        
        **Classes**: 10
        """)
    
    with col2:
        st.subheader("Class Labels")
        st.write(", ".join([name.title() for name in CLASS_NAMES]))
    
    st.subheader("Loading Code")
    code = '''
import urllib.request, tarfile, pickle, numpy as np

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
urllib.request.urlretrieve(url, "cifar-10-python.tar.gz")

with tarfile.open("cifar-10-python.tar.gz", 'r:gz') as tar:
    tar.extractall(".")

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

train_images, train_labels = [], []
for i in range(1, 6):
    batch = unpickle(f'cifar-10-batches-py/data_batch_{i}')
    train_images.append(batch[b'data'])
    train_labels.extend(batch[b'labels'])

train_images = np.vstack(train_images)
train_labels = np.array(train_labels)
'''
    st.code(code, language='python')

# ---------------- PIPELINE ----------------
elif page == "ML Pipeline":
    st.markdown("<h1 style='text-align: center;'>ML Pipeline Implementation</h1>", unsafe_allow_html=True)

    steps = [
        ("1. Data Collection", "Download CIFAR-10 from official source using urllib."),
        ("2. Data Preprocessing", "Normalize pixel values to 0-1, flatten images, standardize features."),
        ("3. Feature Engineering", "Apply PCA to reduce 3072 pixel features to 100 components."),
        ("4. Model Training", "Train Logistic Regression, Random Forest, and SVM."),
        ("5. Model Evaluation", "Use Accuracy, Precision, Recall, F1-Score, Confusion Matrix."),
        ("6. Prediction", "Load saved model and predict class of new uploaded image.")
    ]

    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

# ---------------- SOURCE CODE ----------------
elif page == "Source Code":
    st.markdown("<h1 style='text-align: center;'>Complete Source Code</h1>", unsafe_allow_html=True)

    try:
        with open('train_models.py', 'r') as f:
            train_code = f.read()
        st.subheader("train_models.py")
        st.code(train_code, language='python')
    except Exception:
        st.error("train_models.py not found.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown("<h1 style='text-align: center;'>About This Project</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Project Title**: Image Classification System using CIFAR-10 Dataset
    
    **Problem**: Classify 32x32 color images into 10 categories.
    
    **Language**: Python
    
    **Libraries**: NumPy, Matplotlib, Pandas, scikit-learn, Flask, Streamlit
    
    **Models**: Logistic Regression, Random Forest, Support Vector Machine
    
    **Developed for**: Machine Learning Course Project
    """)

st.markdown("<div class='footer'> 2025 CIFAR-10 Image Classification Project | Built for Machine Learning Course</div>", unsafe_allow_html=True)
