"""
CIFAR-10 Image Classification - Model Training Pipeline
Easy-to-explain ML models: Logistic Regression, Random Forest, SVM
Libraries: numpy, matplotlib, pandas, scikit-learn
"""

import os
import pickle
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

DATASET_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_FILE = "static/data/cifar-10-python.tar.gz"
DATASET_DIR = "static/data/cifar-10-batches-py"


def download_cifar10():
    """Download and extract CIFAR-10 dataset."""
    if not os.path.exists(DATASET_DIR):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        print("Extracting dataset...")
        with tarfile.open(DATASET_FILE, 'r:gz') as tar:
            tar.extractall("static/data")
        print("Dataset ready.")
    else:
        print("Dataset already downloaded.")


def unpickle(file):
    """Load a CIFAR-10 batch file."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar10_data():
    """Load all CIFAR-10 training and test batches."""
    # Training data: 5 batches
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(DATASET_DIR, f'data_batch_{i}'))
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)

    # Test data
    test_batch = unpickle(os.path.join(DATASET_DIR, 'test_batch'))
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    return train_images, train_labels, test_images, test_labels


def preprocess_data(train_images, test_images, n_components=100):
    """
    Preprocessing Pipeline:
    1. Normalize pixel values to 0-1
    2. Flatten images (already flat in CIFAR-10)
    3. Standardize features
    4. Reduce dimensions using PCA for faster training
    """
    # Normalize pixel values
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Standardize
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_images)
    test_scaled = scaler.transform(test_images)

    # PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    return train_pca, test_pca, scaler, pca


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train three easy-to-explain models and evaluate them."""

    # Use smaller hyperparameters so training is fast and memory-safe on free Colab/Streamlit Cloud
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=-1, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=-1, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": y_pred
        }

        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

        # Save model
        joblib.dump(model, f'models/{name.replace(" ", "_").lower()}.joblib')

    return results


def plot_accuracy_comparison(results):
    """Bar chart comparing model accuracies."""
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison on CIFAR-10')
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/charts/accuracy_comparison.png')
    plt.close()


def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["predictions"])
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.2f}%')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/charts/confusion_matrices.png')
    plt.close()


def save_results_csv(results):
    """Save comparison results to CSV."""
    df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [f"{results[m]['accuracy']*100:.2f}%" for m in results],
        "Precision": [f"{results[m]['precision']*100:.2f}%" for m in results],
        "Recall": [f"{results[m]['recall']*100:.2f}%" for m in results],
        "F1-Score": [f"{results[m]['f1']*100:.2f}%" for m in results],
    })
    df.to_csv('static/data/model_comparison.csv', index=False)
    return df


def main():
    print("=" * 60)
    print("CIFAR-10 IMAGE CLASSIFICATION - ML PIPELINE")
    print("Models: Logistic Regression, Random Forest, SVM")
    print("=" * 60)

    # 1. Download dataset
    download_cifar10()

    # 2. Load dataset
    train_images, train_labels, test_images, test_labels = load_cifar10_data()

    # Use a subset for fast training everywhere (15,000 train / 3,000 test)
    # Full CIFAR-10 is 50,000 train / 10,000 test and needs more RAM/time.
    np.random.seed(42)
    train_idx = np.random.choice(len(train_images), size=15000, replace=False)
    test_idx = np.random.choice(len(test_images), size=3000, replace=False)
    train_images = train_images[train_idx]
    train_labels = train_labels[train_idx]
    test_images = test_images[test_idx]
    test_labels = test_labels[test_idx]

    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Image shape: 32x32x3 (flattened to {train_images.shape[1]} features)")

    # 3. Preprocess data
    print("\nPreprocessing data (normalization, standardization, PCA)...")
    X_train, X_test, scaler, pca = preprocess_data(train_images, test_images, n_components=100)
    print(f"Reduced feature dimensions: {X_train.shape[1]}")

    # Save preprocessing objects
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(pca, 'models/pca.joblib')

    # 4. Train and evaluate models
    results = train_and_evaluate_models(X_train, train_labels, X_test, test_labels)

    # 5. Generate visualizations
    print("\nGenerating comparison charts...")
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results, test_labels)

    # 6. Save results table
    df = save_results_csv(results)
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Training complete! Models saved in /models folder.")
    print("Charts saved in /static/charts folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
