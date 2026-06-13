# 🎓 CIFAR-10 Image Classification System

A complete Machine Learning course project that classifies images from the **CIFAR-10 dataset** into 10 categories using three easy-to-explain algorithms:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

The project includes:
- ✅ Full ML pipeline (preprocessing, PCA, training, evaluation)
- ✅ Model comparison with charts and analysis
- ✅ Flask web app with upload & classify functionality
- ✅ Streamlit deployment file
- ✅ Source code ready for VS Code / Google Colab

---

## 📂 Project Structure

```
image-classification-project/
├── app.py                  # Flask web application
├── train_models.py         # ML training pipeline
├── streamlit_app.py        # Streamlit deployment file
├── requirements.txt        # Python dependencies
├── models/                 # Saved trained models
├── static/
│   ├── charts/             # Accuracy & confusion matrix charts
│   ├── css/style.css       # Website styling
│   ├── js/main.js          # Frontend JavaScript
│   └── uploads/            # Uploaded images
├── templates/              # HTML pages
└── README.md               # This file
```

---

## 🛠️ Step-by-Step Setup (VS Code)

### Step 1: Install Python
Download and install Python 3.10 or higher from [python.org](https://python.org).

### Step 2: Open Project in VS Code
1. Open VS Code.
2. Go to `File → Open Folder` and select `image-classification-project`.

### Step 3: Install Dependencies
Open terminal in VS Code (`Ctrl + ` `) and run:

```bash
pip install -r requirements.txt
```

### Step 4: Train the Models
Run the training script:

```bash
python train_models.py
```

This will:
- Download CIFAR-10 dataset
- Preprocess the images
- Train 3 models
- Save models in `/models`
- Generate charts in `/static/charts`

### Step 5: Run the Flask Website

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

You can now upload an image and get predictions!

---

## ☁️ Run in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload all project files
3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. Train models:

```python
!python train_models.py
```

5. Run Flask (use a tunnel like `ngrok` or run Streamlit instead):

```python
!python app.py
```

**Recommended for Colab**: Use `streamlit_app.py` (see below).

---

## 🚀 Deploy to Streamlit Cloud

### Step 1: Push Code to GitHub
1. Create a GitHub repository.
2. Upload all project files.
3. Make sure `streamlit_app.py` is in the root of the repository.

### Step 2: Go to Streamlit Cloud
Visit [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.

### Step 3: Create New App
1. Click **"New app"**.
2. Select your GitHub repository.
3. Set **Main file path** to: `streamlit_app.py`
4. Click **Deploy**.

### Step 4: Your URL
After deployment, your website will be live at:

```
https://yourname-cifar10-classifier.streamlit.app/
```

You can change `yourname-cifar10-classifier` to whatever you want.

> ⚠️ **Important**: Streamlit Cloud free tier has limited memory. The first time you open the app, it may need to download CIFAR-10 and load models, which can take 1-2 minutes.

---

## 🧠 Understanding the Models (Easy to Explain)

### 1. Logistic Regression
Think of it as drawing a straight line (or plane) that separates different image classes. It predicts the probability that an image belongs to each class.

### 2. Random Forest
It creates many small "decision trees" and combines their answers. Each tree asks simple yes/no questions about pixel values, and the forest votes on the final class.

### 3. Support Vector Machine (SVM)
SVM finds the best boundary (margin) that separates classes. It tries to keep the gap between classes as wide as possible, which helps it generalize better.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 39.33% | 38.86% | 39.33% | 38.95% |
| Random Forest | 38.63% | 38.03% | 38.63% | 37.88% |
| **Support Vector Machine** | **48.90%** | **48.72%** | **48.90%** | **48.62%** |

**Conclusion**: SVM performed best among the three traditional ML models.

---

## 📝 For Your Report & Presentation

You can include:
- **Problem**: Classify 32×32 RGB images into 10 classes.
- **Dataset**: CIFAR-10 (60,000 images).
- **Preprocessing**: Normalization, standardization, PCA.
- **Models**: Logistic Regression, Random Forest, SVM.
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Best Model**: SVM with ~49% accuracy.

---

## 📬 Need Help?

If you get any errors:
1. Make sure all files are in the same folder.
2. Run `pip install -r requirements.txt` again.
3. Check that `train_models.py` created the `/models` folder.
4. For Streamlit Cloud, check the app logs for missing files.

---

Made with ❤️ for Machine Learning Course Project 2025.
