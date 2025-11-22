# 🚁 Helipad Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://helipad-detection.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🏆 **1st Place Solution** - SUPCOM Helipad Detection Challenge (97.7% accuracy)

An AI-powered web application that automatically detects helipads in aerial and satellite imagery using deep learning and transfer learning.

---

## 🚀 Live Demo

**Try it now:** [**🚁 Helipad Detection App**](https://helipad-detection.streamlit.app)

Upload an aerial image and get instant helipad detection with confidence scores!

---

## 🌟 Features

- 🧠 **Transfer Learning**: EfficientNet-B0 pre-trained on ImageNet
- 🔄 **Test-Time Augmentation**: 10x predictions averaged for robust results
- ⚡ **Real-time Detection**: Instant predictions through web interface
- 📊 **Confidence Scores**: Visual probability distributions with progress bars
- 🎨 **User-Friendly Interface**: Built with Streamlit for easy interaction
- ⚙️ **Customizable Settings**: Adjustable confidence threshold and TTA toggle

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Model Architecture** | EfficientNet-B0 |
| **Model Size** | 16.8 MB |
| **Inference Time** | ~2 seconds (with TTA) |
| **Parameters** | 4M total, 1.2M trainable |

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, torchvision
- **Architecture**: EfficientNet-B0
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud
- **Model Hosting**: GitHub Releases
- **Language**: Python 3.11

---

## 📊 Model Architecture
```
INPUT: Aerial Image (224×224×3)
    ↓
EfficientNet-B0 (ImageNet Pre-trained)
    ↓
Freeze first 180 layers (preserve ImageNet knowledge)
    ↓
Fine-tune last 20 layers (adapt to helipads)
    ↓
Custom Classifier:
    ├── Dropout(0.3)
    ├── Linear(1280 → 256)
    ├── ReLU
    ├── Dropout(0.3)
    ├── Linear(256 → 1)
    └── Sigmoid
    ↓
OUTPUT: Probability [0, 1] → Helipad or Not
```

---

## 🔬 Key Techniques

### 1. **Transfer Learning**
- Leveraged EfficientNet-B0 pre-trained on 1.2M ImageNet images
- Fine-tuned only the last 20 layers for helipad-specific features
- Preserved low-level features (edges, textures) from ImageNet

### 2. **Advanced Data Augmentation**
- Random horizontal and vertical flips
- Random rotations (up to 180°)
- Affine transformations (translation, zoom)
- Color jitter (brightness, contrast, saturation)
- Random erasing for occlusion robustness
- Grayscale conversion (occasional)

### 3. **Test-Time Augmentation (TTA)**
- 10 different augmentations applied to each test image
- Predictions averaged for more robust results
- **Impact**: +0.4% accuracy improvement
- Reduces variance and increases confidence

### 4. **Optimal Threshold Tuning**
- Systematically tested thresholds from 0.30 to 0.70
- Found optimal threshold at 0.52
- **Impact**: +0.15% accuracy improvement
- Balances precision and recall

---

## 💻 Run Locally

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/rim373/Helipad-Detection-with-Deep-Learning.git
cd Helipad-Detection-with-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

The model will be downloaded automatically on first run (~17 MB).

---

## 📁 Project Structure
```
helipad-detection/
├── app.py                      # Streamlit web application
├── model.ipynb                 # Training notebook (Kaggle)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file

Model weights hosted on GitHub Releases (downloaded automatically)
```

---

## 🎓 Training Details

### Dataset
- **Source**: SUPCOM Helipad Detection Challenge
- **Classes**: Binary classification (Helipad / No Helipad)
- **Split**: 85% training, 15% validation (stratified)
- **Images**: Aerial and satellite imagery from Google Maps

### Hyperparameters
- **Optimizer**: AdamW
  - Learning rate: 0.001
  - Weight decay: 0.01
- **Scheduler**: CosineAnnealingLR
- **Batch Size**: 64
- **Epochs**: 25
- **Loss Function**: Binary Cross-Entropy (BCELoss)



---

## 🎯 Usage

### Web Interface

1. Visit [https://helipad-detection.streamlit.app](https://helipad-detection.streamlit.app)
2. Upload an aerial or satellite image (JPG, JPEG, PNG)
3. Adjust settings in the sidebar:
   - Toggle Test-Time Augmentation
   - Adjust confidence threshold
4. View results:
   - Helipad detection (Yes/No)
   - Confidence score
   - Detailed analysis

### Tips for Best Results

- Use clear, overhead aerial or satellite images
- Ensure the helipad (if present) is visible and not obscured
- Works best with images similar to Google Maps satellite view
- Enable TTA for more robust predictions on challenging images

---

## 📈 Results & Insights

### Model Strengths
- ✅ High precision (few false positives)
- ✅ Excellent recall (98%)
- ✅ Generalizes well to unseen helipads
- ✅ Robust to different helipad types (circular, square)
- ✅ Handles various lighting conditions

### Key Learnings
1. Transfer learning significantly reduces training time and data requirements
2. Test-Time Augmentation provides measurable accuracy improvements
3. Careful threshold tuning can boost performance
4. Data augmentation is crucial for generalization

---

## 🛡️ Model Robustness

The model handles various challenging scenarios:
- Different helipad shapes (circular, square)
- Various surface materials (concrete, painted)
- Different weather conditions
- Urban vs rural settings
- Partial occlusions
- Different altitudes and angles

---

