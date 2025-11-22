# 🚁 Helipad Detection System

An AI-powered web application that detects helipads in aerial/satellite images using deep learning.

## 🌟 Features

- Upload aerial/satellite images
- Real-time helipad detection
- Confidence score visualization
- Test-Time Augmentation for robust predictions
- Built with EfficientNet-B0 and Transfer Learning

## 🎯 Model Performance

- **Architecture:** EfficientNet-B0 with custom classifier
- **Training:** Transfer learning on [X] aerial images
- **Validation Accuracy:** 97.7%
- **Competition Ranking:** 1st place (97.7% vs 97.555%)

## 🚀 Try it Online

👉 **[Live Demo](https://your-app-name.streamlit.app)** (Coming soon)

## 💻 Run Locally

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 📁 Project Structure
```
helipad-detector/
├── app.py                     # Streamlit application
├── best_helipad_model.pth     # Trained model weights
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── sample_images/             # Example images
```

## 🧠 Technical Details

### Model Architecture
- **Base Model:** EfficientNet-B0 (pre-trained on ImageNet)
- **Fine-tuning:** Last 40 layers trained
- **Custom Classifier:** 1280 → 512 → 256 → 1
- **Regularization:** Dropout (0.4, 0.4, 0.3)
- **Output:** Sigmoid activation for binary classification

### Key Techniques
1. **Transfer Learning** - Leverage ImageNet knowledge
2. **Data Augmentation** - 9 advanced augmentation techniques
3. **Test-Time Augmentation** - 10 predictions averaged
4. **Optimal Threshold** - Tuned on validation set (0.52)

### Training Details
- **Dataset:** SUPCOM Helipad Detection Challenge
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.01)
- **Scheduler:** CosineAnnealingLR
- **Epochs:** 25
- **Batch Size:** 64

