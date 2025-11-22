import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Helipad Detection",
    page_icon="🚁",
    layout="centered"
)

# Title and description
st.title("🚁 Helipad Detection System")
st.markdown("""
This AI-powered application detects helipads in aerial/satellite images using deep learning.
Upload an image to get started!
""")

# Model architecture - EXACT MATCH TO YOUR TRAINING CODE
class HelipadClassifier(nn.Module):
    def __init__(self):
        super(HelipadClassifier, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        
        # Freeze early layers - EXACTLY 20 UNFROZEN (same as training)
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier - EXACT MATCH TO TRAINING
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Load model from GitHub Releases
# Load model from GitHub Releases
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = HelipadClassifier()
    
    model_path = 'best_helipad_model.pth'
    
    # Download from GitHub Releases if not exists
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model from GitHub (first time only, ~17MB)..."):
            try:
                url = "https://github.com/rim373/Helipad-Detection-with-Deep-Learning/releases/download/v1.0/best_helipad_model.pth"
                
                st.info(f"Downloading from: {url}")
                
                response = requests.get(url, stream=True, allow_redirects=True)
                
                # Debug: Check what we're getting
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length', '0')
                
                st.info(f"Response status: {response.status_code}")
                st.info(f"Content-Type: {content_type}")
                st.info(f"Content-Length: {content_length} bytes")
                
                # Check if we got HTML instead of binary
                if 'text/html' in content_type:
                    st.error("❌ Got HTML page instead of model file!")
                    st.error("This usually means the file URL is incorrect or the release is not public.")
                    return None, None
                
                response.raise_for_status()
                
                # Download the file
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Check file size
                file_size = os.path.getsize(model_path)
                st.success(f"✅ Model downloaded successfully! ({file_size / 1024 / 1024:.1f} MB)")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Download failed: {e}")
                return None, None
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return None, None
    
    # Load the model
    try:
        # Check if file exists and is not empty
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            st.info(f"Loading model file ({file_size / 1024 / 1024:.1f} MB)...")
            
            # Check if file looks like HTML (starts with <!DOCTYPE or <html)
            with open(model_path, 'rb') as f:
                first_bytes = f.read(100)
                if b'<!DOCTYPE' in first_bytes or b'<html' in first_bytes:
                    st.error("❌ Model file is actually an HTML page!")
                    st.error("Deleting corrupted file...")
                    os.remove(model_path)
                    st.error("Please refresh the page to try downloading again.")
                    return None, None
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # If error, try to show first few bytes of the file for debugging
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    first_100 = f.read(100)
                    st.error(f"First 100 bytes of file: {first_100[:100]}")
            except:
                pass
        
        return None, None
# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction with TTA
def predict_with_tta(model, image, device, num_tta=10):
    """Test-Time Augmentation for robust predictions"""
    model.eval()
    predictions = []
    
    # Standard transform
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # TTA transforms
    tta_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        # Original prediction
        img_tensor = standard_transform(image).unsqueeze(0).to(device)
        output = model(img_tensor)
        predictions.append(output.item())
        
        # TTA predictions
        for _ in range(num_tta - 1):
            img_tensor = tta_transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            predictions.append(output.item())
    
    return np.mean(predictions)

# Load the model
model, device = load_model()

if model is not None:
    # Sidebar information
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **Helipad Detection AI**
    
    This model uses:
    - EfficientNet-B0 architecture
    - Transfer learning
    - Test-Time Augmentation
    - Trained on aerial imagery
    
    **Accuracy:** ~97.7%
    
    **Created for:** SUPCOM Challenge
    """)
    
    st.sidebar.header("⚙️ Settings")
    use_tta = st.sidebar.checkbox("Use Test-Time Augmentation (TTA)", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an aerial/satellite image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear aerial or satellite image"
    )
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            with st.spinner("Analyzing image..."):
                # Make prediction
                if use_tta:
                    probability = predict_with_tta(model, image, device, num_tta=10)
                    st.caption("✨ Using Test-Time Augmentation (10 predictions)")
                else:
                    img_tensor = preprocess_image(image).to(device)
                    with torch.no_grad():
                        probability = model(img_tensor).item()
                    st.caption("⚡ Single prediction")
                
                # Determine prediction
                prediction = 1 if probability > confidence_threshold else 0
                
                # Display results
                st.markdown("---")
                
                if prediction == 1:
                    st.success("✅ **HELIPAD DETECTED!**")
                    st.balloons()
                else:
                    st.error("❌ **NO HELIPAD DETECTED**")
                
                # Confidence score
                st.metric(
                    label="Confidence Score",
                    value=f"{probability:.2%}",
                    delta=f"{abs(probability - confidence_threshold):.2%} from threshold"
                )
                
                # Progress bar visualization
                st.progress(probability)
                
                # Detailed breakdown
                with st.expander("📊 Detailed Analysis"):
                    st.write(f"**Raw Probability:** {probability:.4f}")
                    st.write(f"**Threshold Used:** {confidence_threshold:.2f}")
                    st.write(f"**Prediction:** {'Helipad (1)' if prediction == 1 else 'No Helipad (0)'}")
                    
                    if use_tta:
                        st.write("**Method:** Test-Time Augmentation (10 augmentations)")
                    else:
                        st.write("**Method:** Single forward pass")
                    
                    # Interpretation
                    if probability > 0.9:
                        st.success("🎯 Very confident - strong helipad features detected")
                    elif probability > 0.7:
                        st.info("✓ Confident - likely a helipad")
                    elif probability > 0.5:
                        st.warning("⚠️ Moderate confidence - borderline case")
                    else:
                        st.info("✗ Low confidence - likely not a helipad")
        
        # Additional information
        st.markdown("---")
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Use clear, overhead aerial or satellite images
        - Ensure the helipad (if present) is visible and not obscured
        - Works best with images similar to Google Maps satellite view
        - Enable TTA for more robust predictions
        """)

else:
    st.error("⚠️ Could not load the model. Please check the error messages above.")

