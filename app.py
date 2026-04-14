# ══════════════════════════════════════════════════════════════════
# Intel Image Classification Web App
# Flask + PyTorch + TensorFlow
# ══════════════════════════════════════════════════════════════════

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Classes
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = 64

# ══════════════════════════════════════════════════════════════════
# PYTORCH MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN architecture from scratch"""
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16x16 -> 8x8
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# ══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════

device = torch.device('cpu')  # Use CPU for deployment

# PyTorch model
pytorch_model = SimpleCNN(num_classes=6)
try:
    pytorch_model.load_state_dict(torch.load('models/conde_ansoumane_model.pth', map_location=device))
    pytorch_model.eval()
    print("✓ PyTorch model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load PyTorch model: {e}")
    pytorch_model = None

# TensorFlow model
try:
    tensorflow_model = tf.keras.models.load_model('models/conde_ansoumane_model.keras')
    print("✓ TensorFlow model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load TensorFlow model: {e}")
    tensorflow_model = None

# ══════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════

# PyTorch transforms
pytorch_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_pytorch(image_path):
    """Preprocess image for PyTorch model"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = pytorch_transform(img).unsqueeze(0)
    return img_tensor

def preprocess_tensorflow(image_path):
    """Preprocess image for TensorFlow model"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# ══════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def predict_pytorch(image_path):
    """Predict using PyTorch model"""
    if pytorch_model is None:
        return None
    
    img_tensor = preprocess_pytorch(image_path)
    
    with torch.no_grad():
        outputs = pytorch_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return {
        'class': CLASSES[predicted.item()],
        'confidence': float(confidence.item() * 100),
        'all_probabilities': {
            CLASSES[i]: float(probabilities[0][i].item() * 100) 
            for i in range(len(CLASSES))
        }
    }

def predict_tensorflow(image_path):
    """Predict using TensorFlow model"""
    if tensorflow_model is None:
        return None
    
    img_array = preprocess_tensorflow(image_path)
    predictions = tensorflow_model.predict(img_array, verbose=0)
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return {
        'class': CLASSES[predicted_class],
        'confidence': float(confidence * 100),
        'all_probabilities': {
            CLASSES[i]: float(predictions[0][i] * 100) 
            for i in range(len(CLASSES))
        }
    }

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', classes=CLASSES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model', 'pytorch')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if model_type == 'pytorch':
            result = predict_pytorch(filepath)
            if result is None:
                return jsonify({'error': 'PyTorch model not available'}), 500
        elif model_type == 'tensorflow':
            result = predict_tensorflow(filepath)
            if result is None:
                return jsonify({'error': 'TensorFlow model not available'}), 500
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Add image path to result
        result['image_path'] = f'uploads/{filename}'
        result['model'] = model_type
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pytorch_loaded': pytorch_model is not None,
        'tensorflow_loaded': tensorflow_model is not None,
        'classes': CLASSES
    })

# ══════════════════════════════════════════════════════════════════
# RUN APP
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*70)
    print("    INTEL IMAGE CLASSIFICATION WEB APP")
    print("="*70)
    print(f"PyTorch model: {'✓ Loaded' if pytorch_model else '✗ Not loaded'}")
    print(f"TensorFlow model: {'✓ Loaded' if tensorflow_model else '✗ Not loaded'}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("="*70)
    print("\nStarting server...")
    print("Open http://127.0.0.1:5000 in your browser\n")
    
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(host="0.0.0.0", port=8080)
