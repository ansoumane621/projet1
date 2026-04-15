# # ══════════════════════════════════════════════════════════════════
# # Intel Image Classification Web App
# # Flask + PyTorch + TensorFlow
# # ══════════════════════════════════════════════════════════════════

# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# from pathlib import Path

# # ══════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════════

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # Create upload folder
# Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# # Classes
# CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# IMG_SIZE = 64

# # ══════════════════════════════════════════════════════════════════
# # PYTORCH MODEL DEFINITION
# # ══════════════════════════════════════════════════════════════════

# class SimpleCNN(nn.Module):
#     """
#     CNN simple à 4 blocs convolutifs + 2 couches denses.
#     Architecture :
#         Bloc 1 : Conv(3→32)  + BN + ReLU + MaxPool
#         Bloc 2 : Conv(32→64) + BN + ReLU + MaxPool
#         Bloc 3 : Conv(64→128)+ BN + ReLU + MaxPool
#         Bloc 4 : Conv(128→256)+BN + ReLU + MaxPool
#         Classifier : Flatten → FC(1024) → Dropout → FC(6)
#     """

#     def __init__(self, num_classes=6):
#         super(SimpleCNN, self).__init__()

#         # ── BLOC CONVOLUTIF 1 ─────────────────────────────────
#         # Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         # → entrée : image RGB (3 canaux)
#         # → sortie : 32 feature maps (cartes de caractéristiques)
#         # padding=1 conserve la taille spatiale (64x64 → 64x64)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),   # Extraction de 32 features
#             nn.BatchNorm2d(32),                            # Normalise les activations → stabilise l'apprentissage
#             nn.ReLU(inplace=True),                         # Activation non-linéaire (annule les valeurs négatives)
#             nn.MaxPool2d(kernel_size=2, stride=2)          # Réduit de moitié : 64x64 → 32x32
#         )

#         # ── BLOC CONVOLUTIF 2 ─────────────────────────────────
#         # Entrée : 32 feature maps 32x32
#         # Sortie : 64 feature maps 16x16
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Passe de 32 à 64 filtres
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)          # 32x32 → 16x16
#         )

#         # ── BLOC CONVOLUTIF 3 ─────────────────────────────────
#         # Entrée : 64 feature maps 16x16
#         # Sortie : 128 feature maps 8x8
#         self.block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Passe de 64 à 128 filtres
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)           # 16x16 → 8x8
#         )

#         # ── BLOC CONVOLUTIF 4 ─────────────────────────────────
#         # Entrée : 128 feature maps 8x8
#         # Sortie : 256 feature maps 4x4
#         self.block4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Passe de 128 à 256 filtres
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)            # 8x8 → 4x4
#         )

#         # ── CLASSIFIER (couches entièrement connectées) ───────
#         # Après les 4 blocs conv : 256 feature maps de taille 4x4
#         # → aplati en vecteur de taille 256 * 4 * 4 = 4096
#         self.classifier = nn.Sequential(
#             nn.Flatten(),                         # Aplatit le tenseur 3D → vecteur 1D (256*4*4 = 4096)
#             nn.Linear(256 * 4 * 4, 1024),        # Couche dense : 4096 → 1024 neurones
#             nn.ReLU(inplace=True),                # Activation non-linéaire
#             nn.Dropout(p=0.4),                    # Éteint aléatoirement 50% des neurones (régularisation)
#             nn.Linear(1024, num_classes)          # Couche de sortie : 1024 → 6 scores (un par classe)
#             # Pas de Softmax ici car CrossEntropyLoss l'applique automatiquement
#         )

#     def forward(self, x):
#         """
#         Propagation avant : x passe dans les blocs dans l'ordre.
#         x : tenseur d'entrée de forme (batch_size, 3, 64, 64)
#         """
#         x = self.block1(x)       # (B, 3, 64, 64) → (B, 32, 32, 32)
#         x = self.block2(x)       # (B, 32, 32, 32) → (B, 64, 16, 16)
#         x = self.block3(x)       # (B, 64, 16, 16) → (B, 128, 8, 8)
#         x = self.block4(x)       # (B, 128, 8, 8) → (B, 256, 4, 4)
#         x = self.classifier(x)   # (B, 256, 4, 4) → (B, 6)
#         return x

# # ══════════════════════════════════════════════════════════════════
# # LOAD MODELS
# # ══════════════════════════════════════════════════════════════════

# device = torch.device('cpu')  # Use CPU for deployment

# # PyTorch model
# pytorch_model = SimpleCNN(num_classes=6)
# try:
#     pytorch_model.load_state_dict(torch.load('models/conde_ansoumane_model.pth', map_location=device))
#     pytorch_model.eval()
#     print("✓ PyTorch model loaded successfully")
# except Exception as e:
#     print(f"⚠ Warning: Could not load PyTorch model: {e}")
#     pytorch_model = None

# # TensorFlow model
# try:
#     tensorflow_model = tf.keras.models.load_model('models/conde_ansoumane_model.keras')
#     print("✓ TensorFlow model loaded successfully")
# except Exception as e:
#     print(f"⚠ Warning: Could not load TensorFlow model: {e}")
#     tensorflow_model = None

# # ══════════════════════════════════════════════════════════════════
# # IMAGE PREPROCESSING
# # ══════════════════════════════════════════════════════════════════

# # PyTorch transforms
# pytorch_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def preprocess_pytorch(image_path):
#     """Preprocess image for PyTorch model"""
#     img = Image.open(image_path).convert('RGB')
#     img_tensor = pytorch_transform(img).unsqueeze(0)
#     return img_tensor

# def preprocess_tensorflow(image_path):
#     """Preprocess image for TensorFlow model"""
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     img_array = np.array(img) / 255.0
#     # Normalize with ImageNet stats
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_array = (img_array - mean) / std
#     img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
#     return img_array

# # ══════════════════════════════════════════════════════════════════
# # PREDICTION FUNCTIONS
# # ══════════════════════════════════════════════════════════════════

# def predict_pytorch(image_path):
#     """Predict using PyTorch model"""
#     if pytorch_model is None:
#         return None
    
#     img_tensor = preprocess_pytorch(image_path)
    
#     with torch.no_grad():
#         outputs = pytorch_model(img_tensor)
#         probabilities = F.softmax(outputs, dim=1)
#         confidence, predicted = torch.max(probabilities, 1)
    
#     return {
#         'class': CLASSES[predicted.item()],
#         'confidence': float(confidence.item() * 100),
#         'all_probabilities': {
#             CLASSES[i]: float(probabilities[0][i].item() * 100) 
#             for i in range(len(CLASSES))
#         }
#     }

# def predict_tensorflow(image_path):
#     """Predict using TensorFlow model"""
#     if tensorflow_model is None:
#         return None
    
#     img_array = preprocess_tensorflow(image_path)
#     predictions = tensorflow_model.predict(img_array, verbose=0)
    
#     predicted_class = np.argmax(predictions[0])
#     confidence = predictions[0][predicted_class]
    
#     return {
#         'class': CLASSES[predicted_class],
#         'confidence': float(confidence * 100),
#         'all_probabilities': {
#             CLASSES[i]: float(predictions[0][i] * 100) 
#             for i in range(len(CLASSES))
#         }
#     }

# # ══════════════════════════════════════════════════════════════════
# # HELPER FUNCTIONS
# # ══════════════════════════════════════════════════════════════════

# def allowed_file(filename):
#     """Check if file extension is allowed"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # ══════════════════════════════════════════════════════════════════
# # ROUTES
# # ══════════════════════════════════════════════════════════════════

# @app.route('/')
# def index():
#     """Main page"""
#     return render_template('index.html', classes=CLASSES)

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle image upload and prediction"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
#     model_type = request.form.get('model', 'pytorch')
    
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
    
#     try:
#         # Save uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         # Make prediction
#         if model_type == 'pytorch':
#             result = predict_pytorch(filepath)
#             if result is None:
#                 return jsonify({'error': 'PyTorch model not available'}), 500
#         elif model_type == 'tensorflow':
#             result = predict_tensorflow(filepath)
#             if result is None:
#                 return jsonify({'error': 'TensorFlow model not available'}), 500
#         else:
#             return jsonify({'error': 'Invalid model type'}), 400
        
#         # Add image path to result
#         result['image_path'] = f'uploads/{filename}'
#         result['model'] = model_type
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'ok',
#         'pytorch_loaded': pytorch_model is not None,
#         'tensorflow_loaded': tensorflow_model is not None,
#         'classes': CLASSES
#     })

# # ══════════════════════════════════════════════════════════════════
# # RUN APP
# # ══════════════════════════════════════════════════════════════════

# if __name__ == '__main__':
#     print("\n" + "="*70)
#     print("    INTEL IMAGE CLASSIFICATION WEB APP")
#     print("="*70)
#     print(f"PyTorch model: {'✓ Loaded' if pytorch_model else '✗ Not loaded'}")
#     print(f"TensorFlow model: {'✓ Loaded' if tensorflow_model else '✗ Not loaded'}")
#     print(f"Classes: {', '.join(CLASSES)}")
#     print("="*70)
#     print("\nStarting server...")
#     print("Open http://127.0.0.1:5000 in your browser\n")
    
#     # app.run(debug=True, host='0.0.0.0', port=5000)
#     app.run(host="0.0.0.0", port=8080)
# ═══════════════════════════════════════════════════════════════
# Intel Image Classification - OPTIMIZED FOR FLY.IO
# PyTorch ONLY (lightweight + stable)
# ═══════════════════════════════════════════════════════════════

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = 64

device = torch.device("cpu")  # FORCE CPU (important Fly.io)

# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """
    CNN simple à 4 blocs convolutifs + 2 couches denses.
    Architecture :
        Bloc 1 : Conv(3→32)  + BN + ReLU + MaxPool
        Bloc 2 : Conv(32→64) + BN + ReLU + MaxPool
        Bloc 3 : Conv(64→128)+ BN + ReLU + MaxPool
        Bloc 4 : Conv(128→256)+BN + ReLU + MaxPool
        Classifier : Flatten → FC(1024) → Dropout → FC(6)
    """

    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()

        # ── BLOC CONVOLUTIF 1 ─────────────────────────────────
        # Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # → entrée : image RGB (3 canaux)
        # → sortie : 32 feature maps (cartes de caractéristiques)
        # padding=1 conserve la taille spatiale (64x64 → 64x64)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # Extraction de 32 features
            nn.BatchNorm2d(32),                            # Normalise les activations → stabilise l'apprentissage
            nn.ReLU(inplace=True),                         # Activation non-linéaire (annule les valeurs négatives)
            nn.MaxPool2d(kernel_size=2, stride=2)          # Réduit de moitié : 64x64 → 32x32
        )

        # ── BLOC CONVOLUTIF 2 ─────────────────────────────────
        # Entrée : 32 feature maps 32x32
        # Sortie : 64 feature maps 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Passe de 32 à 64 filtres
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # 32x32 → 16x16
        )

        # ── BLOC CONVOLUTIF 3 ─────────────────────────────────
        # Entrée : 64 feature maps 16x16
        # Sortie : 128 feature maps 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Passe de 64 à 128 filtres
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)           # 16x16 → 8x8
        )

        # ── BLOC CONVOLUTIF 4 ─────────────────────────────────
        # Entrée : 128 feature maps 8x8
        # Sortie : 256 feature maps 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Passe de 128 à 256 filtres
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)            # 8x8 → 4x4
        )

        # ── CLASSIFIER (couches entièrement connectées) ───────
        # Après les 4 blocs conv : 256 feature maps de taille 4x4
        # → aplati en vecteur de taille 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # Aplatit le tenseur 3D → vecteur 1D (256*4*4 = 4096)
            nn.Linear(256 * 4 * 4, 1024),        # Couche dense : 4096 → 1024 neurones
            nn.ReLU(inplace=True),                # Activation non-linéaire
            nn.Dropout(p=0.4),                    # Éteint aléatoirement 50% des neurones (régularisation)
            nn.Linear(1024, num_classes)          # Couche de sortie : 1024 → 6 scores (un par classe)
            # Pas de Softmax ici car CrossEntropyLoss l'applique automatiquement
        )

    def forward(self, x):
        """
        Propagation avant : x passe dans les blocs dans l'ordre.
        x : tenseur d'entrée de forme (batch_size, 3, 64, 64)
        """
        x = self.block1(x)       # (B, 3, 64, 64) → (B, 32, 32, 32)
        x = self.block2(x)       # (B, 32, 32, 32) → (B, 64, 16, 16)
        x = self.block3(x)       # (B, 64, 16, 16) → (B, 128, 8, 8)
        x = self.block4(x)       # (B, 128, 8, 8) → (B, 256, 4, 4)
        x = self.classifier(x)   # (B, 256, 4, 4) → (B, 6)
        return x

# ═══════════════════════════════════════════════════════════════
# SAFE MODEL LOADING (CRITICAL FIX)
# ═══════════════════════════════════════════════════════════════

pytorch_model = None

try:
    model_path = "models/conde_ansoumane_model.pth"

    if os.path.exists(model_path):
        pytorch_model = SimpleCNN(num_classes=6)
        pytorch_model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        pytorch_model.eval()
        print("✓ PyTorch model loaded successfully")
    else:
        print("⚠ Model file not found:", model_path)

except Exception as e:
    print("✗ Model loading failed:", str(e))
    pytorch_model = None

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# ═══════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ═══════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════

def predict(image_path):
    if pytorch_model is None:
        return None

    x = preprocess(image_path)

    with torch.no_grad():
        outputs = pytorch_model(x)
        probs = F.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    return {
        "class": CLASSES[pred.item()],
        "confidence": float(conf.item() * 100),
        "all_probabilities": {
            CLASSES[i]: float(probs[0][i].item() * 100)
            for i in range(len(CLASSES))
        }
    }

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)

@app.route("/predict", methods=["POST"])
def predict_route():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = predict(filepath)

        if result is None:
            return jsonify({"error": "Model not loaded"}), 500

        result["image_path"] = f"uploads/{filename}"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pytorch_model is not None,
        "device": str(device)
    })

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 8080))
    # print("Starting on port:", port)
    # app.run(host="0.0.0.0", port=port)
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)