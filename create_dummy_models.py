"""
Create Dummy Models for Testing
This script creates simple dummy models when you don't have the actual trained models yet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from pathlib import Path

# Create models directory
Path('models').mkdir(exist_ok=True)

print("Creating dummy models for testing...")

# ══════════════════════════════════════════════════════════════════
# PYTORCH DUMMY MODEL
# ══════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create and save PyTorch model
pytorch_model = SimpleCNN(num_classes=6)
torch.save(pytorch_model.state_dict(), 'models/pytorch_model.pth')
print("✓ PyTorch dummy model saved: models/pytorch_model.pth")

# ══════════════════════════════════════════════════════════════════
# TENSORFLOW DUMMY MODEL
# ══════════════════════════════════════════════════════════════════

tensorflow_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

tensorflow_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tensorflow_model.save('models/tensorflow_model.keras')
print("✓ TensorFlow dummy model saved: models/tensorflow_model.keras")

print("\n" + "="*70)
print("DUMMY MODELS CREATED SUCCESSFULLY")
print("="*70)
print("\nIMPORTANT:")
print("These are untrained dummy models for testing the web interface.")
print("Replace them with your actual trained models before deployment!")
print("\nTo replace:")
print("  1. Train your models using the Jupyter notebook")
print("  2. Save PyTorch model as: models/pytorch_model.pth")
print("  3. Save TensorFlow model as: models/tensorflow_model.keras")
print("="*70 + "\n")
