# -*- coding: utf-8 -*-
"""
Test Single Image - VS Code Version
"""

import os
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================
# Paths Configuration
# ============================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'server', 'best_fatigue_detection_model.pth')

# التحقق من وجود الموديل
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\n❌ Model not found!\n   Path: {MODEL_PATH}\n   Please train the model first!")

# ============================================
# Device & Model Setup
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}\n")

# Model architecture
def get_model(num_classes=2):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.7, inplace=True),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.7, inplace=True),
        nn.Linear(512, num_classes)
    )
    
    return model

# Load model
print("📦 Loading model...")
model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!\n")

# Image transformations
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = ['Active', 'Fatigue']

# ============================================
# Test Image
# ============================================

print("="*70)
print("🖼️  IMAGE INFERENCE TEST")
print("="*70)

# حط مسار الصورة هنا
IMAGE_PATH = input("\nEnter image path (or drag & drop the image here): ").strip().strip('"')

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"\n❌ Image not found: {IMAGE_PATH}")

print(f"\n📂 Loading image: {IMAGE_PATH}")

# Load and display image
image = Image.open(IMAGE_PATH).convert("RGB")

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis('off')
plt.title("Test Image", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'test_image_preview.png'))
plt.show()

# Prepare image for model
input_tensor = inference_transforms(image)
input_batch = input_tensor.unsqueeze(0).to(device)

# Prediction
print("\n🔮 Running prediction...")
with torch.no_grad():
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_class = torch.max(probabilities, 0)

predicted_label = class_names[predicted_class.item()]
confidence_percent = confidence.item() * 100

# Display results
print("\n" + "="*70)
print("📊 PREDICTION RESULTS")
print("="*70)
print(f"🎯 Prediction: {predicted_label}")
print(f"📈 Confidence: {confidence_percent:.2f}%")
print("\n📊 Class Probabilities:")

for i, name in enumerate(class_names):
    prob = probabilities[i].item() * 100
    bar = '█' * int(prob // 2)
    print(f"   {name:8s}: {prob:5.2f}%  {bar}")

print("="*70)