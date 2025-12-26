from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model architecture (نفس الـ architecture من Colab)
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

# Image transformations
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class names
class_names = ['Active', 'Fatigue']

# Load model
print("Loading model...")

# تحديد المسار الكامل للموديل
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_fatigue_detection_model.pth')

# التحقق من وجود الموديل
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found!\n"
        f"   Expected location: {MODEL_PATH}\n\n"
        f"   Please do one of the following:\n"
        f"   1. Train the model first: cd training && python train_model.py\n"
        f"   2. Download model from Colab and place it in the 'server' folder\n"
    )

model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded successfully from: {MODEL_PATH}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'message': 'Fatigue Detection API is working!',
        'endpoints': {
            'predict': '/predict (POST)'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استقبال البيانات
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        image_data = data['image']
        
        # تحويل base64 لصورة
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # المعالجة
        input_tensor = inference_transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
        
        predicted_label = class_names[predicted_class.item()]
        confidence_percent = confidence.item() * 100
        
        # إرجاع النتيجة
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': confidence_percent,
            'probabilities': {
                'Active': probabilities[0].item() * 100,
                'Fatigue': probabilities[1].item() * 100
            }
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Fatigue Detection API...")
    print("="*50)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)