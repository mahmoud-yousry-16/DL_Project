import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ============================================
# تحديد المسارات (Paths Configuration)
# ============================================

# المسار الحالي
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# مسارات الداتا
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'images')
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
TEST_DIR = os.path.join(DATA_ROOT, 'test')

# مسار حفظ الموديل
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'server', 'best_fatigue_detection_model.pth')

print("\n" + "="*70)
print("🔍 CHECKING PATHS...")
print("="*70)
print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Root: {DATA_ROOT}")
print(f"📂 Train Directory: {TRAIN_DIR}")
print(f"📂 Test Directory: {TEST_DIR}")
print(f"💾 Model Save Path: {MODEL_SAVE_PATH}")

# التحقق من وجود المجلدات
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"\n❌ ERROR: Train folder not found!\n   Path: {TRAIN_DIR}\n   Please create: data/images/train/")
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"\n❌ ERROR: Test folder not found!\n   Path: {TEST_DIR}\n   Please create: data/images/test/")

# إنشاء مجلد server إذا لم يكن موجوداً
server_dir = os.path.dirname(MODEL_SAVE_PATH)
if not os.path.exists(server_dir):
    os.makedirs(server_dir)
    print(f"✅ Created server directory: {server_dir}")

print("✅ All paths verified successfully!\n")

# ============================================
# Data Cleaning & Analysis
# ============================================

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}\n")

# Hyperparameters
IMG_SIZE = 224
batch_size = 32

# Data Augmentations
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Custom Dataset
class FatigueDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = ["active", "fatigue"]  # 0: active, 1: fatigue

        for label, class_name in enumerate(self.classes):
            folder_path = os.path.join(root_dir, class_name)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.images)} images from {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = test_transforms(image)

        return image, torch.tensor(label, dtype=torch.long)

# Create Datasets
print("="*70)
print("📊 LOADING DATASETS...")
print("="*70)

train_dataset = FatigueDataset(
    root_dir=TRAIN_DIR,
    transform=train_transforms
)

test_dataset = FatigueDataset(
    root_dir=TEST_DIR,
    transform=test_transforms
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # Windows: use 0, Linux/Mac: use 2-4
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("✅ Preprocessing completed\n")

# ============================================
# Model Design
# ============================================

print("="*70)
print("🏗️  MODEL DESIGN...")
print("="*70)

def get_model(device, num_classes=2):
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

    return model.to(device)

# Create model
model = get_model(device, num_classes=2)
print("✅ Model design completed\n")

# ============================================
# Model Training
# ============================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training Hyperparameters
LR = 3e-4
MAX_LR = 8e-4
EPOCHS = 25
PATIENCE = 7

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=1e4
)

# Training Loop
best_acc = 0.0
best_epoch = 0
patience_counter = 0

print("="*70)
print("🚀 STARTING TRAINING...")
print("="*70)

for epoch in range(EPOCHS):
    print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
    print("-" * 70)

    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(test_dataset)
    val_acc = correct / total

    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n📊 Results:")
    print(f"   LR: {current_lr:.2e}")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   💾 Saved New Best Model (Val Acc: {best_acc:.4f})")
        print(f"   📁 Location: {MODEL_SAVE_PATH}")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\n⏹️  Early Stopping! No improvement for {PATIENCE} epochs.")
        break

print("\n" + "="*70)
print("✅ TRAINING FINISHED!")
print(f"🏆 Best Validation Accuracy: {best_acc:.4f} at Epoch {best_epoch}")
print(f"💾 Model saved at: {MODEL_SAVE_PATH}")
print("="*70)

# ============================================
# Model Testing & Inference
# ============================================

print("\n" + "="*70)
print("🧪 MODEL EVALUATION...")
print("="*70)

model.eval()
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print("✅ Loaded the best saved model for evaluation.\n")

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Active', 'Fatigue'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(PROJECT_ROOT, 'confusion_matrix.png'))
print(f"📊 Confusion matrix saved to: {PROJECT_ROOT}/confusion_matrix.png")
plt.show()

# Classification Report
print("\n📈 Classification Report:")
print("="*70)
print(classification_report(all_labels, all_preds,
                            target_names=['Active', 'Fatigue'],
                            digits=4))

print("\n" + "="*70)
print("✅ ALL DONE!")
print("="*70)
print(f"🎯 You can now use the model at: {MODEL_SAVE_PATH}")
print("🚀 Run the Flask API from the 'server' folder to use it!")
print("="*70)