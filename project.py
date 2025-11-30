import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ===============================================
#   1) Data Cleaning & Exploratory Analysis
# ===============================================

SOURCE_DIR = "Data"

CLASSES = ["Fatigue", "NonFatigue"]

BASE_OUTPUT = "Splitted"
TRAIN_DIR = os.path.join(BASE_OUTPUT, "train")
TEST_DIR = os.path.join(BASE_OUTPUT, "test")
PREDICT_DIR = os.path.join(BASE_OUTPUT, "predict")

for path in [TRAIN_DIR, TEST_DIR, PREDICT_DIR]:
    for cls in CLASSES:
        os.makedirs(os.path.join(path, cls), exist_ok=True)

for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = [img for img in os.listdir(class_dir)
    if img.lower().endswith(('jpg', 'jpeg', 'png'))]

    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    for img in train_imgs:
        shutil.copy(os.path.join(class_dir, img),
                    os.path.join(TRAIN_DIR, cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_dir, img),
                    os.path.join(TEST_DIR, cls, img))
        





# ============================================
#   2) Preprocessing & Data Augmentation
# ============================================


#   1) Transformations

IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


#   2) Custom Dataset

class FatigueDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        classes = ["Fatigue", "NonFatigue"]   # 0 - 1

        for label, folder in enumerate(classes):
            folder_path = os.path.join(root_dir, folder)

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                if img_name.lower().endswith(("jpg", "jpeg", "png")):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


#   3) Loaders

train_dataset = FatigueDataset(root_dir="/content/splitted/train", transform=train_transforms)
test_dataset  = FatigueDataset(root_dir="/content/splitted/test",  transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


#   4) Test Loader

for images, labels in train_loader:
    print("Images batch shape:", images.shape)     # [32, 3, 224, 224]
    print("Labels batch shape:", labels.shape)     # [32]
    break

"""### **TensorFlow + ImageDataGenerator** (Option 2)"""




# Make a folder safely. Won't throw error if it already exists
def safe_makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# Check if a file is an image based on its extension
def is_image_file(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# Load an image with OpenCV and resize it
# Returns None if the image can't be read
def read_image_cv2(path, target_size=(224, 224)):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


# Look at the dataset folder and figure out classes
def analyze_dataset_structure(dataset_path):
    p = Path(dataset_path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Get class folder names
    classes = [d.name for d in p.iterdir() if d.is_dir()]

    # Case: no folders, all images in one folder
    if not classes:
        classes = ['images']
        all_images = [f for f in p.iterdir() if f.is_file() and is_image_file(f)]
        return classes, {'images': all_images}

    # Case: folders exist, each folder is a class
    class_files = {}
    for cls in classes:
        cls_path = p / cls
        files = [f for f in cls_path.rglob('*') if f.is_file() and is_image_file(f)]
        class_files[cls] = sorted(files)

    return sorted(classes), class_files


# Remove images that are corrupted or too small
# Moves them to a separate folder
def remove_corrupted_images(class_files, progress_callback=None):
    corrupted_dir = Path("dataset_corrupted")
    safe_makedirs(corrupted_dir)

    report = {"total_checked": 0, "corrupted_count": 0, "corrupted_files": []}
    cleaned = {}

    for cls, files in class_files.items():
        cleaned[cls] = []

        for f in files:
            report["total_checked"] += 1

            try:
                img = read_image_cv2(f)
                # Skip images that are too small or unreadable
                if img is None or img.shape[0] < 10 or img.shape[1] < 10:
                    raise ValueError("Invalid image")
                cleaned[cls].append(f)

            except:
                report["corrupted_count"] += 1
                report["corrupted_files"].append(str(f))
                try:
                    shutil.move(str(f), corrupted_dir / f.name)
                except:
                    pass

            # Optional callback to show progress in a GUI
            if progress_callback:
                progress_callback(report["total_checked"])

    return cleaned, report


# Make a grid of sample images from each class
def generate_sample_grid(class_files, samples_per_class=4, out_path="samples_grid.png", img_size=(224, 224)):
    classes = list(class_files.keys())
    rows = len(classes)
    cols = samples_per_class
    cell_w, cell_h = img_size

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    canvas = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))

    for i, cls in enumerate(classes):
        files = class_files[cls][:samples_per_class]

        for j in range(cols):
            if j < len(files):
                try:
                    img = Image.open(files[j]).convert('RGB').resize((cell_w, cell_h))
                except:
                    img = Image.new('RGB', (cell_w, cell_h), (50, 50, 50))
            else:
                img = Image.new('RGB', (cell_w, cell_h), (80, 80, 80))

            canvas.paste(img, (j * cell_w, i * cell_h))

    canvas.save(out_path)
    return out_path


# Build data generators for training and validation
def build_generators(data_dir, classes, img_size=(224, 224), batch_size=32, val_split=0.2, seed=42):
    # Generator for training images with augmentation
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        horizontal_flip=True,
        validation_split=val_split
    )

    # Generator for validation images (no augmentation)
    val_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    # Training set generator
    train_gen = train_aug.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=classes,
        class_mode='binary' if len(classes) == 2 else 'categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

    # Validation set generator
    val_gen = val_aug.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        classes=classes,
        class_mode='binary' if len(classes) == 2 else 'categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen





# ============================
#   3) Model Design
# ============================




# ==============================================
#   4) Model Training, Saving & Evaluation
# ==============================================





# ============================
#   5) Testing, Inference
# ============================

model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image)
    image = image.unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    classes = ["Fatigue", "NonFatigue"]
    return classes[predicted.item()]

predict_dir = "Splitted/predict/Fatigue"   # أو NonFatigue حسب ما تعمل

for img in os.listdir(predict_dir):
    img_path = os.path.join(predict_dir, img)
    if img.lower().endswith(("jpg", "jpeg", "png")):
        print(img, "=>", predict_image(model, img_path))

correct = 0
total = 0

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Test Accuracy:", accuracy)

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print(cm)