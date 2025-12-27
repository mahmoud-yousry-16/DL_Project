# Fatigue Detection System

A deep learning-based system for detecting fatigue in human faces using EfficientNet-B0 with Flask API and web interface.

---

## Problem Definition

### Overview
Fatigue detection is a critical safety concern across multiple domains. Traditional methods rely on manual observation, wearable sensors, or self-reporting, all of which have significant limitations in accuracy, scalability, and user comfort.

### The Challenge
- Traffic accidents caused by drowsy driving
- Workplace injuries in high-risk environments
- Reduced productivity and performance
- Health complications from chronic fatigue

### Our Solution
An automated, non-intrusive computer vision system that analyzes facial images to classify individuals as Active or Fatigued. The system provides real-time predictions with confidence scores through a simple web interface.

### Applications
- Transportation: Monitor driver alertness to prevent accidents
- Industrial Safety: Track worker fatigue in hazardous environments
- Healthcare: Assess patient energy and recovery levels
- Education: Monitor student engagement and attention

---

## System Overview

### Architecture
The system consists of three main components:
1. Deep learning model (EfficientNet-B0) trained on facial images
2. Flask REST API for model inference
3. Web interface for user interaction

### Workflow
1. User uploads an image through the web interface
2. Image is sent to Flask API as base64 encoded data
3. API preprocesses the image and runs model inference
4. Model returns prediction (Active/Fatigue) with confidence score
5. Results are displayed in the web interface with probability distribution

### Model Design
**Base Architecture**: EfficientNet-B0 pre-trained on ImageNet

**Custom Classifier**:
- Dropout layer (p=0.7) for regularization
- Fully connected layer (1280 to 512 features)
- Batch normalization for training stability
- GELU activation function
- Additional dropout layer (p=0.7)
- Output layer (512 to 2 classes)

**Training Configuration**:
- Optimizer: AdamW with weight decay (1e-3)
- Learning rate: 3e-4 to 8e-4 using OneCycleLR scheduler
- Batch size: 32
- Maximum epochs: 25 with early stopping (patience=7)
- Loss function: CrossEntropyLoss with label smoothing (0.1)

**Data Augmentation**:
- Random horizontal flip (50%)
- Random vertical flip (20%)
- Random rotation (±30 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations (translation, scale, shear)
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

---

## Key Insights

### What Works Well
- Transfer learning with pre-trained weights significantly improves accuracy and reduces training time
- Heavy data augmentation prevents overfitting and improves model generalization
- OneCycleLR scheduler accelerates convergence compared to static learning rates
- High dropout rates (0.7) effectively combat overfitting on limited datasets
- Label smoothing improves model calibration and confidence scores

### Lessons Learned
- Data quality and preprocessing are as important as model architecture
- Proper regularization techniques are essential for preventing overfitting
- Learning rate scheduling has a significant impact on final model performance
- User interface design and API integration require careful planning
- Model interpretability and confidence scores are crucial for real-world deployment

---

## Installation & Usage

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/fatigue-detection.git
cd fatigue-detection
pip install -r requirements.txt
```

### Training the Model
```bash
cd training
python train_model.py
```

### Running the Application
```bash
cd server
python app.py
```
Then open `client/index.html` in your browser and enter API URL: `http://localhost:5000`

---

## Project Structure

```
fatigue-detection/
├── data/images/
│   ├── train/
│   │   ├── active/
│   │   └── fatigue/
│   └── test/
│       ├── active/
│       └── fatigue/
├── training/
│   ├── train_model.py
│   └── test_single_image.py
├── server/
│   ├── app.py
│   └── best_fatigue_detection_model.pth
├── client/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

---

## Team Contributions

### Step 1: Problem Definition & Data Collection

**Responsibilities**:
- Defined the problem statement and its real-world importance
- Researched existing solutions and their limitations
- Documented project objectives and success criteria
- Described dataset characteristics (number of images, classes, general details)

---

### Step 2: Data Cleaning & Exploratory Analysis
**Member**: [Abdulrahman]

**Responsibilities**:
- Verified data integrity and removed corrupted images
- Analyzed class distribution and identified imbalances
- Displayed sample images from each category
- Documented findings for project report

---

### Step 3: Preprocessing & Data Augmentation
**Member**: [Ahmed]

**Responsibilities**:
- Implemented image resizing to 224x224 pixels
- Applied normalization (0-1 range)
- Designed and implemented data augmentation (rotation, flip, zoom)
- Created data generators and loaders

---

### Step 4: Model Design, Training & Evaluation
**Member**: [Ala'a & Malak]

**Responsibilities**:
- Selected model architecture (EfficientNet-B0)
- Built custom classifier layers (GlobalPooling, Dense, Dropout, Output)
- Configured model compilation (loss function, optimizer, metrics)
- Documented architectural decisions and rationale
- Implemented training loop with validation
- Configured ModelCheckpoint and EarlyStopping callbacks
- Generated training curves (accuracy and loss over epochs)
- Evaluated model using confusion matrix and classification report
- Saved best performing model checkpoint

---

### Step 5: Model Optimization
**Member**: [Hanaa]

**Responsibilities**:
- Performed hyperparameter tuning
- Optimized learning rate and batch size
- Analyzed model performance and identified improvements
- Documented optimization process and results

---

### Step 6: API, Inference & GUI
**Member**: [Mahmoud & Youssif]

**Responsibilities**:
- Tested model on new unseen images
- Developed Flask REST API for model inference
- Built web interface (HTML, CSS, JavaScript)
- Integrated frontend with backend API
- Documented API endpoints and usage instructions

---

## Results

### Training Performance
| Metric | Value |
|--------|-------|
| Best Validation Accuracy | [INSERT]% |
| Training Accuracy | [INSERT]% |
| Final Training Loss | [INSERT] |
| Final Validation Loss | [INSERT] |

### Test Set Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Active | [INSERT]% | [INSERT]% | [INSERT]% |
| Fatigue | [INSERT]% | [INSERT]% | [INSERT]% |



## Future Improvements

- Real-time video processing with webcam integration
- Mobile application deployment for Android and iOS
- Multi-level fatigue classification (mild, moderate, severe)
- Additional features: eye tracking, yawn detection, head pose estimation
- Cloud deployment for scalable access



## Technologies Used

- PyTorch 2.1.0
- Flask 3.0.0
- EfficientNet-B0
- HTML5, CSS3, JavaScript
- scikit-learn
- PIL, NumPy, Matplotlib