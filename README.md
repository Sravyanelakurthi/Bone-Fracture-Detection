# Bone Fracture Detection Using Deep Learning

## Project Overview

Accurate and timely detection of bone fractures is critical in medical diagnosis. 
Manual interpretation of X-ray images can be time-consuming and prone to human error.

This project implements Deep Learning models to automatically classify X-ray images as:

- ✅ Fractured
- ❌ Not Fractured

The system compares a Custom CNN model and MobileNet (Transfer Learning) to determine the most effective architecture for fracture detection.

---

## Models Implemented

### Custom CNN
- 3 Convolutional Layers
- Batch Normalization
- Dropout Regularization
- Fully Connected Layers
- Binary Classification Output (Sigmoid)

### MobileNet (Transfer Learning)
- Pretrained MobileNetV3
- Modified Classification Head
- Optimized for lightweight deployment

---

## Model Performance

| Model      | Accuracy |
|------------|----------|
| CNN        | 95.76%   |
| MobileNet  | 85.79%   |

Evaluation Metrics Used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix


---

## Sample Predictions
Green Border → Correct Prediction  
Red Border → Incorrect Prediction  

---

## Project Pipeline

1. Data Collection (Fractured & Non-Fractured X-ray images)
2. Image Preprocessing
   - Resize to 224x224
   - Normalization
   - Data Augmentation
3. Model Training
4. Evaluation
5. Prediction on New Images

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
