 # AI vs Real Human Face Detection System - Comprehensive README

## 📌 Project Overview

This advanced facial identity verification system distinguishes between AI-generated and real human faces with state-of-the-art accuracy (up to **99.39%**). The system implements both traditional machine learning and deep learning approaches, providing a comprehensive solution for facial authenticity verification.

## 🏆 Key Results Summary

| Metric                     | Value   |
|----------------------------|---------|
| **Best Overall Accuracy**  | 99.39%  |
| **Best ML Model**          | SVM (RBF) |
| **Best DL Model**          | ResNet50 |
| **Dataset Size**           | 9,000 images (4,500 AI/4,500 Real) |
| **Feature Dimensions**     | 81 advanced features per image |

## 🔍 Detailed Performance Analysis

### Machine Learning Models Performance

| Model                  | Accuracy | Precision | Recall | F1-Score | Training Time |
|------------------------|----------|-----------|--------|----------|---------------|
| **SVM (RBF)**          | 99.39%   | 0.99      | 1.00   | 0.99     | Moderate      |
| Random Forest          | 98.44%   | 0.98      | 0.99   | 0.99     | Fast          |
| SVM (Linear)           | 98.50%   | 0.99      | 0.98   | 0.99     | Fast          |
| Logistic Regression    | 98.50%   | 0.99      | 0.98   | 0.99     | Fast          |
| Gradient Boosting      | 97.33%   | 0.97      | 0.97   | 0.97     | Moderate      |
| K-Nearest Neighbors    | 94.61%   | 0.95      | 0.94   | 0.94     | Fast          |
| Naive Bayes            | 92.56%   | 0.93      | 0.93   | 0.93     | Very Fast     |

**Confusion Matrix for Best ML Model (SVM-RBF):**
```
              Precision  Recall  F1-Score  Support

        Real     1.00      0.99      0.99      900
AI Generated    0.99      1.00      0.99      900

    Accuracy                         0.99     1800
   Macro Avg     0.99      0.99      0.99     1800
Weighted Avg     0.99      0.99      0.99     1800
```

### Deep Learning Models Performance

| Model                          | Accuracy | Parameters  | Training Time | Batch Size |
|--------------------------------|----------|-------------|---------------|------------|
| **ResNet50 Transfer Learning** | 98.50%   | 24,768,642  | ~38s/epoch    | 32         |
| DenseNet121 Transfer Learning  | 98.39%   | 7,694,146   | ~38s/epoch    | 32         |
| MobileNetV2 Transfer Learning  | 96.11%   | 3,045,698   | ~16s/epoch    | 32         |
| VGG16 Transfer Learning        | 50.00%   | 15,109,186  | ~30s/epoch    | 32         |

**Training Details:**
- Epochs: 20 (with early stopping)
- Optimizer: Adam (lr=0.0001)
- Image Size: 64x64 (resized from original 128x128 for memory efficiency)
- Validation Split: 20%

## 📂 Dataset Information

**Structure:**
```
Human Faces Dataset/
├── AI-Generated Images/ (4,630 images)
└── Real Images/ (5,000 images)
```

**Characteristics:**
- Balanced dataset (4,500 samples per class used)
- Various image formats supported (.jpg, .png, .bmp, .tiff, .webp)
- Preprocessed to 128x128 resolution for ML, 64x64 for DL

## ⚙️ Technical Implementation

### Feature Extraction Pipeline
The system extracts **81 advanced features** per image, including:
- Basic statistical features (mean, std, min, max, etc.)
- Color channel distributions
- Gradient features (edge detection)
- Texture analysis
- Frequency domain characteristics
- Brightness/contrast metrics
- Color distribution histograms

### Model Architectures

**Machine Learning:**
- SVM with RBF kernel (C=10, gamma='scale')
- Random Forest (300 estimators, max_depth=25)
- Gradient Boosting (300 estimators, max_depth=15)
- Logistic Regression (C=10)

**Deep Learning:**
- **ResNet50** (Transfer Learning)
  - GlobalAveragePooling2D
  - Dense(512, relu) + Dropout(0.5)
  - Dense(256, relu) + Dropout(0.3)
  - Output layer (softmax)

- **DenseNet121** (Transfer Learning)
  - Similar architecture to ResNet50

### Training Configuration
- **ML Models:**
  - 80/20 train-test split
  - Feature scaling for SVM/KNN/LR
  - Random seed: 42 for reproducibility

- **DL Models:**
  - Early stopping (patience=5)
  - ReduceLROnPlateau (factor=0.2, patience=3)
  - Batch size: 32
  - Adam optimizer (lr=0.0001)

## 💾 Saved Models

The system automatically saves the best performing models to `saved_models/`:
1. `best_ml_model.pkl` - SVM (RBF) classifier
2. `scaler.pkl` - Feature scaler
3. `best_dl_model.h5` - ResNet50 transfer learning model

**Note:** The DL model is saved in HDF5 format (legacy). For future compatibility, consider using the native Keras format (`.keras`).

## 🛠️ Usage Examples

### Single Image Prediction
```python
result = analyzer.predict_single_image("test_image.jpg")
# Returns: "Prediction: AI Generated (Confidence: 99.2%)"
```

### Quick Test with Saved Model
```python
from sklearn.externals import joblib

model = joblib.load("saved_models/best_ml_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")

# Preprocess and predict new image
```

## 📈 Performance Insights

1. **SVM (RBF) outperformed all other models** including deep learning approaches
2. **ResNet50 achieved the best DL accuracy** (98.50%), slightly better than DenseNet121
3. **VGG16 failed to learn** (50% accuracy - random guessing), suggesting architecture incompatibility
4. **Traditional ML trained much faster** than DL while achieving better accuracy
5. **Feature engineering proved crucial** - the 81 carefully crafted features enabled high ML performance

## 🚀 Future Improvements

1. **Experiment with newer architectures**:
   - Vision Transformers (ViT)
   - EfficientNet variants
   - Diffusion model detectors

2. **Enhanced feature engineering**:
   - Add Fourier transform features
   - Incorporate facial landmark analysis
   - Include wavelet transform features

3. **System optimizations**:
   - Convert to native Keras format
   - Implement ONNX runtime for faster inference
   - Add batch prediction capabilities

4. **Explainability**:
   - Add SHAP/LIME explanations
   - Visualize important features
   - Generate saliency maps for DL models

5. **Deployment**:
   - Create Flask/FastAPI web interface
   - Develop mobile app integration
   - Dockerize the solution

## 📜 License

This project is open-source and available under the MIT License.

## 🔗 Contact

For questions or collaborations, please contact the project maintainers.
