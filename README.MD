# 🩺 Multi-Disease Prediction Model from Chest X-Ray Images  
### Deep Learning | Explainable AI | Medical Imaging

---

## 📖 Project Overview
The **Multi-Disease Prediction Model** is an end-to-end medical image analysis system that automatically detects **multiple thoracic diseases from chest X-ray images** using deep learning techniques.

Unlike traditional single-disease classifiers, this project supports **multi-label classification**, enabling the detection of **multiple co-existing diseases from a single X-ray**. To improve trust and interpretability, the system integrates **Explainable AI (Grad-CAM)** to highlight pathological regions influencing the model’s predictions.

The project also includes a **web-based application** that allows users to upload X-ray images and receive real-time predictions along with visual explanations.

---

## 🎯 Objectives
- Build a robust deep-learning model capable of detecting multiple thoracic diseases
- Support realistic medical scenarios using **multi-label classification**
- Improve interpretability with **visual explanations (Grad-CAM)**
- Provide an end-to-end pipeline: dataset → training → evaluation → deployment
- Create a reusable and extensible architecture for future medical AI research

---

## 🧠 Diseases Covered
- Pneumonia  
- Cardiomegaly  
- Edema  
- Effusion  
- Infiltration  

*(The architecture is easily extendable to more diseases.)*

---

## 📊 Dataset Information
### Primary Dataset
- **NIH Chest X-ray14 Dataset**
- Contains **112,120 frontal X-ray images** from over 30,000 patients
- Includes **14 thoracic disease labels**
- Publicly available for research purposes
Dataset used: NIH Chest X-ray14 – a large-scale public dataset for multi-label thoracic disease classification.


🔗 Dataset link:  
https://www.kaggle.com/nih-chest-xrays/data

> ⚠️ The dataset is NOT included in this repository due to size constraints.  
> Please download it manually and place it inside the `data/raw/` directory.

---

## 🧪 Data Preprocessing
- Image resizing and normalization
- Pixel intensity scaling
- Data augmentation:
  - Horizontal flipping
  - Rotation
  - Zoom
  - Contrast enhancement (optional)
- Multi-label binarization
- Train/Validation/Test split with patient-wise separation

---

## 🏗️ Model Architecture
- **Base Models:**
  - Convolutional Neural Networks (CNN)
  - Transfer Learning (ResNet, EfficientNet, ConvNeXt)
- **Classification Type:** Multi-label
- **Loss Function:** Binary Cross-Entropy
- **Optimization:** Adam optimizer
- **Regularization:** Dropout and data augmentation

---

## 🔍 Explainable AI (Grad-CAM)
To address the black-box nature of deep learning in healthcare:
- Grad-CAM heatmaps are generated for each prediction
- Highlight lung regions contributing to disease detection
- Helps validate model focus and improves transparency

---

## 📈 Evaluation Metrics
- Accuracy
- Precision (per disease)
- Recall (per disease)
- F1-Score
- Area Under ROC Curve (AUROC)
- Confusion Matrix (per class)

---

## 🧰 Technology Stack

| Category | Tools |
|--------|------|
| Programming Language | Python |
| Deep Learning | TensorFlow / PyTorch |
| Image Processing | OpenCV, PIL |
| Explainability | Grad-CAM |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Web App | Streamlit / Flask |
| Version Control | Git, GitHub |

---

## 📁 Project Structure
