# 🩻 ChestXNet++
### Automated Chest X-Ray Disease Detection using Deep Learning

ChestXNet++ is a deep learning–based medical imaging project that detects multiple thoracic diseases from chest X-ray images using the NIH Chest X-ray dataset.  
The project follows **industry best practices** in data preprocessing, model training, evaluation, and documentation.

---

## 🚀 Project Highlights

✅ Multi-label disease classification  
✅ Medical imaging best practices  
✅ ROC-AUC–based evaluation  
✅ Fast and efficient training  
✅ Kaggle & GitHub ready  
✅ Resume and interview friendly  

---

## 🧠 Problem Statement

Chest X-ray interpretation is time-consuming and requires expert radiologists.  
This project aims to assist medical professionals by building an **AI system capable of predicting multiple chest diseases from a single X-ray image**.

---

## 📂 Dataset

- **Name:** NIH Chest X-ray Dataset  
- **Images:** 112,120 frontal chest X-rays  
- **Labels:** 14 thoracic diseases  
- **Source:** National Institutes of Health (NIH)

> ⚠️ Dataset is **NOT uploaded to GitHub** due to its large size.

---

## 🦠 Diseases Classified

- Atelectasis  
- Cardiomegaly  
- Effusion  
- Infiltration  
- Mass  
- Nodule  
- Pneumonia  
- Pneumothorax  
- Consolidation  
- Edema  
- Emphysema  
- Fibrosis  
- Pleural Thickening  
- Hernia  

---

## 🧩 Problem Type

- **Multi-label classification**
- One X-ray can have multiple diseases
- Outputs disease-wise probability scores

---

## 🏗️ Model Architecture

- Backbone: **ResNet-18 (ImageNet pretrained)**
- Input size: `224 × 224`
- Input channels: `1 (grayscale)`
- Output: `14 sigmoid probability scores`
- Framework: **PyTorch**

---

## ⚙️ Training Configuration

| Parameter | Value |
|--------|-------|
| Epochs | 5 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Evaluation Metric | ROC-AUC |

---

## 📊 Model Performance

> **Primary Metric: ROC-AUC**

| Metric | Value |
|------|------|
| Best Validation ROC-AUC | **~0.78** |
| Training Accuracy | ~78% |
| Stable Convergence | ✅ |

✅ ROC-AUC is preferred over accuracy due to **class imbalance** in medical datasets.

---
Input: Chest X-ray image

Predicted Probabilities:

Cardiomegaly: 0.82

Effusion: 0.67

Pneumonia: 0.14

Hernia: 0.03
## 📁 Project Structure

ChestXNet++/
│
├── notebook/
│ └── chestxnet_training.ipynb
│
├── src/
│ ├── config.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
│
├── outputs/
│ ├── best_model.pth
│ └── metrics.txt
│
├── README.md
└── requirements.txt


---

## 📦 Dataset Handling

Due to size limitations:
- Dataset is stored using **Kaggle Datasets**
- Image paths are dynamically mapped at runtime
- CSV file is read during training

---

## 🛠️ Tools & Technologies

- Python
- PyTorch
- Pandas & NumPy
- OpenCV
- Scikit-learn
- Kaggle GPU

---

## ⚠️ Limitations

- Labels may contain noise
- Rare diseases have fewer samples
- No clinical history is used

---

## 🚀 Future Improvements

- EfficientNet / DenseNet
- Class-weighted loss
- Grad-CAM visualization
- Threshold tuning per disease

---

## 🧑‍⚕️ Medical Disclaimer

This project is **for educational and research purposes only**.  
It **must not be used for real-world medical diagnosis**.

---

## 👨‍💻 Author

**Abdul Muzamil**  
Deep Learning | Computer Vision | Medical AI  

📌 Kaggle Notebook: *https://www.kaggle.com/code/shaikmuzamil18/multi-disease-chest-x-ray-classifier*  
📌 GitHub Repository: *https://github.com/Muzamil2004/Multi-disease-chest-xray-classifier-using-pytorch-and-CNNs*  
📌 LinkedIn: *(add link)*  

---

⭐ If you find this project useful, consider giving it a star!


