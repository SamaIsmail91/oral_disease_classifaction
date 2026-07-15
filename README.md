# 🦷 Clinical-Grade Oral Diseases Image Classification

An end-to-end deep learning framework designed to detect and classify multi-class oral and dental pathologies from intraoral photographs. This project implements a high-performance **Custom Convolutional Neural Network (CNN)** alongside state-of-the-art **Transfer Learning models (MobileNetV2 and EfficientNet-B0)** using **TensorFlow/Keras** and **PyTorch**, complete with a user-friendly **Gradio Web Application** for clinical screening simulations.

---

## 📌 Project Overview
Oral diseases represent a major global health burden. Early and accurate screening is essential for timely intervention. This repository provides a robust, production-ready computer vision pipeline that handles clinical image variations, corrects severe data imbalance, and classifies oral pathologies into **six distinct categories**:

1. **Calculus**: Hardened plaque deposits requiring professional scaling.
2. **Caries**: Tooth decay and dental cavities requiring restorative fillings.
3. **Gingivitis**: Gum inflammation and early-stage periodontal disease.
4. **Hypodontia**: Congenitally missing teeth requiring orthodontic spatial management.
5. **Tooth Discoloration**: Extrinsic/intrinsic enamel stains or pulpal necrosis indicators.
6. **Mouth Ulcers**: Painful mucosal lesions requiring clinical evaluation if persistent.

---

## 🚀 Key Features

* **Data Reorganization & Cleanup Pipeline**: Automatically resolves the common Kaggle nested-folder trap, bypasses raw YOLO text annotation files, and structures raw image binaries.
* **Class Imbalance Mitigation**: Computes dynamic inverse-frequency loss penalties (Class Weights) inside the categorical cross-entropy loss function to prevent model bias.
* **High-Performance Data Augmentation**: Simulates clinical photography variations using translation, zoom, rotation, horizontal/vertical flips, and color/brightness jittering.
* **Comparative Model Evaluation**: Evaluates a lightweight Custom CNN against pre-trained SOTA models (MobileNetV2 and EfficientNet-B0) using stratified validation/test sets (80/10/10 split).
* **Interactive Gradio Interface**: A clean web interface showing class probability distributions and generating structured diagnostic summaries with clinical recommendations.

---

## 📂 Repository Structure

```text
├── oral-disease-classification/
│   ├── app.py                     # Gradio Deployment Web Application
│   ├── split_dataset/             # Stratified Dataset Splits (Train / Val / Test)
│   ├── figures/                   # Generated evaluation plots and class distribution charts
│   │   └── sample_images.png      # Auto-generated sample grid of clinical pathologies
│   ├── best_mobilenet_model.pth   # Saved PyTorch checkpoint for production
│   └── notebook.ipynb             # Core training and evaluation notebook
📊 Model Pipeline & ArchitectureA. Data Preprocessing & AugmentationsImages are resized to 128x128 pixels (for TensorFlow models) or 224x224 pixels (for transfer learning backbones) and normalized using ImageNet statistical means:$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$B. Custom CNN ArchitectureOur custom baseline model is built sequentially for robust feature extraction without exploding parameters:Feature Extractor: Four sequential Conv2D blocks (32 ➡️ 64 ➡️ 128 ➡️ 256 filters) featuring BatchNormalization, LeakyReLU activations, and MaxPooling2D.Regularization: Integrated Dropout (ranging from 20% to 50%) to penalize spatial co-adaptations.Classification Head: Dense feedforward layer (128 units) ending with a Softmax output layer predicting class probabilities across the 6 targets.📈 Performance & Comparative ResultsArchitectureValidation LossValidation AccuracyTest Set AccuracyKey StrengthsCustom CNN0.XXXX.X%XX.X%Lightweight, custom built, fast inferenceMobileNetV20.XXXX.X%XX.X%Highly efficient for mobile/edge deploymentEfficientNet-B00.XXXX.X%XX.X%Peak spatial feature sensitivity(Note: Replace italicized placeholders with your final model evaluation metrics from your Kaggle notebook run!)🖥️ Live Web App DemonstrationWe bundle our best-performing model into an interactive screening tool built with Gradio.
Inputs: Clinical intraoral photo (JPEG/PNG).

Outputs:

Interactive progress bars showing the top 3 predicted pathologies.

A Medical Decision Report Summary with actionable clinical pathways and a professional disclaimer.

🎓 Author & Context
Developed by Sama Ismail, a Data Science & AI Specialist at Beni Suef National University. This project demonstrates the practical application of Deep Learning, Data Engineering, and UI development in modern medical AI screening frameworks.

GitHub: github.com/SamaIsmail91

Kaggle: kaggle.com/samaismail91
