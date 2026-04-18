# 🫁 Pneumonia Detection using Vision Transformer + Grad-CAM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 🔍 Overview
This project detects pneumonia from chest X-ray images using a fine-tuned Vision Transformer (ViT) model with Grad-CAM explainability. The model highlights which regions of the X-ray influenced its decision — critical for medical AI where doctors need to understand why a prediction was made, not just what it predicted.

## 🎯 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/mou11/pneumonia-detector)

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 87% |
| AUC-ROC | 0.98 |
| Pneumonia Recall | 100% |
| Normal Precision | 99% |

## 🏗️ Architecture
- **Base Model:** google/vit-base-patch16-224 (pretrained on ImageNet)
- **Fine-tuned on:** Chest X-Ray Images (Pneumonia) — 5,216 training images
- **Explainability:** Grad-CAM with reshape transform for ViT patch attention
- **Classes:** NORMAL vs PNEUMONIA

## 📁 Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
- Train: 5,216 images
- Test: 624 images
- Val: 16 images

## ⚠️ Class Imbalance Handling
The dataset has 3x more PNEUMONIA than NORMAL images. We handled this using weighted cross entropy loss (NORMAL weight: 3.0, PNEUMONIA weight: 1.0).

## 🧠 Why Recall Matters in Medical AI
In medical screening, a false negative (predicting NORMAL for a sick patient) is far more dangerous than a false positive. Our model achieves 100% recall for pneumonia — it never misses a sick patient.

## 🛠️ Tech Stack
- PyTorch
- HuggingFace Transformers
- pytorch-grad-cam
- Gradio
- Google Colab (T4 GPU)

## 🚀 How to Run
1. Open `pneumonia_detector.ipynb` in Google Colab
2. Connect to T4 GPU runtime
3. Run all cells in order

## 📌 Project Status
✅ Model trained and evaluated  
✅ Grad-CAM explainability implemented  
✅ Gradio demo deployed on Hugging Face Spaces
