# 🌿 Plant Disease Classification System (Streamlit + Deep Learning)

A modern, interactive **Streamlit web application** that classifies plant diseases using **three deep learning models**.  
The system provides real-time predictions, model comparison, and dataset analysis in a clean and user-friendly interface.

---

## 📌 Overview

This project is designed to help users identify plant diseases from images using AI.  
It leverages multiple trained deep learning models to improve reliability and allow comparison between architectures.

The application is structured into multiple pages for better usability and organization.

---

## 🚀 Key Features

### 🔍 Multi-Model Prediction
- Uses **3 trained deep learning models**
- Predicts plant disease from a single image
- Displays:
  - Predicted class
  - Confidence score

### 🧠 Model Comparison
- Compare predictions from all models side-by-side
- Highlights differences between models
- Helps evaluate model reliability

### 📊 Dataset Analysis
- Upload multiple images or dataset folder/zip
- Displays dataset statistics
- Shows sample images
- Runs batch predictions
- Visualizes prediction distribution

### 🖼️ Flexible Input
- Upload image file
- Paste image URL
- Preview image before prediction

### ⚡ Performance Optimized
- Cached model loading
- Fast inference pipeline
- Consistent preprocessing across all models

### ❌ Robust Error Handling
- Invalid image detection
- URL validation
- File format checking

---

## 🏗️ Application Structure

The app is organized into sidebar navigation pages:
🏠 Home
🔍 Try Model
📊 Compare Models
📁 Dataset Analysis
📚 About Models
ℹ️ About Project


---

## 📄 Pages Description

### 🏠 Home
- Application introduction
- System overview
- Quick instructions for users
- Simple and clean landing page

---

### 🔍 Try Model
- Upload image or provide image URL
- Display image preview
- Run prediction using all 3 models
- Show results:
  - Predicted class
  - Confidence score (%)
- Clean multi-column layout for comparison

---

### 📊 Compare Models
- Compare outputs of all models side-by-side
- Display:
  - Prediction per model
  - Confidence score
- Highlight disagreements between models
- Simple comparison summary

---

### 📁 Dataset Analysis
- Upload dataset (multiple images / zip supported)
- Features:
  - Dataset size statistics
  - Sample image preview
  - Batch prediction using all models
- Visualization:
  - Prediction distribution chart
  - Summary insights

---

### 📚 About Models
Detailed explanation of each model:

#### 🌱 CNN Model
- Basic Convolutional Neural Network
- Fast inference
- Lower complexity

#### 🌿 ResNet Model
- Deep Residual Network architecture
- High accuracy
- Strong feature extraction

#### 🍃 MobileNet Model
- Lightweight architecture
- Optimized for speed
- Suitable for real-time applications

---

## ⚙️ Installation Guide

1. Clone Repository
```bash
git clone https://github.com/pola-moukhtar/project.git
cd plant-disease-app


2. Install Dependencies
```bash
pip install -r requirements.txt


3. Run Application
```bash
streamlit run app.py