 
# 🌊 **Guzon SeaShore Rocks Classification — Final Project**

### 🪨 *Image Classification using Convolutional Neural Networks (CNN)*

![Status](https://img.shields.io/badge/Status-Active-blue)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Jupyter-lightgrey)

---

## 📑 **Table of Contents**

* [Project Overview](#-project-overview)
* [Technology Stack](#-technology-stack)
* [Dataset Documentation](#-dataset-documentation)
* [CNN Architecture](#-cnn-architecture)
* [Performance Metrics](#-performance-metrics)
* [Development Process](#-development-process)
* [Usage Examples](#-usage-examples)
* [Results & Analysis](#-results--analysis)
* [Future Improvements](#-future-improvements)
* [Author](#-author)

---

## 🎯 **Project Overview**

This Machine Learning project focuses on classifying various **seashore rock types** using images and a custom-built **Convolutional Neural Network (CNN)**.

### **Objectives**

* 📌 Build an efficient image classification model
* 📌 Document dataset and rock types
* 📌 Evaluate accuracy using scientific ML metrics
* 📌 Provide training and prediction scripts
* 📌 Present results, errors, and insights behind model performance

---

## 🛠️ **Technology Stack**

This project uses:

| Category             | Tools / Libraries               |
| -------------------- | ------------------------------- |
| **Core Language**    | Python 3.10+                    |
| **Deep Learning**    | TensorFlow, Keras               |
| **Data Handling**    | NumPy, Pandas                   |
| **Visualization**    | Matplotlib, Seaborn             |
| **Image Processing** | OpenCV, PIL                     |
| **Environment**      | Google Colab / Jupyter Notebook |

---

## 🗂️ **Dataset Documentation**

A custom rock dataset containing labeled images gathered from seashore environments.

### **Rock Classes**

* 🟫 *Basalt*
* 🪨 *Granite*
* 🧱 *Sandstone*
* 🐚 *Limestone*
* 🟪 *Shale*
* 🪸 *Coral Rock*
* 🔹 *Pebbles*
* 🌊 *Coastal Sediments / Mixed Rocks*

Each folder contains:

* Training images
* Validation images
* Testing images

---

## 🧠 **CNN Architecture Diagram**

```
Input Layer (150x150 RGB)
│
├── Conv2D (32 filters) + ReLU
├── MaxPooling2D
│
├── Conv2D (64 filters) + ReLU
├── MaxPooling2D
│
├── Conv2D (128 filters) + ReLU
├── MaxPooling2D
│
├── Flatten Layer
├── Dense (128 units) + ReLU
├── Dropout (0.5)
│
└── Dense (Output: Softmax for Rock Classes)
```

---

## 📊 **Performance Metrics**

| Metric                      | Score |
| --------------------------- | ----- |
| **Training Accuracy**       | `XX%` |
| **Validation Accuracy**     | `XX%` |
| **Testing Accuracy**        | `XX%` |
| **Loss**                    | `XX`  |
| **Precision / Recall / F1** | `XX%` |

> Replace `XX` with your actual results.

---

## 🚧 **Development Process Breakdown**

### **1. Dataset Setup**

* Collected raw rock images
* Cleaned, resized, normalized (150×150)
* Augmented for higher accuracy

### **2. Model Building**

* Designed CNN with 3 convolution blocks
* Applied dropout to prevent overfitting

### **3. Model Training**

* Used 20–30 epochs
* Optimizer: Adam
* Loss: Categorical Crossentropy

### **4. Evaluation**

* Generated accuracy/loss curves
* Confusion matrix to examine misclassifications

### **5. Deployment**

* Included scripts for predictions
* Ready for integration to web or mobile apps

---

## 💻 **Training & Prediction Usage**

### **🔹 Train the Model**

```bash
python train.py
```

Or inside a notebook:

```python
model.fit(train_data, epochs=20, validation_data=val_data)
```

---

### **🔹 Predict Using an Image**

```bash
python predict.py --image path/to/image.jpg
```

Notebook version:

```python
img = load_and_preprocess("sample.jpg")
prediction = model.predict(img)
print(prediction)
```

---

## 📈 **Results & Analysis**

* The CNN successfully learned textural differences across rock types
* Accuracy curve shows steady learning with no major overfitting
* Confusion matrix reveals which rocks are commonly mixed
* Visual prediction tests show **consistent classification**

  

## 🚀 **Future Improvements Roadmap**

* ⬆️ Expand dataset for better generalization
* ⚡ Apply Transfer Learning (ResNet50/MobileNet)
* 📱 Build mobile app classifier (TensorFlow Lite)
* 🌐 Deploy online demo using Flask or Streamlit
* 🔍 Use Grad-CAM for model explainability
* 🎯 Add more local rock categories from PH coastal regions

## 👤 **Author**

**Johnny Guzon**
*BSIT — Machine Learning Final Project*

 
