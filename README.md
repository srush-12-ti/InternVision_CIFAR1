 CIFAR-10 Image Classification using Convolutional Neural Networks (CNN)
 Overview

This project implements an Image Classification model using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.
The goal is to classify images into 10 categories such as airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

This is a complete Deep Learning project involving:
 Data loading
Preprocessing & normalization
Data augmentation
Building a CNN
Model training & validation
 Evaluation metrics
 Plotting accuracy & loss curves
 Predicting sample images

 Project Structure
CIFAR10_CNN_Classification/
│
├── data/                        # (Optional) If you download dataset manually
│
├── models/
│   └── cnn_model.h5             # Saved trained model
│
├── src/
│   ├── load_data.py             # Data loading + augmentation functions
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # Training script
│   └── evaluate.py              # Plotting + predictions
│
├── notebooks/
│   └── CIFAR10_CNN.ipynb        # Full Jupyter Notebook implementation
│
├── outputs/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── sample_predictions.png
│
└── README.md                    # Project documentation

 Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib / Seaborn

Jupyter Notebook / Google Colab

 Steps Performed
 1. Load & preprocess dataset

Normalized pixel values

Converted labels into one-hot encoding

Applied data augmentation

rotation

width/height shift

horizontal flip

 2. Built a CNN

A typical architecture:

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

 3. Model training

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Early stopping callback used to prevent overfitting

 4. Evaluation

Generated plots:

Training vs Validation Accuracy

Training vs Validation Loss

 5. Sample Predictions

Random images displayed with predicted labels.
