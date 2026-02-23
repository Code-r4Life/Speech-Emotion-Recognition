# 🎙️ Speech Emotion Recognition using Deep Learning
### **Author: Shinjan Saha**

This project implements a **Speech Emotion Recognition (SER)** system that detects human emotions from voice recordings using advanced audio signal processing and a deep CNN model.

The system analyzes speech patterns and classifies emotions into multiple categories, enabling emotion-aware AI applications.

---

# 📌 1. Project Overview

This repository provides a complete pipeline for:

- 🎧 Audio preprocessing & augmentation  
- 🔊 Feature extraction from speech signals  
- 🧠 Deep learning model training  
- 📊 Evaluation & visualization  
- 🔮 Real-time emotion prediction  

The model classifies speech into **7 emotions**:

😡 Angry  
🤢 Disgust  
😨 Fear  
😊 Happy  
😐 Neutral  
😢 Sad  
😲 Surprise  

---

# 🧠 2. Dataset Used

## 🎧 RAVDESS Emotional Speech Dataset
- Professional emotional speech recordings
- Multiple actors & emotional intensities

## 🎧 TESS Emotional Speech Dataset
- Clear emotional speech recordings
- High-quality pronunciation & tone variations

---

# ⚙️ 3. Tech Stack

### **Languages & Libraries**

- Python  
- TensorFlow / Keras  
- Librosa (audio processing)  
- NumPy & Pandas  
- Scikit-learn  
- Matplotlib & Seaborn  

---

# 🔊 4. Audio Processing Pipeline

## 🔹 Step 1 — Preprocessing
- Load audio files
- Trim silence & normalize signals
- Standardize sampling rate

## 🔹 Step 2 — Data Augmentation
To improve robustness and generalization:

✔ Noise Injection  
✔ Time Stretching  
✔ Pitch Shifting  
✔ Signal Shifting  

## 🔹 Step 3 — Feature Extraction

Extracted features:

- Zero Crossing Rate (ZCR)
- Chroma STFT
- MFCC (Mel Frequency Cepstral Coefficients)
- Root Mean Square Energy (RMS)
- Mel Spectrogram

---

# 🏗️ 5. Model Architecture

A deep **1D Convolutional Neural Network** designed for sequential audio features.

📷 Model Summary  

[__results___files\cnn_arch.png]


### 🔹 Architecture Details

- Multiple Conv1D feature extraction blocks  
- Batch Normalization for stability  
- MaxPooling for dimensionality reduction  
- Dropout layers to prevent overfitting  
- Dense layers for classification  
- Softmax output layer (7 emotions)

**Total Parameters:** 2.67M  
**Trainable Parameters:** 2.66M  

---

# 📊 6. Training Strategy

- EarlyStopping to prevent overfitting  
- ReduceLROnPlateau for adaptive learning  
- ModelCheckpoint for best model saving  
- StandardScaler for feature normalization  

---

# 📈 7. Model Performance

## 🔹 Classification Report

[__results___files\classification_report.png]


**Overall Accuracy:** **92%**  
**Macro Avg F1 Score:** **0.92**

### Key Observations:
- Neutral emotion has highest recall (0.98)
- Strong precision across fear & angry classes
- Balanced performance across all emotions

---

## 🔹 Training & Validation Performance

[__results___files\__results___45_1.png]


**Insights:**
- Smooth convergence
- Minimal overfitting
- Stable validation accuracy (~91–92%)

---

## 🔹 Confusion Matrix

[__results___files\__results___48_0.png]


**Insights:**
- Strong diagonal dominance → accurate predictions  
- Minor confusion between happy & neutral  
- Fear & angry classification highly reliable  

---

# 🧪 8. Results Summary

✔ Test Accuracy: **92%**  
✔ Balanced performance across classes  
✔ Robust predictions with augmented audio  
✔ Generalizes well across speakers  

---

# 💾 9. Model Export & Inference

The saved pipeline includes:

- trained CNN model  
- scaler  
- label encoder  
- feature extractor  

Saved as: `emotion_preprocessing.joblib`


## 🔮 Predict Emotion from New Audio

```python
prediction = get_predictions("audio.wav", emotion_preprocess)
print(prediction)
```

---

# 📁 10. Project Structure

```
Speech-Emotion-Recognition/
│
├── model
│ └──best_emotion_model.keras
├── metrics
│ ├── __results___45_1.png
│ ├── __results___48_0.png
│ ├── classification_report.png
│ └──cnn_arch.png
├── __results___files
├── emotion_path.csv
├── features.csv
├── speech-emotion-recognition.ipynb
├── emotion_preprocessing.joblib
├── requirements.txt
└── README.md
```

---

# 🛠️ 11. Installation

```
git clone https://github.com/Code-r4Life/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
pip install -r requirements.txt
```

---

# 🌍 12. Real-World Applications

- 🎧 Voice assistants & conversational AI
- 🧠 Mental health & stress monitoring
- 📞 Customer sentiment analysis
- 🎮 Emotion-aware gaming
- 📚 Smart education platforms

---

# 📬 Interested in a Similar Project?

I build smart, ML-integrated applications and responsive web platforms. Let’s build something powerful together!

📧 shinjansaha00@gmail.com

🔗 [LinkedIn Profile](https://www.linkedin.com/in/shinjan-saha-1bb744319/)