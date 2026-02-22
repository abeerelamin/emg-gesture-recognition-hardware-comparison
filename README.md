# 🧠 EMG Hand Gesture Recognition  
## Performance Benchmarking on CPU vs GPU (Deep Learning Study)

> A comprehensive performance comparison of deep learning architectures for EMG-based hand gesture recognition, including hardware benchmarking on CPU and GPU environments.

---

## 📌 Project Overview

This project investigates the effectiveness of various deep learning models for classifying EMG (Electromyography) hand gestures and benchmarks their computational performance across different hardware platforms.

We implemented and evaluated the following architectures:

- RNN
- LSTM
- Bi-LSTM
- GRU
- Bi-GRU
- CNN

The primary objectives were:

- 📊 Compare model classification performance
- ⚡ Benchmark CPU vs GPU training time
- 🧠 Analyze temporal modeling capabilities
- 🔬 Study hardware–performance trade-offs

---

## 📂 Dataset

We used a public EMG dataset recorded using the **Myo Armband**, which provides:

- 8 surface EMG channels
- 1 ms sampling rate
- Multiple hand gesture classes

For this study, we focused on 6 rehabilitation-relevant gestures:

1. Radial deviation  
2. Wrist flexion  
3. Ulnar deviation  
4. Wrist extension  
5. Hand close  
6. Hand open  

### 🔧 Preprocessing Pipeline

- 200 ms sliding windows (50% overlap)
- Pure-label window filtering
- Z-score normalization (train-only fit)
- 80/20 stratified split
- Final dataset: **13,697 windows**

Input shape:

(samples, 200 timesteps, 8 channels)


---

## 🏗 Model Architectures

We evaluated:

- Vanilla RNN (baseline)
- LSTM
- Bidirectional LSTM
- GRU
- Bidirectional GRU
- 1D Convolutional Neural Network (CNN)

All models were trained using identical preprocessing, splits, and evaluation metrics to ensure fair comparison.

---

## 📊 Results

### 🔹 Classification Performance

| Model     | Accuracy |
|------------|----------|
| RNN        | 74.41%   |
| LSTM       | 93.57%   |
| Bi-LSTM    | 93.24%   |
| GRU        | 92.73%   |
| Bi-GRU     | 92.29%   |
| **CNN**    | ⭐ **95.0%** |

**Key Insight:**  
CNN achieved the highest accuracy, demonstrating strong spatial feature extraction across EMG channels.

Recurrent models significantly outperformed vanilla RNN due to their ability to model long-term temporal dependencies.

---

## ⚡ Hardware Benchmarking

### Environment

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon @ 2.00 GHz |
| GPU | NVIDIA Tesla T4 (16GB GDDR6) |
| GPU Bandwidth | 320 GB/s |
| Peak Performance | 8.1 TFLOPS |

### ⏱ Training Time Comparison (LSTM Example)

| Hardware | Training Time |
|----------|---------------|
| CPU      | 232.6s |
| GPU      | 116.3s |

GPU reduced training time by approximately **50%**.

---

## 📈 Analysis & Discussion

- RNN struggled due to vanishing gradient limitations.
- LSTM and GRU variants captured long-term temporal dependencies effectively.
- CNN excelled due to spatial pattern extraction across EMG channels.
- Most classification confusion occurred between physiologically similar gestures.

### Hardware Insight

GPU acceleration dramatically reduced training time due to:

- High memory bandwidth
- Massive parallel matrix computation
- Optimized CUDA kernels

Trade-off observed: Increased energy consumption for faster execution.

---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab (Linux environment)


---

## 🔬 Future Work

- TPU benchmarking
- Hybrid CNN-LSTM architecture
- Feature engineering (frequency-domain EMG features)
- Cross-subject generalization analysis

---


## ⭐ If You Found This Interesting

Give the repo a star ⭐ and feel free to fork or contribute!
