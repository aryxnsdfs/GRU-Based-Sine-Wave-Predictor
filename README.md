# 📈 GRU-Based Sine Wave Predictor

This project demonstrates how to use a **GRU (Gated Recurrent Unit)** neural network to predict the next value in a sine wave. It uses a simple time series forecasting setup to help beginners understand how RNNs work for sequence modeling.

---

## 🔧 Features

- Predicts the next value of a sine wave using deep learning
- Built with:
  - TensorFlow / Keras
  - NumPy
  - Matplotlib
- Visual comparison between actual sine wave and predicted values

---

## 🧠 How It Works

- Generates sine wave data from:
  ```python
  x = np.arange(0, 100, 0.1)
  data = np.sin(x)
Uses a sliding window of 50 values to predict the next one:

X = data[i:i+50]
y = data[i+50]

📊 Sample Output
The output is a plot showing:

🔵 True Sine Wave (y_test)

🟠 Model Prediction (pred)
