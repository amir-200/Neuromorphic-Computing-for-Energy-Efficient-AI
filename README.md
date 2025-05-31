# 🧠 Neuromorphic Computing for Energy-Efficient AI

A mini-project comparing the performance of **Spiking Neural Networks (SNNs)** and **Feedforward Neural Networks (FFNNs)** on energy-constrained devices like the Raspberry Pi.

> **Objective**: Evaluate and demonstrate how biologically inspired Spiking Neural Networks can reduce power consumption while maintaining comparable accuracy to conventional neural networks.

---

## 📌 Project Overview

This project explores the potential of SNNs as a low-power alternative to FFNNs in classification tasks. It implements and deploys both models on a **Raspberry Pi 4**, measuring key metrics like:

- **Accuracy**
- **Inference Time**
- **Energy Consumption**

🔋 **Key Result**: SNNs showed ~40% lower energy consumption per inference while maintaining nearly the same accuracy as FFNNs.

---

## 🚀 Features

- 📊 Comparison of SNN vs FFNN on MNIST dataset
- ⚡ Real-time energy and latency measurements on Raspberry Pi
- 🔁 Poisson spike encoding for input processing
- 🔬 Custom loss function for spike count learning
- 📈 Performance evaluation with accuracy, time, and energy benchmarks

---

## ⚙️ Requirements

- **Hardware**:
  - Raspberry Pi 4 (2GB+ recommended)
  - USB multimeter or INA219 energy meter
  - Power supply + HDMI display (for initial setup)

- **Software**:
  - Python 3.x
  - PyTorch
  - snntorch
  - torchvision
  - matplotlib
  - numpy


🧪 How to Run

Clone the repo:
git clone https://github.com/<your-org>/Neuromorphic-AI.git
cd Neuromorphic-AI

Train FFNN:
python train/train_ffnn.py

Train SNN:
python train/train_snn.py

Compare Results:
Logs are saved in results/logs.csv or printed directly to console.

🌱 Future Scope
Support for Intel Loihi / SpiNNaker neuromorphic hardware

Datasets beyond MNIST (e.g., DVS Gesture, N-MNIST)

Hybrid architectures (SNN + FFNN)

STDP-based learning and unsupervised SNN training


👨‍👩‍👧‍👦 Team Members
This project was developed by the students of Atharva College of Engineering, University of Mumbai (2024-25):

Mayuri Sonawane

Amir Shaikh

Divye Sharma

Arsh Sakaria 

🙌 Acknowledgments
Special thanks to the Neuromorphic AI community and contributors to snntorch.

Inspired by research from W. Maass, Intel Loihi, IBM TrueNorth, and others.
