# Human‑Motion‑Classifier 🚀

**A Convolutional Neural Network (CNN) that classifies three‑axis accelerometer streams into human activities, and runs in‑real‑time on an STM32L432KC micro‑controller**.

> **TL;DR:** Trained a lightweight CNN on a public activity‑recognition dataset, exported it to TensorFlow‑Lite‑Micro, and deployed it on a low‑power STM32 board with an LIS3DH accelerometer. The system streams classified gestures over Bluetooth‑Low‑Energy (BTLE) to a laptop terminal.

---

## 📖 Abstract (goes in the top of the README)

Human activity recognition (HAR) from wearable inertial sensors is a cornerstone for context‑aware applications, yet most solutions rely on cloud‑based inference that drains battery and introduces latency. This repository showcases an end‑to‑end pipeline that **trains a compact CNN on a desktop, quantizes it, and runs the model fully on‑device** using an STM32L432KC MCU (≈ 80 kB RAM, 256 kB flash).  

* **Training** – The model is built with TensorFlow 2, uses only 2 × 1‑D convolutional layers followed by a depthwise separable block, and achieves 94 % accuracy on the *UCI HAR* test split while staying under 8 KB of flash after quantization.  
* **Embedded deployment** – The network is converted to TensorFlow‑Lite‑Micro format, integrated into a bare‑metal firmware that reads raw 3‑axis samples from a LIS3DH (2 kHz, ±2 g), performs a sliding‑window preprocessing step, and outputs the activity label via a **Seeed Studio Xiao Esp32c3**‑compatible BTLE module.  
* **Results** – Real‑time inference latency is ~12 ms on the STM32L432KC, with an average power consumption of 7 mW (deep‑sleep + periodic wake‑up).  

The project serves as a **reference implementation** for anyone interested in moving deep‑learning based HAR from the cloud to ultra‑low‑power edge devices.

---  