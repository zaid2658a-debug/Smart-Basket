# 🛒 Smart Basket

**An AI-powered smart shopping basket that automatically detects, counts, and prices grocery items in real time** — combining computer vision, embedded hardware, cloud sync, and a mobile app into one seamless shopping experience.

Built as a graduation/university project, Smart Basket demonstrates a full end-to-end IoT + AI product pipeline: from a custom-trained object detection model, to on-device inference on a Raspberry Pi, to a live Firebase-synced Flutter app that shows customers their basket total as they shop.

---
# 🛒 Cartly — Smart Basket

**An AI-powered smart shopping basket that automatically detects, counts, and prices grocery items in real time** — combining computer vision, embedded hardware, and cloud sync into one seamless shopping experience.

Cartly is a graduation/university project that demonstrates a full end-to-end IoT + AI product pipeline: a custom-trained object detection model running on-device on a Raspberry Pi, syncing live to a companion mobile app so customers can see their basket total as they shop.

> 📌 **This repository covers the AI / Computer Vision module of Cartly** — dataset creation, model training, and edge deployment (Raspberry Pi + laptop inference). The mobile app (Flutter) is a separate module built by a teammate and is not included here.

---

## 📌 How It Works

1. A camera mounted on the physical basket continuously captures frames.
2. A custom-trained **YOLO26n** object detection model identifies grocery items in real time.
3. Detections are stabilized and aggregated into item counts (avoiding false triggers from a single noisy frame).
4. The basket's current contents and total price are pushed live to **Firebase Realtime Database**.
5. A companion **Flutter mobile app** (separate module, built by a teammate) listens to Firebase and updates the customer's basket view instantly — no scanning, no waiting in line.

---

## ✨ Key Features

- 🎯 **Real-time object detection** with a custom YOLO26n model trained on a self-collected grocery dataset
- ⚡ **Edge-optimized inference** — exported to both ONNX (laptop/PC) and NCNN (Raspberry Pi 5, ARM-native, no NMS needed)
- 🔄 **Live cloud sync** via Firebase Realtime Database, updating only when the basket state actually changes
- 🧠 **Stable detection logic** — a confirm/hold frame tracker filters out flicker and prevents duplicate counting
- 📱 **Companion Flutter app** showing live basket contents and running total
- 🔌 **Hardware-deployed** — running live on a Raspberry Pi 5 with a Pi Camera v2, built into a physical basket prototype

---

## 🧠 The AI Model

| | |
|---|---|
| **Architecture** | YOLO26n (nano) — 2.37M parameters, 5.2 GFLOPs |
| **Framework** | Ultralytics + PyTorch, trained on Kaggle (Tesla T4 GPU) |
| **Dataset** | 868 self-annotated images, 4 grocery product classes, labeled and exported via Roboflow |
| **Classes** | Doritos (Sweet Chili), Indomie (Chicken Curry), Pepsi (Diet), Tea El-Arosa |
| **Training** | 100 epochs, image size 640, MuSGD optimizer, early stopping (patience 20) |
| **Input size (deployed)** | 320×320 for fast edge inference |

### 📊 Validation Results

| Metric | Score |
|---|---|
| Precision | 0.960 |
| Recall | 0.946 |
| **mAP@50** | **0.969** |
| **mAP@50-95** | **0.848** |

The model was exported into two optimized formats for two different deployment targets:
- **`best.onnx` / `best_int8.onnx`** — for laptop/PC inference (CPU or GPU) via ONNX Runtime
- **`model.ncnn.bin` / `model.ncnn.param`** — for Raspberry Pi 5 inference via NCNN (Tencent's ARM-native inference engine), chosen for its low latency and NMS-free architecture

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLO26 (Ultralytics), trained via Kaggle notebooks |
| Dataset Management | Roboflow |
| Edge Inference | ONNX Runtime, NCNN |
| Embedded Hardware | Raspberry Pi 5 + Pi Camera v2 (`picamera2` / `libcamera`) |
| Backend / Sync | Firebase Realtime Database |
| Mobile App | Flutter |
| Core Libraries | OpenCV, NumPy |

---

## 📂 Project Structure

```
Smart-Basket/
├── AI/
│   ├── final_notebook.ipynb     # Full training pipeline: data prep → train → validate → export
│   ├── inference_onnx.py        # Inference + Firebase sync (laptop/PC, webcam or video)
│   ├── version_pi.py            # Inference + Firebase sync (Raspberry Pi 5 + Pi Camera v2)
│   ├── best.onnx / best_int8.onnx   # Exported ONNX model (FP32 / quantized INT8)
│   ├── laptop_package.zip       # Ready-to-deploy package for PC inference
│   ├── rpi5_package.zip         # Ready-to-deploy package for Raspberry Pi (NCNN)
│   ├── classes.txt              # Class label list
│   └── data.yaml                # Dataset configuration (Roboflow export)
├── LICENSE
└── README.md
```

---

## 🚀 Running Inference Locally

```bash
# Install dependencies
pip install opencv-python numpy onnxruntime firebase-admin

# Run on a webcam (no cloud sync)
python AI/inference_onnx.py --model AI/best.onnx --source 0

# Run on an image
python AI/inference_onnx.py --model AI/best.onnx --source image.jpg --output result.jpg

# Run with Firebase sync enabled
python AI/inference_onnx.py --model AI/best.onnx --source 0 \
    --firebase --service-account serviceAccountKey.json \
    --database-url https://YOUR-PROJECT.firebaseio.com
```

> On Raspberry Pi 5, use `AI/version_pi.py` with the NCNN model package (`rpi5_package.zip`) instead.

---

## 🎥 Demo

*(Add a photo of the physical basket prototype, a screenshot of the Flutter app, and a demo video/GIF of the detection running here.)*

---

## 👥 Team & Context

Cartly was built as a graduation project combining computer vision, embedded systems, and mobile development into a working hardware prototype.

| Module | Description | Owner |
|---|---|---|
| 🧠 AI / Computer Vision (this repo) | Dataset collection & annotation, YOLO26n training, edge optimization (ONNX/NCNN), Raspberry Pi deployment, Firebase integration | **Ziad Abdullah** |
| 📱 Mobile App | Flutter app consuming the live Firebase basket data | Teammate |
| 🔌 Hardware Assembly | Physical basket build, Raspberry Pi 5 + Pi Camera v2 setup | Team |

## 📄 License

See [LICENSE](./LICENSE) for details.
## 📌 How It Works

1. A camera mounted on the physical basket continuously captures frames.
2. A custom-trained **YOLO26n** object detection model identifies grocery items in real time.
3. Detections are stabilized and aggregated into item counts (avoiding false triggers from a single noisy frame).
4. The basket's current contents and total price are pushed live to **Firebase Realtime Database**.
5. The **Flutter mobile app** listens to Firebase and updates the customer's basket view instantly — no scanning, no waiting in line.

---

## ✨ Key Features

- 🎯 **Real-time object detection** with a custom YOLO26n model trained on a self-collected grocery dataset
- ⚡ **Edge-optimized inference** — exported to both ONNX (laptop/PC) and NCNN (Raspberry Pi 5, ARM-native, no NMS needed)
- 🔄 **Live cloud sync** via Firebase Realtime Database, updating only when the basket state actually changes
- 🧠 **Stable detection logic** — a confirm/hold frame tracker filters out flicker and prevents duplicate counting
- 📱 **Companion Flutter app** showing live basket contents and running total
- 🔌 **Hardware-deployed** — running live on a Raspberry Pi 5 with a Pi Camera v2, built into a physical basket prototype

---

## 🧠 The AI Model

| | |
|---|---|
| **Architecture** | YOLO26n (nano) — 2.37M parameters, 5.2 GFLOPs |
| **Framework** | Ultralytics + PyTorch, trained on Kaggle (Tesla T4 GPU) |
| **Dataset** | 868 self-annotated images, 4 grocery product classes, labeled and exported via Roboflow |
| **Classes** | Doritos (Sweet Chili), Indomie (Chicken Curry), Pepsi (Diet), Tea El-Arosa |
| **Training** | 100 epochs, image size 640, MuSGD optimizer, early stopping (patience 20) |
| **Input size (deployed)** | 320×320 for fast edge inference |

### 📊 Validation Results

| Metric | Score |
|---|---|
| Precision | 0.960 |
| Recall | 0.946 |
| **mAP@50** | **0.969** |
| **mAP@50-95** | **0.848** |

The model was exported into two optimized formats for two different deployment targets:
- **`best.onnx` / `best_int8.onnx`** — for laptop/PC inference (CPU or GPU) via ONNX Runtime
- **`model.ncnn.bin` / `model.ncnn.param`** — for Raspberry Pi 5 inference via NCNN (Tencent's ARM-native inference engine), chosen for its low latency and NMS-free architecture

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLO26 (Ultralytics), trained via Kaggle notebooks |
| Dataset Management | Roboflow |
| Edge Inference | ONNX Runtime, NCNN |
| Embedded Hardware | Raspberry Pi 5 + Pi Camera v2 (`picamera2` / `libcamera`) |
| Backend / Sync | Firebase Realtime Database |
| Mobile App | Flutter |
| Core Libraries | OpenCV, NumPy |

---

## 📂 Project Structure

```
Smart-Basket/
├── AI/
│   ├── final_notebook.ipynb     # Full training pipeline: data prep → train → validate → export
│   ├── inference_onnx.py        # Inference + Firebase sync (laptop/PC, webcam or video)
│   ├── version_pi.py            # Inference + Firebase sync (Raspberry Pi 5 + Pi Camera v2)
│   ├── best.onnx / best_int8.onnx   # Exported ONNX model (FP32 / quantized INT8)
│   ├── laptop_package.zip       # Ready-to-deploy package for PC inference
│   ├── rpi5_package.zip         # Ready-to-deploy package for Raspberry Pi (NCNN)
│   ├── classes.txt              # Class label list
│   └── data.yaml                # Dataset configuration (Roboflow export)
├── LICENSE
└── README.md
```

---

## 🚀 Running Inference Locally

```bash
# Install dependencies
pip install opencv-python numpy onnxruntime firebase-admin

# Run on a webcam (no cloud sync)
python AI/inference_onnx.py --model AI/best.onnx --source 0

# Run on an image
python AI/inference_onnx.py --model AI/best.onnx --source image.jpg --output result.jpg

# Run with Firebase sync enabled
python AI/inference_onnx.py --model AI/best.onnx --source 0 \
    --firebase --service-account serviceAccountKey.json \
    --database-url https://YOUR-PROJECT.firebaseio.com
```

> On Raspberry Pi 5, use `AI/version_pi.py` with the NCNN model package (`rpi5_package.zip`) instead.

---

## 🎥 Demo

*(Add a photo of the physical basket prototype, a screenshot of the Flutter app, and a demo video/GIF of the detection running here.)*

---

## 👥 Team & Context

Built as a graduation project combining computer vision, embedded systems, and mobile development into a working hardware prototype.

## 📄 License

See [LICENSE](./LICENSE) for details.
