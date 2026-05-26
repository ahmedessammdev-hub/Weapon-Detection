# 🛡️ Weapon Detection System

Real-time **Gun & Knife detection** powered by **YOLOv8** and **Streamlit**. This system provides an intuitive web interface for AI-based weapon detection across multiple input sources — images, videos, and live camera feeds.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ **Image Detection** | Upload an image and instantly detect weapons with side-by-side comparison (original vs. annotated) |
| 🎥 **Video Detection** | Upload a video file and process it frame-by-frame with a full detection log |
| 📷 **Live Camera** | Real-time weapon detection using your webcam |
| 🚨 **Alert System** | Visual alerts triggered when a weapon (gun/knife) is detected |
| 📊 **Detection Logs** | Timestamped logs of all detected weapons during video/live sessions |
| 🎚️ **Confidence Threshold** | Adjustable confidence slider to fine-tune detection sensitivity |

---

## 🔍 Supported Classes

- 🔫 **Gun**
- 🔪 **Knife**

---

## 🛠️ Tech Stack

- **[Python 3.14+](https://www.python.org/)** — Core language
- **[YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)** — Object detection model
- **[Streamlit](https://streamlit.io/)** — Web UI framework
- **[OpenCV](https://opencv.org/)** — Image & video processing
- **[Pillow](https://python-pillow.org/)** — Image handling
- **[NumPy](https://numpy.org/)** — Array operations
- **[Pandas](https://pandas.pydata.org/)** — Detection log tables
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package manager

---

## 📁 Project Structure

```
Weapon Detection/
├── app.py              # Main Streamlit application
├── main.py             # CLI entry point
├── best.pt             # Trained YOLOv8 model weights
├── pyproject.toml      # Project metadata & dependencies
├── uv.lock             # Locked dependency versions
├── .python-version     # Python version (3.14)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** package manager
- A webcam (optional — for live detection)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/weapon-detection.git
   cd weapon-detection
   ```

2. **Install dependencies with uv**

   ```bash
   uv sync
   ```

3. **Ensure the model weights exist**

   Make sure `best.pt` (the trained YOLO model) is in the project root directory.

### Running the App

```bash
uv run streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**.

---

## 📸 Usage

### Image Detection
1. Select **"Image Upload"** from the sidebar.
2. Upload a `.jpg`, `.jpeg`, or `.png` image.
3. View the original image alongside the detection result.

### Video Detection
1. Select **"Video Upload"** from the sidebar.
2. Upload a `.mp4`, `.mov`, or `.avi` video.
3. Watch the processed video with detection overlays and review the detection log.

### Live Camera
1. Select **"Live Camera"** from the sidebar.
2. Click **"Start Camera"** to begin real-time detection.
3. Detected weapons will trigger visual alerts in real time.

### Configuration
- **Confidence Threshold** — Adjust via the sidebar slider (0.1 – 1.0, default: 0.4).
- **Alert System** — Toggle on/off from the sidebar.

---

## 🧠 Model Training

The YOLO model (`best.pt`) was trained on a custom dataset containing annotated images of guns and knives. If you want to train your own model:

1. Prepare a labeled dataset in YOLO format.
2. Train using Ultralytics:

   ```bash
   yolo detect train data=your_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```

3. Replace `best.pt` with your newly trained weights.

---

## 📄 License

This project is open source. Feel free to use and modify it for your own purposes.

---

<p align="center">
  <strong>AI Surveillance • YOLO Object Detection • Real-Time Monitoring</strong>
</p>
