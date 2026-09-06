# Face Matching Mini Tool

A desktop face verification application that compares two facial images to determine if they belong to the same person.

---

## Features

- **Side-by-Side Preview:** Clean interface with drag-and-drop image support and reset functionality.
- **Responsive UI:** Background worker (`QThread`) keeps the interface smooth with live status updates.
- **Accurate Model Pipeline:** Combines **RetinaFace** for face detection and **ArcFace** for identity verification.
- **Optimized Performance:** Detected face crops are forwarded directly to matching, eliminating redundant detection passes.
- **In-Memory Caching:** SHA-256 content hashing allows instant sub-millisecond results for repeated comparisons.
- **Robust File Handling:** Safe image decoding supporting Arabic and Unicode file paths on Windows.
- **Clear Biometric Decision:** Displays **Match** or **No Match**, accompanied by similarity score, distance, and threshold.

---

## Prerequisites

- **Python 3.11** (TensorFlow dependency does not support Python 3.12+ / 3.14).
- Internet connection on first launch to cache model weights locally.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
python main.py
```

---

## Running Tests

```bash
python -m unittest tests
```

---

## Project Structure

```text
face_matching_tool/
├── main.py          # Application entry point
├── main_window.py   # PySide6 desktop interface and worker thread
├── comparator.py    # ArcFace matching and in-memory cache
├── face_detector.py # RetinaFace detection and validation
├── validator.py     # Image decoding and Unicode safety
├── tests.py         # Automated unit test suite
└── requirements.txt # Project dependencies
```
