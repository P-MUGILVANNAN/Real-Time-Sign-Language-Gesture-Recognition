# SignVision AI — Troubleshooting & Common Issues Guide

## 1. Webcam Access & Permission Issues

### Issue: Camera initialization returns black frame or "Unable to access camera index"
**Causes & Solutions**:
- **Camera In-Use**: Check if another application (Teams, Zoom, Discord, Windows Camera) is currently accessing the webcam. Close background camera applications.
- **DirectShow Driver on Windows**: SignVision AI attempts to initialize OpenCV with `cv2.CAP_DSHOW` by default on Windows:
  ```python
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  ```
  If index 0 fails, test index 1 or 2 using the hardware index selector in the sidebar.
- **Browser Snapshot Fallback**: If desktop OpenCV camera access is restricted in your environment, switch to **"Browser Snapshot Mode"** in the Live Recognition page to provide camera input directly through the browser.

---

## 2. MediaPipe Model Loading

### Issue: `models/hand_landmarker.task` not found
**Solution**:
SignVision AI automatically attempts to download the official MediaPipe HandLandmarker task bundle on initial startup. If you are behind an enterprise firewall or offline:
1. Download `hand_landmarker.task` manually:
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
2. Place the file at `models/hand_landmarker.task`.

---

## 3. High Latency or Low FPS

### Tips for Optimizing Real-Time Performance:
1. **Reduce Resolution**: If your webcam defaults to 1080p, adjust OpenCV to capture at 640×480:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
2. **Disable Visual Skeleton Overlay**: Toggling off "Draw Hand Landmarks Skeleton" in the sidebar bypasses OpenCV rendering pipelines, saving 3–5 ms per frame.
3. **Single Hand Mode**: Ensure `mode: "single_hand"` is selected in `config/config.yaml` to extract 63 features rather than dual-hand 126 features.

---

## 4. Frequent "No confident gesture detected" Predictions

### Causes & Calibration:
- **Low Confidence Threshold**: The default threshold is set to $0.70$. If gestures are subtle or hands are far from the camera, lower the threshold to $0.60$ using the slider.
- **Buffer Warmup**: A rolling window of 30 frames (1 second at 30 FPS) is required. Ensure gestures are performed smoothly across the full 1-second duration.
- **Lighting & Contrast**: Ensure your hand is adequately illuminated without extreme backlighting.
