# Walkthrough: Surface Roughness Scanning System (Fase 1, 2, & 3)

We have successfully completed all three phases of the Surface Roughness Scanning System, integrating offline training, threaded live video streaming, real-time ROI feature extraction, and dynamic cybernetic Sci-Fi HUD visualization.

---

## FASE 1: Offline Training & Parameter Comparison Pipeline
- **Script**: [train_experiment.py](file:///d:/Project/Gargon/src/train_experiment.py)
- **Normalisation**: Injected `StandardScaler` from `sklearn.preprocessing` to standardise feature ranges.
- **Trained Model**: Achieved **98.26% test accuracy** using MLPClassifier (Run 2: GLCM $d=1$, LBP $R=1$, 15 epochs, `learning_rate_init=0.01`).

---

## FASE 2: Edge Video Streaming & Input Calibration Pipeline
- **Script**: [live_stream.py](file:///d:/Project/Gargon/src/live_stream.py)
- **Threaded Frame Acquisition**: Implemented `ThreadedVideoReader` background thread with a buffer size of 1.
- **Scanning Zone ROI**: Centered a static 250 x 250 pixels ROI bounding box at the bottom-center of the frame.
- **HUD Frame Rate**: Captured camera feeds smoothly at target **~30 FPS**.

---

## FASE 3: Real-Time Inference & Cybernetic HUD Activation

We successfully serialized the best models from Fase 1, integrated them into the live stream pipeline from Fase 2, and activated the dynamic HUD color-coding alerts.

### 1. Model Serialization (Fase 1 Export)
The `train_experiment.py` script was modified to save the trained `MLPClassifier` and `StandardScaler` to:
- [models/surface_classifier.pkl](file:///d:/Project/Gargon/models/surface_classifier.pkl)
- [models/scaler.pkl](file:///d:/Project/Gargon/models/scaler.pkl)

### 2. Live Stream Integration (Fase 2 to Fase 3)
The `live_stream.py` script was modified to load these models at startup using `joblib.load()`. During the stream:
- **ROI Resizing**: The cropped 250x250 pixels Scanning Zone is resized in memory to 50x50 pixels (matching the training input size).
- **Inference**: Hybrid features ($d=1, R=1$) are extracted from the 50x50 image, transformed via `scaler.pkl`, and predicted using `surface_classifier.pkl`.
- **Dynamic Bounding Box Color & Teks HUD**:
  - **Nominal (Label 0)**: Green Bounding Box & Corners. Display text: `SURFACE: NOMINAL | CONF: [Value]%`.
  - **Anomalous/Rough (Label 1)**: Red Alert Bounding Box & Corners (with rapid blinking animation). Display text: `STATUS: ANOMALOUS (ROUGH) | CONF: [Value]%`.

### 3. Verification & Performance Logs
We verified the complete pipeline in headless test mode for 30 frames:
```powershell
venv\Scripts\python.exe src\live_stream.py --test 30
```

- **Output Session Log**: [logs/camera_session_20260604_215834.txt](file:///d:/Project/Gargon/logs/camera_session_20260604_215834.txt)
- **Log Verification Details**:
  - Model Load Status: **[OK] Model & Scaler Berhasil Dimuat**
  - Total Frame Prediksi: **30 frame**
  - Release State: **OK** (cap.release() gracefully executed)
