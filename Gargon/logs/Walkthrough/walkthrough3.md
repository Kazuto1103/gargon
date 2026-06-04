# Walkthrough: Surface Roughness Scanning System (Fase 1 & Fase 2)

We have successfully completed Fase 1 (Offline Training & Parameter Comparison Pipeline) and Fase 2 (Edge Video Streaming & Input Calibration Pipeline).

---

## FASE 1: Offline Training & Parameter Comparison Pipeline

We resolved the convergence issues by introducing feature scaling and training hyperparameter tuning.

### Changes Made
- **Comparison Pipeline Script**: [train_experiment.py](file:///d:/Project/Gargon/src/train_experiment.py)
- **Normalisation**: Injected `StandardScaler` from `sklearn.preprocessing` to standardiseGLCM features (which were in hundreds) and LBP features (between 0 and 1).
- **Accelerated Learning**: Increased the initial learning rate (`learning_rate_init=0.01`) in `MLPClassifier` to achieve fast convergence.

### Summary of Results
| Scenario | GLCM d | LBP R | Epochs | Final Loss | Train Acc | Test Acc |
|---|---|---|---|---|---|---|
| **RUN 1** (Mikro) | 1 | 1 | 8 | 0.058182 | 97.84% | 98.03% |
| **RUN 2** (Mikro) | 1 | 1 | 15 | 0.038988 | 98.15% | 98.26% |
| **RUN 3** (Makro) | 3 | 2 | 15 | 0.016955 | 99.69% | 98.15% |

- **Log Report**: [logs/experiment_report_20260604_211931.txt](file:///d:/Project/Gargon/logs/experiment_report_20260604_211931.txt)

---

## FASE 2: Edge Video Streaming & Input Calibration Pipeline

We implemented a modular, object-oriented live video stream processor that reads frames in a background thread, isolates the Scanning Zone, draws a Cybernetic Sci-Fi HUD overlay, and saves session logs.

### Changes Made
- **File Created**: [live_stream.py](file:///d:/Project/Gargon/src/live_stream.py)
- **Threaded Frame Acquisition**: Built `ThreadedVideoReader` to fetch and queue frames asynchronously. Set buffer size (`CAP_PROP_BUFFERSIZE`) to 1 to reduce lag.
- **Dynamic Camera Source Loading**: Configured default `CAMERA_SOURCE = 1` for DroidCam virtual camera. Added robust automatic fallbacks to camera index 2, then index 0. Also supports IP camera URL strings.
- **Scanning Zone ROI**: Computes a static 250 x 250 pixels Scanning Zone centered horizontally at the bottom-center of the screen. Added the placeholder comment inside the processing loop: `# TODO: Suntikkan Ekstraksi Fitur Hybrid & Prediksi Model Fase 1 di sini`.
- **Cybernetic HUD Overlay**: Renders a neon green HUD frame containing:
  - Brackets surrounding the Scanning Zone.
  - Information indicators for FPS, Video Resolution, and status `SYS_STATUS: CALIBRATED / READY`.
  - A flashing live scanner dot.
- **Session Logging**: Pressing 'q' exits gracefully, releasing the camera resources (`cap.release()`), and logs session statistics (source, resolution, average FPS) to `logs/camera_session_[TIMESTAMP].txt`.

### Verification & Performance
We verified the script running in headless test mode for 15 frames:
```powershell
venv\Scripts\python.exe src\live_stream.py --test 15
```

- **Output Session Log**: [logs/camera_session_20260604_214106.txt](file:///d:/Project/Gargon/logs/camera_session_20260604_214106.txt)
- **Log Summary**:
  - Source: camera index 1
  - Resolution: 640 x 480
  - Frame Rate: **29.35 FPS** (smooth live streaming target)
  - Release State: **OK** (gracefully released resource)
