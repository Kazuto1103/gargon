# Walkthrough: Offline Training & Parameter Comparison Pipeline (Modifikasi & Normalisasi)

We have successfully resolved the convergence issues and completed the patch for the offline training and parameter comparison pipeline.

## Changes Made

### 1. Comparison Pipeline Patch
- **File Updated**: [train_experiment.py](file:///d:/Project/Gargon/src/train_experiment.py)
- **Features Injected**:
  - **Feature Scaling**: Added `StandardScaler` from `sklearn.preprocessing` to standardise feature ranges. It fits parameters on `X_train` and transforms both `X_train` and `X_test` to prevent bias towards GLCM values.
  - **Accelerated Convergence**: Configured `learning_rate_init=0.01` in the `MLPClassifier` to ensure fast learning and stabilization over 8-15 epochs.
  - **Strict Directory Lock**: Ensured dataset loading is locked exactly to `data/NewData/Cropped50x/`, mapping 00-21 to Nominal (Label 0) and 24-45 to Anomalous (Label 1).
  - **Educational Log Report**: Writes reports containing details of `StandardScaler` injection, dataset distribution, ASCII loss progression charts, confusion matrices, and metrics.

## Verification & Results

### Execution Command
```powershell
venv\Scripts\python.exe src\train_experiment.py
```

### Generated Log Report
- [logs/experiment_report_20260604_211931.txt](file:///d:/Project/Gargon/logs/experiment_report_20260604_211931.txt)

### Summary of Results
| Scenario | GLCM d | LBP R | Epochs | Final Loss | Train Acc | Test Acc |
|---|---|---|---|---|---|---|
| **RUN 1** (Mikro) | 1 | 1 | 8 | 0.058182 | 97.84% | 98.03% |
| **RUN 2** (Mikro) | 1 | 1 | 15 | 0.038988 | 98.15% | 98.26% |
| **RUN 3** (Makro) | 3 | 2 | 15 | 0.016955 | 99.69% | 98.15% |

- **Observations**: With the injection of `StandardScaler` and a learning rate of `0.01`, testing accuracy immediately soared to **98.03% - 98.26%** with final losses dropping as low as **0.016955**. This confirms that aligning the GLCM and LBP feature scales was the key to successful convergence.
