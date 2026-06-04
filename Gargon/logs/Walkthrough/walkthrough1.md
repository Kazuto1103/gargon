# Walkthrough: Offline Training & Parameter Comparison Pipeline

We have successfully designed, implemented, and verified the offline training and parameter comparison pipeline.

## Changes Made

### 1. New Comparison Pipeline Script
- **File Created**: [train_experiment.py](file:///d:/Project/Gargon/src/train_experiment.py)
- **Features**:
  - **Dataset Loading**: Scans subfolders (`00` to `45`) under `data/NewData/Cropped50x/` and maps folder ranges to labels (`0: Nominal/Mulus`, `1: Anomalous/Kasar`) dynamically.
  - **Imbalance Analysis**: Computes class counts and prints the results in a formatted layout to analyze data imbalance.
  - **Train/Test Splitting**: Splits image paths cleanly into 60% Training and 40% Testing using `train_test_split(random_state=42)`.
  - **Hybrid Feature Extraction**: Extracts statistical GLCM parameters (Contrast, Homogeneity, Energy, Correlation) and Local Binary Pattern (LBP) histograms, then fuses them.
  - **MLP Classifier Epoch Monitoring**: Trains an `MLPClassifier` using a manual epoch loop via `.partial_fit()`, logging training loss, training accuracy, and test accuracy on every epoch.
  - **Sequential Scenario Execution**: Runs Run 1 ($d=1, R=1$, 8 epochs), Run 2 ($d=1, R=1$, 15 epochs), and Run 3 ($d=3, R=2$, 15 epochs).
  - **Educational Text Report**: Generates a timestamped report under `logs/` detailing parameter logs, a custom ASCII loss trend progress chart, confusion matrices, and detailed classification reports.

## Verification & Results

### Execution Command
```powershell
venv\Scripts\python.exe src\train_experiment.py
```

### Generated Log Report
- [logs/experiment_report_20260604_210737.txt](file:///d:/Project/Gargon/logs/experiment_report_20260604_210737.txt)

### Summary of Results
| Scenario | GLCM d | LBP R | Epochs | Final Loss | Train Acc | Test Acc |
|---|---|---|---|---|---|---|
| **RUN 1** (Mikro) | 1 | 1 | 8 | 0.775400 | 44.48% | 44.44% |
| **RUN 2** (Mikro) | 1 | 1 | 15 | 0.766865 | 45.37% | 46.35% |
| **RUN 3** (Makro) | 3 | 2 | 15 | 0.811268 | 55.59% | 55.56% |

- **Observations**: The Macro configuration in RUN 3 (GLCM $d=3$, LBP $R=2$) shows higher testing accuracy (55.56%) compared to the micro parameter settings of RUN 1 & 2, showing that larger spatial patterns provide better representation for roughness classification in this dataset.
