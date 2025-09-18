# SoM2025 Baseline (Official) – README

> **Adapting WiFo for Wireless Physical-Layer Tasks.**  
> This repository provides the **official baseline** for the **SoM2025 Challenge**.  
> It includes training scripts for the three tasks, instructions to reproduce the scores, and the **expected submission format**.

---

## 1) Challenge Overview

The SoM2025 challenge evaluates how well a WiFi-oriented foundation model (WiFo-Base) can be **adapted** to multiple wireless physical-layer tasks via fine-tuning.

**Tasks (from the slide):**

- **Task 1 – Channel Estimation from Pilots**  
  Reconstruct the **full channel matrix** from sparse pilots (interpolation / recovery).

- **Task 2 – LoS vs. NLoS Classification**  
  Predict whether a radio link contains a **Line-of-Sight** path.

- **Task 3 – (Vision-assisted) Wireless Vehicle Localization**  
  Estimate precise vehicle position from channel data (the baseline uses channel features only).

**Data sources and settings (as shown in the slide):**

| Item | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Generator** | Quadriga | Quadriga | SynthSoM |
| **Scenario** | UMa-NLoS | UMi-LoS + UMi-NLoS | Cross-Road |
| **# Train Samples** | 900 | 300 | 500 |
| **Channel Tensor Dim.** | `4 × 32 × 64` | `24 × 8 × 128` | `1 × 128 × 32` |
| **Pilot/Config** | pilots on every 4 subcarriers | — | — |
| **Baseline Metric** | NMSE = **0.318** | F1 = **0.828** | MAE = **9.83** |

> **Overall score (illustrative from slide):**  
> Combined/normalized across tasks with a 0–1 cap.

---

## 2) Dataset Structure & File Formats

Place the **official dataset** under `./dataset/` (see the download section below).  
Each task has a consistent, lightweight format intended for easy NumPy loading.

dataset/
├── Task1/
│   ├── train/
│   │   ├── H_pilot.npy        # (N1, 4, 32, 64) sparse pilots
│   │   ├── pilot_mask.npy     # (N1, 4, 32, 64) boolean/binary pilot mask
│   │   └── H_full.npy         # (N1, 4, 32, 64) full ground-truth channels (target)
│   ├── val/                   # same files as train (smaller N)
│   └── test/
│       ├── H_pilot.npy        # (T1, 4, 32, 64)
│       └── pilot_mask.npy     # (T1, 4, 32, 64)
├── Task2/
│   ├── train/
│   │   ├── H.npy              # (N2, 24, 8, 128) channel tensors
│   │   └── y.npy              # (N2,) int {0=NLoS, 1=LoS}
│   ├── val/                   # (H.npy, y.npy)
│   └── test/
│       └── H.npy              # (T2, 24, 8, 128)
└── Task3/
    ├── train/
    │   ├── H.npy              # (N3, 1, 128, 32) channel tensors
    │   └── y.npy              # (N3, 2) continuous (x, y) in meters
    ├── val/                   # (H.npy, y.npy)
    └── test/
        └── H.npy              # (T3, 1, 128, 32)

