# SoM2025 Baseline (Official) – README

> This repository provides the **official baseline** for the **SoM2025 Challenge: Adapting WiFo for Wireless Physical-Layer Tasks.**.
> It includes training scripts for the three tasks, instructions to reproduce the scores, and the **expected submission format**.

## 1) Challenge Overview

The SoM2025 challenge assesses how effectively WiFo-Base can be **adapted** to multiple wireless physical-layer tasks through fine-tuning.

**Tasks:**

- **Task 1 – Channel Estimation from Pilots**
  Reconstruct the **full channel matrix** from sparse pilots (interpolation / recovery).

- **Task 2 – LoS vs. NLoS Classification**
  Predict whether a radio link contains a **Line-of-Sight** path.

- **Task 3 – (Vision-aided) Wireless Localization**
  Estimate precise vehicle position from channel data (the baseline uses channel features only).

**Data sources and settings:**

| Item | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| **Generator** | Quadriga | Quadriga | SynthSoM |
| **Scenario** | UMa-NLoS | UMi-LoS + UMi-NLoS | Cross-Road |
| **# Train Samples** | 900 | 300 | 500 |
| **Channel Tensor Dim.** | `4 × 32 × 64` | `24 × 8 × 128` | `1 × 128 × 32` |
| **Pilot/Config** | pilots on every 4 subcarriers | — | — |
| **Baseline Metric** | NMSE = **0.318** | F1 = **0.828** | MAE = **9.83** |

> **Overall score:**
> Combined/normalized across tasks with a 0–1 cap.

---

## 2) Dataset Structure & File Formats

Place the **[official dataset](https://huggingface.co/datasets/PPASS/som2025/tree/main)** under `./dataset/` (see the download section below).
Each task has a consistent, lightweight format intended for easy NumPy loading.

```
dataset/
├── Task1/
│   ├── X_pilot_train.mat    # (N1, 4, 32, 16)  sparse pilots
│   ├── X_train.mat          # (N1, 4, 32, 64)  ull ground-truth channels (target)
│   └── ...
├── Task2/
│   ├── X_train.mat          # (N2, 24, 8, 128) channel tensors
│   ├── L_train.mat          # (N2,) int {0=NLoS, 1=LoS}
│   └── ...
└── Task3/
    ├── X_train.mat          # (N3, 1, 128, 32) channel tensors
    ├── imgs_train.mat       # (N3, 3, 512, 512) rgb
    ├── location_train.mat   # (N3, 2)  continuous (x, y) in meters
    └── ...
```
 
---

## 3) Getting Started

### 3.1 Environment

We recommend Python **3.10+** and CUDA-enabled PyTorch (optional but recommended).

```bash
# (Optional) Create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

> You may pin CUDA-specific wheels for PyTorch according to your platform.

### 3.2 Download the Official Dataset

Download the official release and unzip it to `./dataset/`:

- **Official download: [Link](https://huggingface.co/datasets/PPASS/som2025/tree/main)**

---

## 4) Train the Baseline

Each task has a single entry-point script. **No extra setup** is required beyond `requirements.txt` and `./dataset/`.

> Default paths assume `dataset/Task{1,2,3}`. Use `--data_dir` to override.

### 4.1 Task 1 – Channel Estimation

```bash
python main_fine_tune_T1.py 
```

**Metric:** NMSE (lower is better).
The baseline reaches **~0.318 NMSE** as shown in the slide with default settings.

### 4.2 Task 2 – LoS/NLoS Classification

```bash
python main_fine_tune_T2.py 
```

**Metric:** F1 score (higher is better).
The baseline reaches **~0.828 F1**.

### 4.3 Task 3 – (Vision-aided) Wireless Localization

```bash
python main_fine_tune_T3.py 
```

**Metric:** MAE in meters (lower is better).
The baseline reaches **~9.83 MAE**.

> Reproducibility: we fix random seeds in the scripts and save checkpoints/metrics to `./outputs/<task>/`.

---

## 5) Expected Results

After training with the official dataset and default hyper-parameters, your validation/test performance should be close to the slide:

| Task | Metric | Baseline Result (approx.) |
|---|---|---|
| Task 1 | NMSE ↓ | **0.318** |
| Task 2 | F1 ↑ | **0.828** |
| Task 3 | MAE ↓ | **9.83** |

> Minor deviations (±) are normal due to random seeds and hardware stack.

---

## 6) Submission Format

Submit **two files**:

1) **Predictions JSON**: `submission.json` (see `[submission_demo.json](https://huggingface.co/datasets/PPASS/som2025/tree/main)` for reference)
2) **Average Trainable Parameters**: `avg_trainable_params.txt` (see below)

### 6.1 Predictions JSON

The predictions for each **test** split must be serialized as **lists** (JSON-serializable) using the following **top-level keys**:

```json
{
  "Task1": [[[[...], ...], ...], ...],  // reconstructed full channels for Task 1
  "Task2": [0, 1, 0, ...],              // 0 = NLoS, 1 = LoS (or probability if required by phase)
  "Task3": [[x1, y1], [x2, y2], ...]    // predicted (x, y) in meters
}
```

- **Task 1**: Each element is a **`4×32×64`** channel tensor (full reconstruction).
- **Task 2**: One scalar **class** per sample (`0`/`1`) or, if the evaluation server specifies probabilities, a single float in `[0,1]` per sample.
- **Task 3**: One **2-D coordinate** `(x, y)` per sample, in meters.

A convenient Python snippet to write the JSON:

```python
import json, numpy as np

# Load or compute your predictions here:
pred_t1 = np.random.randn(N1, 4, 32, 64)     # full channel reconstructions
pred_t2 = np.random.randint(0, 2, size=(N2,))# classes or probabilities
pred_t3 = np.random.randn(N3, 2)             # (x, y) in meters

payload = {
    "Task1": pred_t1.tolist(),
    "Task2": pred_t2.tolist(),
    "Task3": pred_t3.tolist(),
}

with open("submission.json", "w") as f:
    json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
```

> Ensure that the **order of predictions** exactly matches the test set order provided by the official loader in each script.

### 6.2 Average Trainable Parameters

Report the **average number of trainable parameters** used by your fine-tuning scheme (e.g., full-tuning vs. LoRA/PEFT).
Save a single number to `avg_trainable_params.txt` (float or int; same count used for all three tasks or the average across the three, depending on your method).

Example:

```
# avg_trainable_params.txt
5.2e6
```

A helper in PyTorch:

```python
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total)
```

---




## 7) Citation

If you find this baseline useful in your work or publication, please cite the SoM2025 challenge and this repository.

```bibtex
@misc{SoM2025Baseline,
  title  = {SoM2025: Adapting WiFo for Wireless Physical-Layer Tasks — Official Baseline},
  year   = {2025},
  note   = {GitHub repository},
  howpublished = {\url{<REPO_URL>}}
}
```

---

## 8) License

This baseline is released under the **Apache 2.0** License (unless otherwise specified in the repo).

---

## 9) Contact

For questions or issues, please open a GitHub Issue or reach the organizers at **[INSERT CONTACT/EMAIL]**.

> **Checklist before you submit**
> - [ ] You trained with the **official dataset** and default splits
> - [ ] You exported predictions to **`submission.json`** with keys `Task1/Task2/Task3`
> - [ ] You included **`avg_trainable_params.txt`** with the average trainable parameter count
> - [ ] Your submission files are named exactly as required and are readable
```






