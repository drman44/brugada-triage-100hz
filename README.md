# brugada-triage-100hz

Open and reproducible deep learning pipeline for confidence-gated (selective referral) triage of Brugada syndrome from low-resolution resting ECG (100 Hz) using the PhysioNet Brugada-HUCA dataset.

## What this repository reproduces
- Input: V1–V3 median beat extracted from resting 12-lead ECG sampled at 100 Hz
- Model: compact 1D CNN producing p(BrS) via sigmoid
- Evaluation: stratified 3-fold cross-validation with pooled out-of-fold predictions
- Triage: high-specificity operating point with a locked threshold (tau)

## Dataset (public)
We use the PhysioNet Brugada-HUCA database (WFDB format: .hea/.dat) and its metadata.csv.
This repository does NOT redistribute any ECG data.

## Quickstart (end-to-end)

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download data (to local data/brugada-huca/1.0.0/)
```bash
bash scripts/00_download_data.sh
```

### 3) Preprocess (creates V1–V3 median beats at 100 Hz)
```bash
bash scripts/01_preprocess.sh
```

### 4) Train + evaluate (CV, pooled OOF predictions)
```bash
bash scripts/02_train_eval_cnn.sh
```

### 5) Generate figures
```bash
bash scripts/03_make_figures.sh
```

Or run everything:
```bash
bash scripts/05_run_all.sh
```

Outputs are written to outputs/ (models, predictions, logs, figures).

## Reproducibility
- Fixed random seeds
- Configuration tracked in configs/
- No threshold tuning on test: the triage threshold is selected from pooled out-of-fold predictions (validation-only) and then applied unchanged.

## Code availability
Full analysis code remains publicly available in this repository. No patient-level data are included.
