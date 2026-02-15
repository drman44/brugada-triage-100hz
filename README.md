[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18649820.svg)](https://doi.org/10.5281/zenodo.18649820)

# brugada-triage-100hz

Open and reproducible deep learning pipeline for confidence-gated (selective referral) triage of Brugada syndrome from low-resolution resting ECG (100 Hz) using the PhysioNet Brugada-HUCA dataset.

⚠️ **This repository corresponds to the version used in the submitted manuscript.**

---

## Overview

This repository provides a fully reproducible pipeline for detecting Brugada syndrome from resting 12-lead ECG recordings sampled at 100 Hz.

The project emphasizes:

- ✅ Reproducibility  
- ✅ Version control  
- ✅ Locked evaluation protocol  
- ✅ Long-term archival via DOI  

---

## What this repository reproduces

**Input**
- V1–V3 median beat extracted from resting 12-lead ECG sampled at 100 Hz

**Model**
- Compact 1D Convolutional Neural Network (CNN) producing probability of Brugada syndrome via sigmoid output

**Evaluation**
- Stratified 3-fold cross-validation  
- Pooled out-of-fold predictions  

**Clinical framing**
- High-specificity operating point  
- Locked triage threshold (τ)  
- Confidence-gated referral strategy  

---

## Dataset (Public)

We use the **PhysioNet Brugada-HUCA database** (WFDB format: `.hea` / `.dat` + metadata).

🚨 **Important:**  
This repository does **NOT** redistribute ECG recordings or any patient-level data.

Users must obtain the dataset directly from PhysioNet and comply with its usage terms.

---

## Reproducibility & Archival

- Version-controlled repository  
- Tagged release  
- Archived with DOI: **10.5281/zenodo.18649820**  
- Fixed random seeds  
- Configuration tracked in `/configs`

This ensures long-term scientific reproducibility.

---

## Quickstart (End-to-End)

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
