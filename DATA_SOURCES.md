# Data sources and access

## Primary dataset
- Name: PhysioNet Brugada-HUCA (version 1.0.0)
- Format: WFDB (.hea/.dat) + metadata.csv
- Access: Public via PhysioNet. Users must comply with PhysioNet terms.
- Local storage: `data/brugada-huca/1.0.0/`

This repository does NOT redistribute any ECG recordings.

## How to obtain the data
Run:
- `bash scripts/00_download_data.sh`

This downloads:
- `metadata.csv`
- `files.zip` (WFDB records)

and extracts WFDB files under:
- `data/brugada-huca/1.0.0/files/<patient_id>/<patient_id>.hea/.dat`

## What we do with the data (high level)
1) Read WFDB records using `wfdb`
2) Detect R-peaks on Lead II (band-pass 5–15 Hz + peak finding)
3) Extract beats (0.2 s pre-R, 0.4 s post-R) and compute median beat per lead
4) Build V1–V3 median-beat tensor (3 × 60 samples) as CNN input
5) Train/evaluate compact 1D CNN with stratified 3-fold CV
6) Report pooled out-of-fold performance and confidence-gated triage metrics

## Data sharing
Only code, configs, and derived plots/metrics are shared here.
