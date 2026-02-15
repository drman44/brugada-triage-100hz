#!/usr/bin/env bash
set -e
bash scripts/00_download_data.sh
bash scripts/01_preprocess.sh
bash scripts/02_train_eval_cnn.sh
bash scripts/03_make_figures.sh
echo "DONE. See outputs/figures and outputs/metrics.json"
