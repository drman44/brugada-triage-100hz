import argparse
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    out_dir = cfg["data"]["out_dir"]
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    y = np.load(os.path.join(out_dir, "y.npy"))
    p = np.load(os.path.join(out_dir, "oof_pred.npy"))

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (pooled OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure1_ROC_CNN.png"), dpi=300)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (pooled OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure2_PR_CNN.png"), dpi=300)
    plt.close()

    # Confusion matrix at locked tau
    with open(os.path.join(out_dir, "metrics.json"), "r", encoding="utf-8") as f:
        m = json.load(f)
    tau = float(m["triage_locked"]["tau"])
    yhat = (p >= tau).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (tau={tau:.3f})")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure3_ConfusionMatrix_CNN.png"), dpi=300)
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (pooled OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "Figure4_Calibration_CNN.png"), dpi=300)
    plt.close()

    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
