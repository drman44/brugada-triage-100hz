import argparse
import os
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        m = x.mean()
        s = x.std() + 1e-6
        x = (x - m) / s
        return torch.tensor(x), torch.tensor(self.y[idx])


class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        z = self.net(x)
        z = self.gap(z).squeeze(-1)
        return self.fc(z).squeeze(-1)  # logits


def find_tau_for_specificity(y_true, p, target_spec=0.90):
    thresholds = np.linspace(0, 1, 1001)
    best = None
    for t in thresholds:
        y_hat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-12)
        sens = tp / (tp + fn + 1e-12)
        if spec >= target_spec:
            if best is None or sens > best["sens"]:
                best = {"tau": float(t), "spec": float(spec), "sens": float(sens)}
    return best


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    seed = int(cfg["seed"])
    set_seed(seed)

    out_dir = cfg["data"]["out_dir"]
    X = np.load(os.path.join(out_dir, "X.npy"))
    y = np.load(os.path.join(out_dir, "y.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(
        n_splits=int(cfg["cv"]["n_splits"]), shuffle=True, random_state=seed
    )
    oof = np.zeros(len(y), dtype=np.float32)

    epochs = int(cfg["train"]["epochs"])
    bs = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        ds_tr = ECGDataset(Xtr, ytr)
        ds_va = ECGDataset(Xva, yva)
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False)

        model = CompactCNN().to(device)

        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_auc = -1.0
        best_state = None

        for _ in range(epochs):
            model.train()
            for xb, yb in dl_tr:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for xb, _ in dl_va:
                    xb = xb.to(device)
                    p = torch.sigmoid(model(xb)).cpu().numpy()
                    preds.append(p)
            pva = np.concatenate(preds).astype(np.float32)
            auc = roc_auc_score(yva, pva) if len(np.unique(yva)) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl_va:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(p)
        oof[va] = np.concatenate(preds).astype(np.float32)

        os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
        torch.save(best_state, os.path.join(out_dir, "models", f"cnn_fold{fold}.pt"))
        print(f"Fold {fold}: best val AUROC={best_auc:.3f}")

    auroc = float(roc_auc_score(y, oof))
    apv = float(average_precision_score(y, oof))

    tau_locked = float(cfg["triage"]["tau_locked"])
    target_spec = float(cfg["triage"]["target_specificity"])
    tau_best = find_tau_for_specificity(y, oof, target_spec=target_spec)

    def summary_at_tau(tau):
        yhat = (oof >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-12)
        sens = tp / (tp + fn + 1e-12)
        ppv = tp / (tp + fp + 1e-12)
        npv = tn / (tn + fn + 1e-12)
        ref = (yhat == 1).mean()
        prev = y.mean()
        prev_flag = y[yhat == 1].mean() if (yhat == 1).any() else 0.0
        return {
            "tau": float(tau),
            "specificity": float(spec),
            "sensitivity": float(sens),
            "ppv": float(ppv),
            "npv": float(npv),
            "referral_rate": float(ref),
            "prevalence_overall": float(prev),
            "prevalence_flagged": float(prev_flag),
        }

    metrics = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "auroc_oof": auroc,
        "average_precision_oof": apv,
        "triage_locked": summary_at_tau(tau_locked),
        "triage_datadriven_from_oof": tau_best,
    }

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_pred.npy"), oof)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
