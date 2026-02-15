import argparse
import os
import yaml
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks


def bandpass(x, fs, lo=5.0, hi=15.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(sig_ii, fs):
    x = bandpass(sig_ii, fs)
    x = np.abs(x)
    peaks, _ = find_peaks(
        x,
        distance=max(25, int(0.25 * fs)),  # ~250 ms
        prominence=np.std(x),
    )
    return peaks


def median_beat(signals, lead_idx_map, fs, leads=("V1", "V2", "V3")):
    # 0.2 s pre-R, 0.4 s post-R -> 60 samples at 100 Hz
    pre = int(0.2 * fs)
    post = int(0.4 * fs)

    ii = signals[:, lead_idx_map["II"]]
    r = detect_r_peaks(ii, fs)

    beats = []
    for p in r:
        if p - pre < 0 or p + post > len(ii):
            continue
        segs = []
        for ld in leads:
            seg = signals[p - pre : p + post, lead_idx_map[ld]]
            segs.append(seg)
        beats.append(np.stack(segs, axis=0))  # (3, L)

    if len(beats) < 3:
        return None

    beats = np.stack(beats, axis=0)  # (n_beats, 3, L)
    return np.median(beats, axis=0).astype(np.float32)  # (3, L)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    root = cfg["data"]["root"]
    fs = int(cfg["data"]["fs"])
    leads = cfg["data"]["leads"]
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(root, "metadata.csv"))
    ids = meta["patient_id"].astype(str).tolist()
    y_all = meta["brugada"].astype(int).values

    X_list, y_list, id_list = [], [], []
    files_root = os.path.join(root, "files")

    for pid, y in zip(ids, y_all):
        rec_path = os.path.join(files_root, pid, pid)
        try:
            rec = wfdb.rdrecord(rec_path)
        except Exception:
            continue

        sig = rec.p_signal  # (samples, leads)
        names = list(rec.sig_name)
        lead_idx = {n: i for i, n in enumerate(names)}

        needed = ["II"] + list(leads)
        if any(n not in lead_idx for n in needed):
            continue

        mb = median_beat(sig, lead_idx, fs, leads=tuple(leads))
        if mb is None:
            continue

        X_list.append(mb)
        y_list.append(int(y))
        id_list.append(pid)

    X = np.stack(X_list, axis=0)  # (N, 3, 60)
    y = np.array(y_list, dtype=np.int64)
    ids_out = np.array(id_list)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "ids.npy"), ids_out)

    print(f"Saved: X={X.shape} y={y.shape}")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
