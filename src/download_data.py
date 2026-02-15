import argparse
import os
import zipfile
import urllib.request

PHYSIONET_BASE = "https://physionet.org/files/brugada-huca/1.0.0"


def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder, e.g., data/brugada-huca/1.0.0")
    args = ap.parse_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    metadata_path = os.path.join(out_root, "metadata.csv")
    files_zip_path = os.path.join(out_root, "files.zip")

    download(f"{PHYSIONET_BASE}/metadata.csv", metadata_path)
    download(f"{PHYSIONET_BASE}/files.zip", files_zip_path)

    files_dir = os.path.join(out_root, "files")
    os.makedirs(files_dir, exist_ok=True)

    with zipfile.ZipFile(files_zip_path, "r") as z:
        z.extractall(files_dir)

    print("OK.")
    print(f"Metadata: {metadata_path}")
    print(f"WFDB files extracted under: {files_dir}")


if __name__ == "__main__":
    main()
