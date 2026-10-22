"""Download and extract the UC Merced Land Use dataset.

Usage:
    python scripts/download_data.py                 # default location
    python scripts/download_data.py --dest /content/data

Several mirrors are tried in order (the original university server is
sometimes unreachable). The zip on the Hugging Face mirror (TorchGeo) is
a redistribution of the original file without modification.

If every mirror fails, download the zip manually, extract it so that you
get <dest>/UCMerced_LandUse/Images/<21 class folders>, and re-run this script
(it will only verify the data), or point URBAN_DATA_ROOT to the folder that
contains the 21 class folders.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import PROJECT_ROOT  # noqa: E402

MIRRORS = [
    "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip",
    "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
]
TIMEOUT = 30  # seconds without data before giving up on a mirror


def download(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(out_path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, step = 0, 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and done * 10 // total > step:
                step = done * 10 // total
                print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB")


def count_images(images_dir):
    classes = [d for d in images_dir.iterdir() if d.is_dir()]
    n = sum(1 for c in classes for _ in c.glob("*.tif"))
    return len(classes), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--url", default=None, help="override the mirror list")
    args = ap.parse_args()

    dest = Path(args.dest)
    images_dir = dest / "UCMerced_LandUse" / "Images"
    if images_dir.is_dir():
        print(f"Dataset already present: {images_dir}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "UCMerced_LandUse.zip"
        for url in ([args.url] if args.url else MIRRORS):
            print(f"Downloading {url} ...")
            try:
                download(url, zip_path)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  failed: {e}")
                zip_path.unlink(missing_ok=True)
        else:
            print("All mirrors failed. See the instructions at the top of this file.")
            sys.exit(1)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink()

    n_cls, n_img = count_images(images_dir)
    print(f"Found {n_cls} classes and {n_img} images in {images_dir}")
    if (n_cls, n_img) != (21, 2100):
        print("WARNING: expected 21 classes and 2100 images.")
        sys.exit(1)
    print("OK. Data root:", images_dir)


if __name__ == "__main__":
    main()