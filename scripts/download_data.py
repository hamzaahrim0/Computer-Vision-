"""Download and extract the UC Merced Land Use dataset.

Usage:
    python scripts/download_data.py                 # default location
    python scripts/download_data.py --dest /content/data

If the university server is down, download the zip manually, extract it so
that you get <dest>/UCMerced_LandUse/Images/<21 class folders>, or point
URBAN_DATA_ROOT to the folder that contains the 21 class folders.
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import PROJECT_ROOT  # noqa: E402

URL = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"


def count_images(images_dir):
    classes = [d for d in images_dir.iterdir() if d.is_dir()]
    n = sum(1 for c in classes for _ in c.glob("*.tif"))
    return len(classes), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    dest = Path(args.dest)
    images_dir = dest / "UCMerced_LandUse" / "Images"
    if images_dir.is_dir():
        print(f"Dataset already present: {images_dir}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "UCMerced_LandUse.zip"
        print(f"Downloading {args.url} ...")
        urllib.request.urlretrieve(args.url, zip_path)
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
