"""Dataset indexing and stratified splits (no PyTorch dependency)."""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_index(root):
    """Scan root/<class>/<image> and return (samples, classes).

    samples: list of (relative_path, label_index), sorted for determinism.
    classes: sorted list of class names (folder names).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Run `python scripts/download_data.py` first."
        )
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples = []
    for label, cls in enumerate(classes):
        files = sorted(
            p for p in (root / cls).iterdir() if p.suffix.lower() in IMG_EXTS
        )
        samples.extend((str(p.relative_to(root)), label) for p in files)
    return samples, classes


def stratified_split(samples, test_size, seed):
    """Stratified train/test split; the result depends only on `seed`."""
    idx = list(range(len(samples)))
    labels = [s[1] for s in samples]
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=seed
    )
    return sorted(train_idx), sorted(test_idx)


def get_split(samples, classes, test_size, seed, results_dir):
    """Return (train_idx, test_idx), creating and saving the split if needed.

    The split is stored in results/splits/seed{seed}.json so that regimes
    A, B and C are guaranteed to use exactly the same data.
    """
    split_file = Path(results_dir) / "splits" / f"seed{seed}.json"
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["classes"] == classes and data["n_samples"] == len(samples):
            return data["train_idx"], data["test_idx"]

    train_idx, test_idx = stratified_split(samples, test_size, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "test_size": test_size,
                "classes": classes,
                "n_samples": len(samples),
                "train_idx": train_idx,
                "test_idx": test_idx,
            },
            f,
        )
    return train_idx, test_idx
