"""Phase 1 sanity checks. Run:  python scripts/check_phase1.py"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.splits import build_index, stratified_split  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def check(cond, msg):
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    return cond


def main():
    cfg = load_config()
    ok = True

    samples, classes = build_index(cfg["data"]["root"])
    ok &= check(len(classes) == 21, f"21 classes found ({len(classes)})")
    ok &= check(len(samples) == 2100, f"2100 images found ({len(samples)})")

    tr, te = stratified_split(samples, cfg["data"]["test_size"], seed=0)
    ok &= check(len(tr) == 1680 and len(te) == 420,
                f"split sizes: train={len(tr)} test={len(te)} (expected 1680/420)")
    c_tr = Counter(samples[i][1] for i in tr)
    c_te = Counter(samples[i][1] for i in te)
    ok &= check(all(v == 80 for v in c_tr.values()) and
                all(v == 20 for v in c_te.values()),
                "stratified: 80 train / 20 test images in every class")
    ok &= check(not set(tr) & set(te), "no overlap between train and test")

    tr2, te2 = stratified_split(samples, cfg["data"]["test_size"], seed=0)
    ok &= check(tr == tr2 and te == te2, "same seed -> identical split")
    tr3, _ = stratified_split(samples, cfg["data"]["test_size"], seed=1)
    ok &= check(tr != tr3, "different seeds -> different splits")

    # Torch part: dataloaders and visual check of one augmented batch
    try:
        import torch
        from torchvision.utils import make_grid, save_image
        from src.data import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders

        set_seed(0)
        train_loader, test_loader, _ = get_dataloaders(cfg, seed=0)
        ok &= check(len(train_loader.dataset) == 1680 and
                    len(test_loader.dataset) == 420, "DataLoader dataset sizes")
        x, y = next(iter(train_loader))
        ok &= check(tuple(x.shape[1:]) == (3, 224, 224),
                    f"batch tensor shape {tuple(x.shape)}")

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        grid = make_grid((x[:16] * std + mean).clamp(0, 1), nrow=8)
        out = Path(cfg["paths"]["results_dir"]) / "figures"
        out.mkdir(parents=True, exist_ok=True)
        save_image(grid, out / "augmented_batch.png")
        print(f"[INFO] Saved {out / 'augmented_batch.png'} -> inspect it visually")
    except ImportError:
        print("[SKIP] PyTorch not installed: DataLoader checks skipped")

    print("\nPHASE 1:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
