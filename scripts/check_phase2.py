"""Phase 2 sanity checks. Run:  python scripts/check_phase2.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.data import get_dataloaders  # noqa: E402
from src.model import (BN_TYPES, build_model, count_params, make_optimizer,  # noqa: E402
                       set_regime, set_train_mode)
from src.utils import get_device, load_config, set_seed  # noqa: E402


def check(cond, msg):
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    return cond


def main():
    cfg = load_config()
    device = get_device()
    ok = True
    print("Device:", device)

    # 1. Trainable parameter counts per regime
    counts = {}
    for regime in "ABC":
        m = set_regime(build_model(21), regime)
        total, trainable = count_params(m)
        counts[regime] = trainable
        print(f"[INFO] regime {regime}: trainable {trainable:,} / total {total:,}")
    ok &= check(counts["A"] == 1024 * 21 + 21, "regime A trainable params = 21,525")
    ok &= check(counts["A"] < counts["B"] < counts["C"], "A < B < C trainable params")

    # 2. Frozen BatchNorm layers stay in eval mode
    for regime in "ABC":
        m = set_regime(build_model(21), regime)
        set_train_mode(m)
        bns = [b for b in m.modules() if isinstance(b, BN_TYPES)]
        n_train = sum(b.training for b in bns)
        print(f"[INFO] regime {regime}: {n_train}/{len(bns)} BN layers in train mode")
        if regime == "A":
            ok &= check(n_train == 0, "A: all BN layers frozen (eval)")
        if regime == "C":
            ok &= check(n_train == len(bns), "C: all BN layers in train mode")
        if regime == "B":
            ok &= check(0 < n_train < len(bns), "B: only the last module's BN in train mode")

    # 3. Overfit a single batch (regime C): the loss must drop a lot
    set_seed(0)
    train_loader, _, _ = get_dataloaders(cfg, seed=0)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model = set_regime(build_model(21), "C").to(device)
    opt = make_optimizer(model, {"lr_head": 1e-2, "lr_backbone": 1e-2,
                                 "momentum": 0.9, "weight_decay": 0.0})
    crit = nn.CrossEntropyLoss()
    first = last = None
    for step in range(40):
        set_train_mode(model)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    ok &= check(last < 0.5 * first, f"overfit one batch: loss {first:.3f} -> {last:.3f}")

    print("\nPHASE 2:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
