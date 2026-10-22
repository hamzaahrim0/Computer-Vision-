"""Training / evaluation of one (regime, seed) run (Phase 2)."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import get_dataloaders
from .model import build_model, count_params, make_optimizer, set_regime, set_train_mode
from .utils import set_seed


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_epoch(model, loader, optimizer, criterion, device):
    set_train_mode(model)
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, loss_sum = [], [], 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item()
        y_true += y.cpu().tolist()
        y_pred += out.argmax(1).cpu().tolist()
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    return acc, loss_sum / len(y_true), y_true, y_pred


def run_experiment(cfg, regime, seed, device, runs_dir):
    """Train one run and save results/runs/{regime}_seed{seed}.json.

    Skips the run if its result file already exists (resumable).
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_file = runs_dir / f"{regime}_seed{seed}.json"
    if out_file.exists():
        print(f"[skip] {out_file.name} already done")
        return json.loads(out_file.read_text())

    tcfg = cfg["train"]
    epochs = tcfg["epochs"]

    # Same seed -> same head initialisation and same split for A, B and C
    set_seed(seed)
    model = build_model(num_classes=21)
    train_loader, test_loader, classes = get_dataloaders(cfg, seed)
    model = model.to(device)
    set_regime(model, regime)
    total_params, trainable_params = count_params(model)

    optimizer = make_optimizer(model, tcfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # eta_min = 0
    criterion = nn.CrossEntropyLoss()

    print(f"=== Regime {regime} | seed {seed} | trainable {trainable_params:,} "
          f"/ {total_params:,} params ===")
    history = []
    for epoch in range(1, epochs + 1):
        lr_head = optimizer.param_groups[0]["lr"]
        _sync(device)
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        _sync(device)
        epoch_time = time.time() - t0
        scheduler.step()

        # Test accuracy is logged for learning curves only; the reported
        # result is always the LAST epoch (no model selection on the test set).
        test_acc, test_loss, y_true, y_pred = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch, "lr_head": lr_head,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "epoch_time_s": epoch_time,
        })
        print(f"[{regime} s{seed}] ep {epoch:02d}/{epochs} "
              f"loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"test acc {test_acc:.4f} | {epoch_time:.1f}s")

    result = {
        "regime": regime, "seed": seed, "epochs": epochs,
        "classes": classes,
        "total_params": total_params, "trainable_params": trainable_params,
        "final_test_acc": history[-1]["test_acc"],
        "mean_epoch_time_s": sum(h["epoch_time_s"] for h in history) / len(history),
        "history": history,
        "y_true": y_true, "y_pred": y_pred,   # used in Phase 3
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = out_file.with_suffix(".tmp")           # atomic write: no half-written JSON
    tmp.write_text(json.dumps(result))
    os.replace(tmp, out_file)

    if tcfg.get("save_checkpoint", False):
        torch.save(model.state_dict(), runs_dir / f"{regime}_seed{seed}.pth")
    print(f"Saved {out_file}")
    return result
