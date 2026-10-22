"""Small helpers: config loading, seeding, device selection."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/config.yaml"):
    """Load the YAML config and apply environment-variable overrides."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if os.environ.get("URBAN_DATA_ROOT"):
        cfg["data"]["root"] = os.environ["URBAN_DATA_ROOT"]
    if os.environ.get("URBAN_RESULTS_DIR"):
        cfg["paths"]["results_dir"] = os.environ["URBAN_RESULTS_DIR"]

    # Make relative paths absolute with respect to the project root
    for section, key in (("data", "root"), ("paths", "results_dir")):
        p = Path(cfg[section][key])
        if not p.is_absolute():
            cfg[section][key] = str(PROJECT_ROOT / p)
    return cfg


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
