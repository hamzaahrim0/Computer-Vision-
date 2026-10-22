"""Metrics and statistics computed from the saved runs (Phase 3). No torch needed."""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def load_runs(runs_dir):
    """Return {(regime, seed): run_dict} from results/runs/*.json."""
    runs = {}
    for p in sorted(Path(runs_dir).glob("*_seed*.json")):
        r = json.loads(p.read_text())
        runs[(r["regime"], r["seed"])] = r
    return runs


def run_metrics(run):
    """Overall accuracy, per-class P/R/F1 and confusion matrix of one run."""
    labels = list(range(len(run["classes"])))
    yt, yp = np.asarray(run["y_true"]), np.asarray(run["y_pred"])
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, zero_division=0
    )
    return {
        "oa": float((yt == yp).mean()),
        "precision": p, "recall": r, "f1": f1,
        "cm": confusion_matrix(yt, yp, labels=labels),
    }


def class_indices(classes, names):
    missing = [n for n in names if n not in classes]
    if missing:
        raise ValueError(f"Classes not found in the dataset: {missing}")
    return [classes.index(n) for n in names]


def paired_tests(a, b):
    """Paired comparison of two metric vectors (same seeds/splits), a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {
        "n": len(d),
        "mean_diff": float(d.mean()),
        "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "p_ttest": float("nan"),
        "p_wilcoxon": float("nan"),
    }
    if len(d) > 1 and np.ptp(d) > 0:
        out["p_ttest"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:      # all differences are zero
        pass
    return out


def top_confusions(cm_norm, classes, k=5):
    """Largest off-diagonal entries of a row-normalised confusion matrix."""
    pairs = [
        (classes[i], classes[j], float(cm_norm[i, j]))
        for i in range(len(classes)) for j in range(len(classes))
        if i != j and cm_norm[i, j] > 0
    ]
    return sorted(pairs, key=lambda t: -t[2])[:k]
