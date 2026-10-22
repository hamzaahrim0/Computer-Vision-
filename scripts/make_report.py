"""Phase 3: aggregate the runs into LaTeX tables, figures and a summary.

    python scripts/make_report.py                  # uses results/runs
    python scripts/make_report.py --runs-dir results/runs_smoke

Outputs (in <results_dir>/phase3/): tables/*.tex, figures/*.pdf|png,
metrics_per_class.csv, summary.md
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (class_indices, load_runs, paired_tests,  # noqa: E402
                          run_metrics, top_confusions)
from src.utils import load_config  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})

REGIME_NAMES = {"A": "A (frozen backbone)", "B": "B (last inception + FC)",
                "C": "C (full fine-tuning)"}
PRETTY = {
    "baseballdiamond": "baseball diamond", "denseresidential": "dense residential",
    "golfcourse": "golf course", "mediumresidential": "medium residential",
    "mobilehomepark": "mobile home park", "parkinglot": "parking lot",
    "sparseresidential": "sparse residential", "storagetanks": "storage tanks",
    "tenniscourt": "tennis court",
}
RESIDENTIAL = ["denseresidential", "mediumresidential", "sparseresidential"]


def pretty(c):
    return PRETTY.get(c, c)


def ms(values, dec=2, scale=100.0):
    v = np.asarray(values, float) * scale
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{dec}f} $\\pm$ {sd:.{dec}f}"


def bold(s):
    return f"\\textbf{{{s}}}"


def fmt_p(p):
    if np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def save_fig(fig, path_no_ext):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(f"{path_no_ext}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def write_table(path, header, rows, align, caption_hint=""):
    lines = [f"% {caption_hint}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion(cmn, classes, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labels = [pretty(c) for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), labels, fontsize=6)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=5, color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Fraction of true class")
    save_fig(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    runs_dir = Path(args.runs_dir) if args.runs_dir else results_dir / "runs"
    out = Path(args.out_dir) if args.out_dir else results_dir / "phase3"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    if not runs:
        sys.exit(f"No runs found in {runs_dir}. Run `python run_all.py` first.")

    regimes = sorted({r for r, _ in runs})
    seeds = sorted(set.intersection(*[{s for r, s in runs if r == reg} for reg in regimes]))
    dropped = sorted({s for _, s in runs} - set(seeds))
    if dropped:
        print(f"WARNING: seeds {dropped} are incomplete and ignored (paired design).")
    if not seeds:
        sys.exit("No seed is complete for all regimes.")
    print(f"Regimes: {regimes} | seeds used: {seeds}")

    classes = runs[(regimes[0], seeds[0])]["classes"]
    urban = cfg["urban_classes"]
    u_idx = class_indices(classes, urban)
    res_idx = class_indices(classes, RESIDENTIAL)

    # ---- per-run metrics ------------------------------------------------
    M = {reg: [run_metrics(runs[(reg, s)]) for s in seeds] for reg in regimes}
    oa = {g: np.array([m["oa"] for m in M[g]]) for g in regimes}
    f1 = {g: np.stack([m["f1"] for m in M[g]]) for g in regimes}          # (seeds, C)
    mf1 = {g: f1[g].mean(1) for g in regimes}
    mf1u = {g: f1[g][:, u_idx].mean(1) for g in regimes}
    cm_sum = {g: sum(m["cm"] for m in M[g]) for g in regimes}
    cmn = {g: cm_sum[g] / cm_sum[g].sum(1, keepdims=True) for g in regimes}
    params = {g: runs[(g, seeds[0])]["trainable_params"] for g in regimes}
    t_ep = {g: np.mean([runs[(g, s)]["mean_epoch_time_s"] for s in seeds]) for g in regimes}

    # ---- Table 1: main results -----------------------------------------
    best = {k: max(regimes, key=lambda g: d[g].mean())
            for k, d in (("oa", oa), ("mf1", mf1), ("mf1u", mf1u))}
    rows = []
    for g in regimes:
        c = [ms(oa[g]), ms(mf1[g]), ms(mf1u[g])]
        for k, i in (("oa", 0), ("mf1", 1), ("mf1u", 2)):
            if best[k] == g:
                c[i] = bold(c[i])
        rows.append([REGIME_NAMES.get(g, g), f"{params[g]:,}".replace(",", "{,}")]
                    + c + [f"{t_ep[g]:.1f}"])
    write_table(out / "tables" / "table_main.tex",
                ["Regime", "Trainable params", "OA (\\%)", "Macro-F1 (\\%)",
                 "Macro-F1$_{\\mathcal{U}}$ (\\%)", "s/epoch"],
                rows, "lrccc r".replace(" ", ""),
                f"mean $\\pm$ std over {len(seeds)} seeds; U = urban subset")

    # ---- Table 2: per-class F1 on urban classes ------------------------
    rows = []
    for name, j in zip(urban, u_idx):
        means = {g: f1[g][:, j].mean() for g in regimes}
        top = max(means, key=means.get)
        cells = [bold(ms(f1[g][:, j], 1)) if g == top else ms(f1[g][:, j], 1)
                 for g in regimes]
        rows.append([pretty(name)] + cells)
    rows.append(["\\midrule Macro-average (U)"] + [ms(mf1u[g], 1) for g in regimes])
    write_table(out / "tables" / "table_urban_f1.tex",
                ["Class"] + [f"Regime {g}" for g in regimes], rows,
                "l" + "c" * len(regimes), "F1-score (\\%) per urban class")

    # ---- Table 3: paired statistical tests -----------------------------
    comps = [("C", "A"), ("C", "B"), ("B", "A")]
    rows, stat_lines = [], []
    for a, b in comps:
        if a not in regimes or b not in regimes:
            continue
        for label, d in (("OA", oa), ("Macro-F1$_{\\mathcal{U}}$", mf1u)):
            t = paired_tests(d[a], d[b])
            rows.append([f"{a} vs {b}", label,
                         f"{100 * t['mean_diff']:+.2f} $\\pm$ {100 * t['std_diff']:.2f}",
                         fmt_p(t["p_ttest"]), fmt_p(t["p_wilcoxon"])])
            stat_lines.append(f"- {a} vs {b} | {'Macro-F1(U)' if 'F1' in label else label}: diff {100 * t['mean_diff']:+.2f} pts, "
                              f"t-test p={t['p_ttest']:.4f}, Wilcoxon p={t['p_wilcoxon']:.4f}")
    write_table(out / "tables" / "table_stats.tex",
                ["Comparison", "Metric", "Mean diff. (pts)", "$p$ (paired $t$)",
                 "$p$ (Wilcoxon)"], rows, "llccc",
                f"paired over {len(seeds)} seeds; with n=5 the smallest possible "
                "two-sided Wilcoxon p-value is 0.0625")

    # ---- Table 4: top confusions of the best regime --------------------
    g_best = best["oa"]
    top = top_confusions(cmn[g_best], classes, k=5)
    write_table(out / "tables" / "table_top_confusions.tex",
                ["True class", "Predicted as", "Rate (\\%)"],
                [[pretty(t), pretty(p), f"{100 * v:.1f}"] for t, p, v in top],
                "llc", f"top confusions of regime {g_best} (summed over seeds)")

    # ---- Figures --------------------------------------------------------
    for g in regimes:
        plot_confusion(cmn[g], classes, out / "figures" / f"fig_confusion_{g}",
                       f"Regime {g}: normalised confusion matrix (%)")

    fig, axes = plt.subplots(1, len(regimes), figsize=(2.3 * len(regimes) + 0.6, 2.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    cols = ["dense", "medium", "sparse", "other"]
    for ax, g in zip(axes, regimes):
        cm = cm_sum[g]
        sub = np.zeros((3, 4))
        for a, i in enumerate(res_idx):
            sub[a, :3] = cm[i, res_idx]
            sub[a, 3] = cm[i].sum() - cm[i, res_idx].sum()
            sub[a] /= cm[i].sum()
        ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(4):
                ax.text(b, a, f"{100 * sub[a, b]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if sub[a, b] > 0.5 else "black")
        ax.set_xticks(range(4), cols, fontsize=7)
        ax.set_yticks(range(3), cols[:3], fontsize=7)
        ax.set_title(f"Regime {g}")
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.tight_layout()
    save_fig(fig, out / "figures" / "fig_residential")

    def curve(key, g):
        h = [runs[(g, s)]["history"] for s in seeds]
        n = min(len(x) for x in h)
        return np.array([[e[key] for e in x[:n]] for x in h])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for g in regimes:
        for ax, key, scale in ((a1, "train_loss", 1.0), (a2, "test_acc", 100.0)):
            v = curve(key, g) * scale
            x = np.arange(1, v.shape[1] + 1)
            mu, sd = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else 0 * v[0])
            ax.plot(x, mu, label=f"Regime {g}")
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Training loss")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Test accuracy (%)")
    a2.legend(frameon=False)
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out / "figures" / "fig_learning_curves")

    # ---- CSV + summary --------------------------------------------------
    rows = []
    for j, c in enumerate(classes):
        row = {"class": c, "urban": c in urban}
        for g in regimes:
            row[f"F1_{g}_mean"] = 100 * f1[g][:, j].mean()
            row[f"F1_{g}_std"] = 100 * (f1[g][:, j].std(ddof=1) if len(seeds) > 1 else 0)
        rows.append(row)
    pd.DataFrame(rows).round(2).to_csv(out / "metrics_per_class.csv", index=False)

    L = [f"# Phase 3 summary ({len(seeds)} seeds: {seeds})", ""]
    for g in regimes:
        L.append(f"- Regime {g}: OA {100 * oa[g].mean():.2f} ± {100 * oa[g].std(ddof=1):.2f} | "
                 f"macro-F1 {100 * mf1[g].mean():.2f} | macro-F1(U) {100 * mf1u[g].mean():.2f} | "
                 f"{params[g]:,} params | {t_ep[g]:.1f} s/epoch")
    L += ["", "## Paired tests"] + stat_lines
    L += ["", f"## Top confusions, regime {g_best} (summed over seeds)"]
    L += [f"- {pretty(t)} -> {pretty(p)}: {100 * v:.1f}%" for t, p, v in top]
    worst = np.argsort(f1[g_best].mean(0))[:5]
    L += ["", f"## Lowest-F1 classes, regime {g_best}"]
    L += [f"- {pretty(classes[j])}: {100 * f1[g_best][:, j].mean():.1f}%" for j in worst]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n".join(L))
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
