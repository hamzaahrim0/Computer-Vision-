"""Run all experiments: 3 regimes x 5 seeds (resumable).

    python run_all.py --smoke          # 1 epoch, seed 0, A/B/C (quick test)
    python run_all.py                  # full study (15 runs)
    python run_all.py --summary        # only print the summary table
    python run_all.py --regimes C --seeds 0 1
"""
import argparse
import json
import statistics as st
from pathlib import Path

from src.utils import get_device, load_config


def summarize(runs_dir):
    runs = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*.json"))]
    if not runs:
        print("No finished runs found in", runs_dir)
        return
    print(f"\n{'Regime':<8}{'n':>3}{'OA (%)':>18}{'trainable':>14}{'s/epoch':>10}")
    for regime in sorted({r["regime"] for r in runs}):
        rs = [r for r in runs if r["regime"] == regime]
        accs = [100 * r["final_test_acc"] for r in rs]
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        t = st.mean(r["mean_epoch_time_s"] for r in rs)
        print(f"{regime:<8}{len(rs):>3}{st.mean(accs):>12.2f} ± {sd:<4.2f}"
              f"{rs[0]['trainable_params']:>14,}{t:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch, seed 0 (separate folder)")
    ap.add_argument("--summary", action="store_true", help="only print the summary")
    ap.add_argument("--regimes", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    runs_dir = results_dir / ("runs_smoke" if args.smoke else "runs")

    if args.summary:
        summarize(runs_dir)
        return

    regimes = args.regimes or cfg["experiment"]["regimes"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    if args.smoke:
        cfg["train"]["epochs"] = 1
        seeds = [0]

    from src.train import run_experiment  # imported here so --summary needs no torch

    device = get_device()
    print("Device:", device, "| regimes:", regimes, "| seeds:", seeds,
          "| epochs:", cfg["train"]["epochs"])
    # seed-major order: complete seeds first, so partial results stay balanced
    for seed in seeds:
        for regime in regimes:
            run_experiment(cfg, regime, seed, device, runs_dir)
    summarize(runs_dir)


if __name__ == "__main__":
    main()
