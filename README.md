# Urban Scene Classification with GoogLeNet

This project studies transfer learning for aerial image classification on the [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html). It compares three GoogLeNet fine-tuning strategies across five reproducible train/test splits and produces metrics, figures, and LaTeX tables from the saved runs.

| Regime | Trainable layers |
| --- | --- |
| A | Final classification layer only (frozen backbone) |
| B | `inception5b` and the final classification layer |
| C | Entire network |

Each split is stratified, and all three regimes use the same split for a given seed. The default configuration uses seeds `0–4`, an 80/20 split, and 30 training epochs. The final epoch is reported; the test set is not used for model selection.

## Dataset

The UC Merced dataset contains 2,100 aerial images across 21 land-use classes (100 images per class). The data downloader places it at `data/UCMerced_LandUse/Images`, where each class has its own directory.

```bash
python scripts/download_data.py
```

To choose another download destination:

```bash
python scripts/download_data.py --dest /path/to/data
```

The dataset root can also be set to any directory containing the 21 class folders by using `URBAN_DATA_ROOT` (see [Configuration](#configuration)).

## Setup

Use Python 3.9 or newer. Install a PyTorch and torchvision build appropriate for your system by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the CUDA-enabled PyTorch build that matches your machine. The scripts automatically use CUDA when available, otherwise they run on CPU. GoogLeNet's ImageNet weights are downloaded by torchvision the first time a model is built.

## Run the experiments

Run a one-epoch smoke run for all regimes (stored separately from full runs):

```bash
python run_all.py --smoke
```

Run the complete default study (3 regimes × 5 seeds × 30 epochs):

```bash
python run_all.py
```

Runs are resumable: a completed regime/seed pair is skipped when its JSON result already exists. You can run selected regimes and seeds, or print a summary of existing runs:

```bash
python run_all.py --regimes C --seeds 0 1
python run_all.py --summary
```

Full results and shared split files are written under `results/` by default. Smoke-run results go in `results/runs_smoke/`.

## Generate the report

After completing the full experiment, create the Phase 3 report:

```bash
python scripts/make_report.py
```

The script reads `results/runs/` and writes to `results/phase3/`:

- `summary.md` with aggregate metrics and paired comparisons
- `metrics_per_class.csv`
- LaTeX tables in `tables/`
- PDF and PNG figures in `figures/`

To report on another run directory or choose a different output directory:

```bash
python scripts/make_report.py --runs-dir results/runs_smoke --out-dir results/smoke_report
```

The report uses SciPy for paired tests; it is included in `requirements.txt`.

## Configuration

Edit [`configs/config.yaml`](configs/config.yaml) to change image size, batch size, train/test ratio, seeds, regimes, epochs, optimizer settings, and output location. Relative paths are resolved from the project root. These environment variables override the configured paths:

| Variable | Purpose | Default |
| --- | --- | --- |
| `URBAN_DATA_ROOT` | Directory containing the dataset's class folders | `data/UCMerced_LandUse/Images` |
| `URBAN_RESULTS_DIR` | Directory for splits, runs, and report outputs | `results` |

For example:

```bash
URBAN_DATA_ROOT=/datasets/UCMerced_LandUse/Images \
URBAN_RESULTS_DIR=/scratch/urban-results \
python run_all.py
```

## Repository layout

```text
configs/config.yaml       Experiment settings
src/data.py               Image transforms, dataset, and data loaders
src/splits.py             Dataset indexing and reproducible stratified splits
src/model.py              Pretrained GoogLeNet and regimes A/B/C
src/train.py              Training and evaluation for one regime/seed run
src/evaluate.py           Metrics and paired statistical tests
src/utils.py              Configuration, seeding, and device helpers
run_all.py                Experiment runner and run summary
scripts/download_data.py  Dataset downloader
scripts/make_report.py    Phase 3 tables, figures, and summary
scripts/check_phase1.py   Phase 1 checks
scripts/check_phase2.py   Phase 2 checks
notebooks/                Google Colab notebooks
```

## Google Colab

The notebooks provide a hosted workflow for running the project in Colab. Start with the Phase 3 notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzaahrim0/Computer-Vision-/blob/main/notebooks/02_colab_phase3.ipynb)

## Reproducibility notes

- Python, NumPy, and PyTorch random generators are seeded for each run.
- Stratified splits are saved under `results/splits/` and reused across regimes.
- Run JSON files contain per-epoch metrics, predictions, class names, parameter counts, and run metadata. Model checkpoints are optional and disabled by default (`train.save_checkpoint` in the config).
- GPU kernels and differences between software or hardware versions may still affect exact numerical results.
