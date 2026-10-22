# Urban scene classification from aerial imagery with GoogLeNet

Fine-tuning strategies (A: frozen backbone, B: last inception block, C: full)
for GoogLeNet on the UC Merced Land Use dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzaahrim0/Computer-Vision-/blob/main/notebooks/00_colab_phase1.ipynb)

## Structure
```
configs/config.yaml     hyperparameters and paths
src/splits.py           indexing + stratified split (saved per seed)
src/data.py             transforms, Dataset, DataLoaders
src/utils.py            config loading, seeding, device
scripts/download_data.py   downloads UC Merced (2100 images, 21 classes)
scripts/check_phase1.py    Phase 1 sanity checks
notebooks/              Colab notebooks
```

## Quick start (Colab)
Click the badge above, select a GPU runtime and run all cells.

## Quick start (local)
```bash
pip install -r requirements.txt   # + torch/torchvision from pytorch.org
python scripts/download_data.py
python scripts/check_phase1.py
```

## Environment variables
- `URBAN_DATA_ROOT`: folder containing the 21 class folders
- `URBAN_RESULTS_DIR`: where splits, results and figures are saved

## Phases
1. Data and reproducibility (this commit)
2. Model, regimes A/B/C and training
3. Evaluation, aggregation and LaTeX tables
