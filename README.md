# Flood Damage Prediction — Kanto Region, Japan

Comparing five regression models on per-capita flood damage across 28 years of
Japanese flood records (1993–2020), with time-aware validation throughout.

Accompanies the manuscript *"A Head-to-Head Study of Ensemble and Deep Learning
Algorithms for Flood Damage Prediction in Japan"* (under revision, ICDPN 2026).

---

## Results

Chronological holdout — trained on 1993–2012, tested on 2013–2020 (1174 events):

| Model             |  RMSE           |  MAE            |  R²             |
|-------------------|-----------------|-----------------|-----------------|
| XGBoost           | **0.802**       | **0.632**       | **0.372**       |
| Random Forest     | 0.806           | 0.633           | 0.365           |
| Deep Neural Net   | 0.837 ± 0.002   | 0.655 ± 0.002   | 0.316 ± 0.003   |
| SVR               | 0.863           | 0.684           | 0.273           |
| Linear Regression | 0.924           | 0.734           | 0.168           |

Rolling-origin validation — five expanding windows, mean ± std:

| Model             |  RMSE           |  R²             |
|-------------------|-----------------|-----------------|
| Random Forest     | **0.676 ± 0.099** | **0.406 ± 0.059** |
| XGBoost           | 0.678 ± 0.097   | 0.403 ± 0.057   |
| Deep Neural Net   | 0.707 ± 0.099   | 0.349 ± 0.056   |
| SVR               | 0.743 ± 0.087   | 0.281 ± 0.043   |
| Linear Regression | 0.807 ± 0.092   | 0.151 ± 0.050   |

The DNN row is averaged over five random seeds; the other models are
deterministic single fits.

**XGBoost and Random Forest are statistically indistinguishable.** The gap
between them is 0.004 RMSE on the holdout and 0.002 the other way under
rolling-origin, against a window-to-window standard deviation of 0.099. The
ordering flips with the protocol, so neither should be claimed as the winner.

---

## Quick start

```bash
git clone https://github.com/Parth-KG/Flood-Prediction.git
cd Flood-Prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd src
python main.py
```

Runs end to end in roughly 30–60 minutes on a laptop. The dataset is included,
so no download step is needed.

To capture the full log (hyperparameters, feature selection, per-seed results):

```bash
python main.py 2>&1 | tee run_log.txt
```

---

## Repository layout

```
├── src/
│   ├── config.py              paths, column groups, seeds, split ratio
│   ├── preprocessing.py       load, chronological sort, split, scale
│   ├── feature_selection.py   VIF, RF importance, mutual information, consensus
│   ├── models.py              the five models + hyperparameter search
│   ├── evaluation.py          metrics table and figures
│   ├── rolling_evaluation.py  expanding-window validation
│   └── main.py                pipeline entry point
├── dataset/
│   ├── data/                  source CSVs (Zenodo, CC-BY 4.0)
│   ├── data_structures.xlsx   variable dictionary
│   └── readme.docx            dataset documentation
├── assets/                    figures and results used in the paper
├── requirements.txt
└── README.md
```

---

## Data

Sourced from the Zenodo repository accompanying Wakai et al., *"Historical
precipitation and flood damage in Japan: functional data analysis and evaluation
of models"* — <https://doi.org/10.5281/zenodo.14015790>, CC-BY 4.0.

This analysis uses `DB_input+res_ptn02d14_logDmgPop.csv`:

| | |
|---|---|
| Events | 5704 |
| Water systems | 75 |
| Rivers | 962 |
| Period | 1993–2020 |
| Missing values | none |
| Duplicate rows | none |

**Geographic coverage is the Kanto region, not all of Japan.** The water system
codes resolve to three prefectures — Ibaraki (08), Chiba (12), Kanagawa (14) —
plus seven nationally managed Class-A river systems. The source study describes
its scope as the Kanto and Koshin regions. Results should not be assumed to
transfer to other parts of Japan.

**Target variable:** `log₁₀(damageObs / population)` — observed monetary flood
damage per resident, log-transformed. Per-capita rather than absolute damage,
so that small basins with high per-resident losses remain visible.

Three columns are dropped before modelling: `log₁₀(dmgPred/pop)` (predictions
from the source study's own model), `damageObs` (the target's numerator — a
direct leak), and `date` (used only for chronological sorting).

---

## Methodology notes

Three choices are worth knowing about before reading the code.

**Chronological splitting.** The train/test boundary is the 80th percentile of
the year distribution, which falls at **2013** — 4530 training events (1993–2012)
against 1174 test events (2013–2020), a 79.4 / 20.6 split. Rows are sorted by
date at load time, because both `TimeSeriesSplit` and Keras' `validation_split`
slice by row position and are only meaningful on ordered data.

**`year` is excluded as a predictor.** It is retained in the dataframe for
splitting and window labelling, but never reaches a model. Under a chronological
split every test-set year lies outside the training range: trees cannot
extrapolate past their final split point, and scaled year values push SVR and
DNN inputs beyond the domain they were fitted on. Removing it is the single
largest contributor to the DNN's performance.

**Consensus feature selection.** A feature is retained only if it ranks in the
top half by *both* Random Forest importance and mutual information — 9 features
qualify. Basin characteristics and river codes are always kept, giving 11:

```
2d, 3d, 11d, 17d, 22d, 27d, area, population, slope, wsysCd, rivCd
```

The two measures disagree substantially — RF ranks population, area, slope while
mutual information ranks population, slope, area — which is what makes requiring
agreement more informative than either ranking alone.

Scaling (`MinMaxScaler`) is fitted on training rows only, and refitted inside
each rolling-origin window. Hyperparameters for Random Forest, XGBoost and SVR
come from a 30-iteration `RandomizedSearchCV` using five-fold `TimeSeriesSplit`.

---

## Outputs

Running `main.py` writes to the working directory:

| File | Contents |
|---|---|
| `model_comparison_summary.csv` | metrics for all five models |
| `results_rolling_origin.csv` | rolling-origin means and standard deviations |
| `feature_importance_rf.png` | Random Forest feature importances |
| `rainfall_correlation_heatmap.png` | Spearman correlation, antecedent rainfall |
| `predicted_vs_observed.png` | scatter plots, all five models |
| `residual_plots.png` | residuals, all five models |
| `model_comparison_metrics.png` | metric bar chart |
| `dnn_training_history.png` | DNN loss and MAE curves |

All figures render at 300 dpi. Copies of the versions used in the manuscript are
in `assets/`.

---

## Reproducibility

Seeded at 42 throughout (`config.RANDOM_SEED`), with the DNN additionally run
across seeds 42–46 and reported as mean ± standard deviation.

Two details matter for exact reproduction. Feature selection returns a `sorted()`
list rather than an unordered set, because XGBoost's `colsample_bytree` samples
column *indices* — a different column order gives different results from the same
seed. And estimators run with `n_jobs=1` inside the parallel search, since nesting
XGBoost's threads under `RandomizedSearchCV(n_jobs=-1)` introduces run-to-run
variation.

---

## Citation

If you use this code, please cite both the software and the underlying dataset:

```bibtex
@software{goswami_flood_prediction,
  author = {Goswami, Parth Krishan and Sen, Aarushi and
            Trivedi, Soahum and Arora, Jyoti},
  title  = {Flood Damage Prediction for the Kanto Region, Japan},
  year   = {2026},
  url    = {https://github.com/Parth-KG/Flood-Prediction}
}

@dataset{wakai_2025,
  author    = {Wakai, A.},
  title     = {Historical precipitation and flood damage in Japan:
               functional data analysis and evaluation of models},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.14015790}
}
```

---

## License

Code is released under the MIT License (see `LICENSE`).

The dataset in `dataset/` is redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and remains the work
of Wakai et al. (2025); attribution as above.

---

## Authors

Parth Krishan Goswami, Aarushi Sen, Soahum Trivedi, Jyoti Arora
Maharaja Surajmal Institute of Technology, GGSIPU, New Delhi, India
