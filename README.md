# 🌊 Flood Damage Prediction — Multi-Model ML Pipeline

An end-to-end machine learning pipeline that predicts **population-normalized flood damage** from antecedent rainfall and catchment characteristics. Five models — from Linear Regression to a Deep Neural Network — are trained, tuned, and benchmarked against each other.

---

## 📌 Overview

Flood damage is modelled as `log10(damageObs / population)`, a normalized metric that accounts for exposure. The pipeline ingests a tabular dataset of historical flood events, applies feature selection using Random Forest importance and Mutual Information, then trains and compares five regression models with time-series-aware cross-validation.

All outputs (plots, metrics CSV, model comparisons) are saved automatically on a single `python main.py` run.

---

## 🧠 Models Compared

| Model | Tuning Strategy |
|---|---|
| Linear Regression | None (baseline) |
| Random Forest | RandomizedSearchCV + TimeSeriesSplit |
| XGBoost | RandomizedSearchCV + TimeSeriesSplit |
| SVR (RBF kernel) | RandomizedSearchCV + TimeSeriesSplit |
| DNN (Keras) | EarlyStopping + ReduceLROnPlateau |

All tree and kernel models are tuned over 30 random parameter combinations with 5-fold time-series cross-validation to prevent data leakage from temporal autocorrelation.

---

## 📁 Project Structure

```
├── main.py               # Orchestrates the full pipeline
├── config.py             # All constants: paths, column names, seeds, colours
├── preprocessing.py      # Load, clean, encode, chronological split, MinMax scale
├── feature_selection.py  # VIF check, RF importance, Mutual Information, heatmap
├── models.py             # Training logic for all five models
├── evaluation.py         # Metrics summary, plots, and CSV export
└── data/
    └── DB_input+res_ptn02d14_logDmgPop.csv   # Input dataset (not tracked)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost tensorflow statsmodels matplotlib seaborn
```

### Run the Full Pipeline

```bash
python main.py
```

This runs all five stages in sequence and saves every output to the working directory.

---

## ⚙️ Pipeline Walkthrough

### 1. Preprocessing (`preprocessing.py`)
- Loads the CSV and drops rows with a missing target
- Label-encodes categorical watershed/river codes (`wsysCd`, `rivCd`)
- **Chronological train/test split** at the 80th percentile year — preserving temporal order, no shuffling
- MinMax scales all features (scaler fit on train only, applied to test)

### 2. Feature Selection (`feature_selection.py`)

Three complementary techniques are applied:

**VIF (Variance Inflation Factor)** — checks for multicollinearity among the 30 antecedent rainfall features (`0d` to `29d`).

**Random Forest Importance** — features are ranked by impurity reduction; the cumulative top 95% are identified.

**Mutual Information** — ranks features by non-linear statistical dependence with the target.

The final feature set is the **union of the top 50% from RF and MI**, with static and categorical features always retained.

Outputs saved:
- `feature_importance_rf.png` — horizontal bar chart of top-20 features
- `rainfall_correlation_heatmap.png` — Spearman correlation matrix of all rainfall lag features

### 3. Model Training (`models.py`)

Each model is trained on the selected feature set. Key hyperparameter search spaces:

**Random Forest**
```python
n_estimators: [100, 200, 300, 500]
max_depth:    [None, 10, 20, 30]
max_features: ["sqrt", "log2", 0.5]
```

**XGBoost**
```python
n_estimators:     [100–500]
learning_rate:    [0.01–0.2]
subsample:        [0.6–1.0]
colsample_bytree: [0.6–1.0]
reg_alpha/lambda: L1 + L2 regularization
```

**SVR**
```python
C:       [0.1–500]
gamma:   ["scale", "auto", 0.001–0.1]
epsilon: [0.01–0.5]
kernel:  rbf
```

**DNN Architecture**
```
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
Dense(64,  ReLU) → BatchNorm → Dropout(0.3)
Dense(32,  ReLU) → BatchNorm → Dropout(0.2)
Dense(1,   Linear)
```
Trained with Adam (lr=0.001), EarlyStopping (patience=20), ReduceLROnPlateau (factor=0.5).

Output saved: `dnn_training_history.png`

### 4. Evaluation (`evaluation.py`)

All models are evaluated on the held-out test set using RMSE, MAE, and R².

Outputs saved:
- `model_comparison_summary.csv` — full metrics table
- `predicted_vs_observed.png` — scatter plots for all five models
- `residual_plots.png` — residual vs. predicted plots
- `model_comparison_metrics.png` — side-by-side bar charts for RMSE, MAE, R²

---

## 📊 Output Files Summary

| File | Description |
|---|---|
| `model_comparison_summary.csv` | RMSE / MAE / R² for all models |
| `predicted_vs_observed.png` | Scatter plots (predicted vs. observed) |
| `residual_plots.png` | Residual diagnostics |
| `model_comparison_metrics.png` | Metric bar chart comparison |
| `feature_importance_rf.png` | Top-20 feature importances |
| `rainfall_correlation_heatmap.png` | Spearman correlation of rainfall lags |
| `dnn_training_history.png` | DNN loss and MAE training curves |

---

## ⚙️ Configuration (`config.py`)

All tunable constants live in one place:

```python
DATA_PATH            = "DB_input+res_ptn02d14_logDmgPop.csv"
TARGET               = "log10(dmgObs/pop)"
RAINFALL_COLS        = ["29d", "28d", ..., "0d"]   # 30 antecedent rainfall lags
STATIC_COLS          = ["area", "slope", "population", "year"]
CATEGORICAL_COLS     = ["wsysCd", "rivCd"]
TRAIN_SPLIT_QUANTILE = 0.80
RANDOM_SEED          = 42
```

---

## 💡 Design Decisions

- **Chronological split over random split** — flood events are temporally correlated; random shuffling would leak future data into training.
- **TimeSeriesSplit for CV** — same reasoning applied during hyperparameter search.
- **Log-transformed target** — `log10(damage/population)` compresses the heavy-tailed damage distribution and normalizes for catchment size.
- **Union of RF + MI for feature selection** — neither method alone captures both linear and non-linear dependencies; the union is a robust compromise.

---

## 🗺️ Possible Extensions

- [ ] SHAP values for model interpretability
- [ ] Stacking / ensemble of top-performing models
- [ ] Spatial cross-validation (leave-one-catchment-out)
- [ ] Precipitation forecasts as input (operational forecasting mode)
- [ ] Streamlit dashboard for interactive prediction

---

## 📄 License

MIT License — free to use and adapt with attribution.
