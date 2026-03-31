"""
models.py — training for all five models:
            Linear Regression, Random Forest, XGBoost, SVR, DNN.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam            # type: ignore

from config import RANDOM_SEED

# ---- helpers ----------------------------------------------------------------

def _metrics(y_true, y_pred) -> dict:
    return {
        "preds": y_pred,
        "rmse":  np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":   mean_absolute_error(y_true, y_pred),
        "r2":    r2_score(y_true, y_pred),
    }


def _tscv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


# ---- individual model trainers ----------------------------------------------

def train_linear_regression(X_train, y_train, X_test, y_test) -> dict:
    print("\nTraining: Linear Regression...")
    lr    = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    res   = _metrics(y_test, preds)
    print(f"  RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}")
    return res


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    print("\nTraining: Random Forest...")
    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=30, cv=_tscv(),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    preds = search.best_estimator_.predict(X_test)
    res   = _metrics(y_test, preds)
    print(f"  RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}")
    return res


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    print("\nTraining: XGBoost...")
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 0.5, 1.0],
        "reg_lambda":       [1, 1.5, 2.0],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED, verbosity=0),
        param_distributions=param_dist,
        n_iter=30, cv=_tscv(),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    preds = search.best_estimator_.predict(X_test)
    res   = _metrics(y_test, preds)
    print(f"  RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}")
    return res


def train_svr(X_train, y_train, X_test, y_test) -> dict:
    print("\nTraining: SVR...")
    param_dist = {
        "C":       [0.1, 1, 10, 100, 500],
        "gamma":   ["scale", "auto", 0.001, 0.01, 0.1],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "kernel":  ["rbf"],
    }
    search = RandomizedSearchCV(
        SVR(),
        param_distributions=param_dist,
        n_iter=30, cv=_tscv(),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    preds = search.best_estimator_.predict(X_test)
    res   = _metrics(y_test, preds)
    print(f"  RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}")
    return res


def train_dnn(
    X_train, y_train, X_test, y_test,
    save_history_plot: str = "dnn_training_history.png",
) -> dict:
    print("\nTraining: DNN...")
    n_features = X_train.shape[1]

    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=200,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
        verbose=1,
    )

    # training history plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history.history["mae"],      label="Train")
    axes[1].plot(history.history["val_mae"],  label="Validation")
    axes[1].set_title("MAE"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    plt.suptitle("DNN Training History")
    plt.tight_layout()
    plt.savefig(save_history_plot, dpi=150)
    plt.close()

    preds = model.predict(X_test, verbose=0).flatten()
    res   = _metrics(y_test, preds)
    print(f"  RMSE: {res['rmse']:.4f} | R2: {res['r2']:.4f}")
    return res


# ---- main entry point -------------------------------------------------------

def train_all(X_train, y_train, X_test, y_test) -> dict:
    """
    Train all five models and return a results dict keyed by model name.
    Each value is a dict with keys: preds, rmse, mae, r2.
    """
    return {
        "Linear Regression": train_linear_regression(X_train, y_train, X_test, y_test),
        "Random Forest":     train_random_forest(X_train, y_train, X_test, y_test),
        "XGBoost":           train_xgboost(X_train, y_train, X_test, y_test),
        "SVR":               train_svr(X_train, y_train, X_test, y_test),
        "DNN":               train_dnn(X_train, y_train, X_test, y_test),
    }
