"""
preprocessing.py — data loading, cleaning, encoding, train/test split, scaling.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from config import (
    DATA_PATH, TARGET, CATEGORICAL_COLS, DROP_COLS, TRAIN_SPLIT_QUANTILE
)


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV, drop unused columns, drop rows with missing target."""
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df.dropna(subset=[TARGET], inplace=True)
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns in-place."""
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Chronological train/test split then MinMax scaling.

    Returns
    -------
    X_train_s, X_test_s : scaled DataFrames
    y_train, y_test     : numpy arrays
    feature_cols        : list of feature column names
    """
    split_year = int(df["year"].quantile(TRAIN_SPLIT_QUANTILE))
    print(f"Splitting at year {split_year}")

    train_df = df[df["year"] < split_year].copy()
    test_df  = df[df["year"] >= split_year].copy()

    feature_cols = [c for c in df.columns if c != TARGET]

    X_train_raw = train_df[feature_cols]
    y_train     = train_df[TARGET].values
    X_test_raw  = test_df[feature_cols]
    y_test      = test_df[TARGET].values

    scaler    = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_cols)
    X_test_s  = pd.DataFrame(scaler.transform(X_test_raw),      columns=feature_cols)

    return X_train_s, X_test_s, y_train, y_test, feature_cols
