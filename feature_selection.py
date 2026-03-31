import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import RAINFALL_COLS, STATIC_COLS, CATEGORICAL_COLS, RANDOM_SEED


def compute_vif(X_train_s: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF for all rainfall features and print results."""
    rainfall_cols = [c for c in RAINFALL_COLS if c in X_train_s.columns]

    vif_df = pd.DataFrame({
        "feature": rainfall_cols,
        "VIF": [
            variance_inflation_factor(X_train_s[rainfall_cols].values, i)
            for i in range(len(rainfall_cols))
        ],
    }).sort_values("VIF", ascending=False)

    print("\nVIF scores (rainfall features):")
    print(vif_df.to_string(index=False))
    return vif_df


def rf_feature_importance(
    X_train_s: pd.DataFrame,
    y_train,
    feature_cols: list,
    save_path: str = "feature_importance_rf.png",
) -> pd.Series:
    """Fit a Random Forest and return feature importances as a sorted Series."""
    rf_sel = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_sel.fit(X_train_s, y_train)

    importances = (
        pd.Series(rf_sel.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    cumulative   = importances.cumsum()
    top_features = importances[cumulative <= 0.95].index.tolist()
    print(f"\nTop features by RF importance (95% threshold): {len(top_features)}")
    print(importances.head(20))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return importances


def mutual_info_scores(
    X_train_s: pd.DataFrame,
    y_train,
    feature_cols: list,
) -> pd.Series:
    """Compute mutual information scores and print top 20."""
    mi_scores = mutual_info_regression(X_train_s, y_train, random_state=RANDOM_SEED)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    print("\nMutual Information scores (top 20):")
    print(mi_series.head(20))
    return mi_series


def select_features(
    X_train_s: pd.DataFrame,
    X_test_s: pd.DataFrame,
    y_train,
    feature_cols: list,
):
    """
    Full feature selection pipeline.

    Returns
    -------
    X_train, X_test : numpy arrays with the selected features
    selected        : list of selected feature names
    """
    compute_vif(X_train_s)

    importances = rf_feature_importance(X_train_s[feature_cols], y_train, feature_cols)
    mi_series   = mutual_info_scores(X_train_s[feature_cols], y_train, feature_cols)

    # union of top-50% from RF and MI
    rf_top   = set(importances.head(len(importances) // 2).index)
    mi_top   = set(mi_series.head(len(mi_series) // 2).index)
    selected = list(rf_top.union(mi_top))

    # always keep static and categorical features
    for col in STATIC_COLS + CATEGORICAL_COLS:
        if col in feature_cols and col not in selected:
            selected.append(col)

    print(f"\nFinal feature set: {len(selected)} features")

    # rainfall correlation heatmap
    rainfall_cols = [c for c in RAINFALL_COLS if c in X_train_s.columns]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        X_train_s[rainfall_cols].corr(method="spearman"),
        cmap="RdYlGn_r", center=0, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Spearman Correlation - Antecedent Rainfall Features")
    plt.tight_layout()
    plt.savefig("rainfall_correlation_heatmap.png", dpi=150)
    plt.close()

    X_train = X_train_s[selected].values
    X_test  = X_test_s[selected].values
    return X_train, X_test, selected
