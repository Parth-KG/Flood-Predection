import warnings
warnings.filterwarnings("ignore")

from preprocessing     import load_and_clean, encode_categoricals, split_and_scale
from feature_selection import select_features
from models            import train_all
from evaluation        import run_evaluation


def main():
    # 1. load & clean
    df = load_and_clean()
    df = encode_categoricals(df)

    # 2. split & scale
    X_train_s, X_test_s, y_train, y_test, feature_cols = split_and_scale(df)

    # 3. feature selection
    X_train, X_test, selected = select_features(
        X_train_s, X_test_s, y_train, feature_cols
    )

    # 4. train all models
    results = train_all(X_train, y_train, X_test, y_test)

    # 5. evaluate & save plots / CSV
    run_evaluation(results, y_test)

    print("\nDone. All outputs saved.")


if __name__ == "__main__":
    main()
