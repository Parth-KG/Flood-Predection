import numpy as np
import tensorflow as tf

# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# data
DATA_PATH = "DB_input+res_ptn02d14_logDmgPop.csv"

# columns
TARGET          = "log10(dmgObs/pop)"
CATEGORICAL_COLS = ["wsysCd", "rivCd"]
DROP_COLS        = ["damageObs", "log10(dmgPred/pop)", "date"]
RAINFALL_COLS    = [f"{i}d" for i in range(29, -1, -1)]
STATIC_COLS      = ["area", "slope", "population", "year"]

# train/test split
TRAIN_SPLIT_QUANTILE = 0.80

# model colours used in plots
MODEL_COLORS = {
    "Linear Regression": "gray",
    "Random Forest":     "steelblue",
    "XGBoost":           "darkorange",
    "SVR":               "green",
    "DNN":               "crimson",
}
