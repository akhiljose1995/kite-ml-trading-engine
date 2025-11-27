# models/training/param_config.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Class mapping used during training (stable)
CLASS_MAPPING = {
    "down": 0,
    "sideways": 1,
    "up": 2,
    "fake": 3
}

# Models to evaluate
MODEL_REGISTRY = {
    "logistic": LogisticRegression(max_iter=2000, n_jobs=-1),
    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(objective="multi:softprob", num_class=4, use_label_encoder=False, eval_metric="mlogloss")
}

# Hyperparameter grids for RandomizedSearchCV
PARAM_GRID = {
    "logistic": {
        # Logistic has few hyperparams; use C and penalty
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["saga"]
    },
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 6, 10, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2]
    }
}

# RandomizedSearchCV defaults
RANDOM_SEARCH_N_ITER = 30
RANDOM_SEARCH_CV = 3
RANDOM_SEARCH_SCORING = "f1_weighted"
RANDOM_SEARCH_N_JOBS = -1
