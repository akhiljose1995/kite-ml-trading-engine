# models/training/model_selector.py
import os
import joblib
import json
import time
from typing import Dict, Tuple
from sklearn.model_selection import RandomizedSearchCV
from param_config import MODEL_REGISTRY, PARAM_GRID, RANDOM_SEARCH_N_ITER, RANDOM_SEARCH_CV, RANDOM_SEARCH_SCORING, RANDOM_SEARCH_N_JOBS
from evaluator import evaluate_model
import numpy as np

def tune_and_train(model_name: str, model, param_dist: dict, X_train, y_train, n_iter: int = RANDOM_SEARCH_N_ITER):
    """
    Runs RandomizedSearchCV for the provided model and param distribution.
    Returns the best estimator.
    """
    if not param_dist:
        # No hyperparams provided -> return model fitted directly
        model.fit(X_train, y_train)
        return model, {}

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=RANDOM_SEARCH_SCORING,
        cv=RANDOM_SEARCH_CV,
        verbose=1,
        n_jobs=RANDOM_SEARCH_N_JOBS,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def run_auto_ml(X_train, y_train, X_test, y_test, output_dir: str, n_iter: int = RANDOM_SEARCH_N_ITER) -> Dict:
    """
    Trains and evaluates all models defined in MODEL_REGISTRY.
    Saves all model files and metrics. Returns metadata with best model info.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    start_time = time.time()

    for name, model in MODEL_REGISTRY.items():
        print(f"\n===== Training {name} =====")
        param_dist = PARAM_GRID.get(name, {})
        best_estimator, best_params = tune_and_train(name, model, param_dist, X_train, y_train, n_iter=n_iter)
        # Evaluate on test
        metrics = evaluate_model(best_estimator, X_test, y_test)
        results[name] = {
            "best_params": best_params,
            "metrics": metrics
        }
        # Save model artifact
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(best_estimator, model_path)
        results[name]["model_path"] = model_path
        print(f"Saved {name} model to {model_path}")

    # Determine best model by weighted_f1
    best_name = None
    best_score = -np.inf
    for name, r in results.items():
        score = r["metrics"]["weighted_f1"]
        if score > best_score:
            best_score = score
            best_name = name

    # Save meta summary
    metadata = {
        "best_model": best_name,
        "best_score": float(best_score),
        "results": results,
        "time_seconds": time.time() - start_time
    }
    meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nTraining completed. Best model: {best_name} (weighted_f1={best_score:.4f})")
    return metadata
