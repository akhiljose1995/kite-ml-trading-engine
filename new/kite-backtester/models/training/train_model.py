# models/training/train_model.py
import os
import argparse
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
from sklearn.utils import compute_class_weight
from model_selector import run_auto_ml
from label_mapper import save_label_mapping
from evaluator import evaluate_model, save_metrics_json
from param_config import CLASS_MAPPING
import numpy as np

# Setup directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(logfile_path: str):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    # file handler
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def main(x_train_path, x_test_path, y_train_path, y_test_path, n_iter, output_dir):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOGS_DIR, f"training_{ts}.log")
    logger = setup_logger(logfile)
    logger.info("Starting training run")
    # Also record summary to a metadata json
    run_output_dir = os.path.join(SAVED_MODELS_DIR, f"run_{ts}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Load data
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    # Map labels using CLASS_MAPPING
    y_train = y_train.map(CLASS_MAPPING)
    y_test = y_test.map(CLASS_MAPPING)

    # Save feature order and sample
    feature_list = X_train.columns.tolist()
    with open(os.path.join(run_output_dir, "feature_list.json"), "w") as f:
        json.dump(feature_list, f, indent=2)

    # Save label mapping
    save_label_mapping(os.path.join(run_output_dir, "class_mapping.json"))

    # Optionally log class distribution
    logger.info("Class distribution (train): " + str(y_train.value_counts().to_dict()))
    logger.info("Feature count: %d", len(feature_list))

    # Run AutoML (trains and saves candidate models)
    metadata = run_auto_ml(X_train, y_train, X_test, y_test, output_dir=run_output_dir, n_iter=n_iter)

    # Load best model artifact and evaluate on test
    best_name = metadata["best_model"]
    best_model_path = metadata["results"][best_name]["model_path"]
    best_model = joblib.load(best_model_path)
    metrics = evaluate_model(best_model, X_test, y_test)
    # Save metrics JSON into run dir
    save_metrics_json(metrics, os.path.join(run_output_dir, f"{best_name}_metrics.json"))

    # Save final chosen model to a stable path (symlink-like)
    final_model_path = os.path.join(SAVED_MODELS_DIR, "best_price_dir_model.pkl")
    joblib.dump(best_model, final_model_path)

    # Save metadata summarizing run
    summary = {
        "timestamp": ts,
        "run_output_dir": run_output_dir,
        "best_model_name": best_name,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "training_metadata": metadata
    }
    with open(os.path.join(run_output_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write a final log summary
    logger.info(f"Training finished. Best model: {best_name}. Saved canonical model at {final_model_path}")
    logger.info("Confusion Matrix for best model:\n%s", str(metrics["confusion_matrix"]))

    # PRINT only final model name and confusion matrix to stdout
    print("\n=== TRAINING RESULT ===")
    print("Best model:", best_name)
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(pd.DataFrame(metrics["confusion_matrix"]))
    print("=======================\n")
    logger.info("Run completed successfully.")

if __name__ == "__main__":
    """
    From repo root:

    python models/training/train_model.py \
    --x_train data/x_train.csv \
    --x_test data/x_test.csv \
    --y_train data/y_train.csv \
    --y_test data/y_test.csv \
    --n_iter 30
    """
    parser = argparse.ArgumentParser(description="AutoML Training (Logistic, RF, XGBoost) and selection")
    parser.add_argument("--x_train", type=str, default="data/x_train.csv")
    parser.add_argument("--x_test", type=str, default="data/x_test.csv")
    parser.add_argument("--y_train", type=str, default="data/y_train.csv")
    parser.add_argument("--y_test", type=str, default="data/y_test.csv")
    parser.add_argument("--n_iter", type=int, default=30, help="RandomizedSearchCV n_iter")
    parser.add_argument("--output_dir", type=str, default=os.path.join(SAVED_MODELS_DIR, "runs"))
    args = parser.parse_args()
    main(args.x_train, args.x_test, args.y_train, args.y_test, args.n_iter, args.output_dir)
