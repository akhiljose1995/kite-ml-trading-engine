# models/training/evaluator.py
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X, y, label_map=None):
    """
    Returns a dict with accuracy, weighted_f1, classification_report, confusion_matrix.
    """
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    w_f1 = f1_score(y, preds, average="weighted")
    report = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)
    return {
        "accuracy": float(acc),
        "weighted_f1": float(w_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm
    }

def save_metrics_json(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
