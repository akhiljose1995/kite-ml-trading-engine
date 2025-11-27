# models/training/label_mapper.py
import json
from param_config import CLASS_MAPPING

def save_label_mapping(output_path: str):
    with open(output_path, "w") as f:
        json.dump(CLASS_MAPPING, f, indent=2)

def load_label_mapping(path: str):
    with open(path, "r") as f:
        return json.load(f)
