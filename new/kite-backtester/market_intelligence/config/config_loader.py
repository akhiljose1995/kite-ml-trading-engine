import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}