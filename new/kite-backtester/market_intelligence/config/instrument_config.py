from typing import Dict
from .base_config import BaseConfig


class InstrumentConfig(BaseConfig):
    def _validate(self) -> None:
        if "instruments" not in self._raw:
            raise ValueError("Missing 'instruments' section")

    def _load(self) -> None:
        self.instruments: Dict[str, Dict] = self._raw["instruments"]

    def get(self, key: str) -> Dict:
        if key not in self.instruments:
            raise KeyError(f"Instrument not found: {key}")
        return self.instruments[key]