from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseConfig(ABC):
    def __init__(self, raw_config: Dict[str, Any]):
        self._raw = raw_config
        self._validate()
        self._load()

    @abstractmethod
    def _validate(self) -> None:
        pass

    @abstractmethod
    def _load(self) -> None:
        pass