from .base_config import BaseConfig


class LLMSettings(BaseConfig):
    def _validate(self) -> None:
        if "llm" not in self._raw:
            raise ValueError("Missing 'llm' section")

    def _load(self) -> None:
        cfg = self._raw["llm"]

        self.enabled = cfg.get("enabled", False)
        self.model = cfg.get("model")
        self.temperature = cfg.get("temperature", 0.2)
        self.max_tokens = cfg.get("max_tokens", 800)

        self.history_enabled = cfg.get("history", {}).get("enabled", True)
        self.max_turns = cfg.get("history", {}).get("max_turns", 10)

        self.cost_guard = cfg.get("cost_guard", {})