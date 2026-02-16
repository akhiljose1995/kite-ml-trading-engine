from .base_config import BaseConfig

class NewsSettings(BaseConfig):
    def _validate(self) -> None:
        if "news" not in self._raw:
            raise ValueError("Missing 'news' section")

    def _load(self) -> None:
        cfg = self._raw["news"]

        self.enabled = cfg.get("enabled", False)
        self.provider = cfg.get("provider")
        self.api_key = cfg.get("api_key")
        self.max_results = cfg.get("max_results", 10)
        self.language = cfg.get("language", "en")
        self.country = cfg.get("country", {})