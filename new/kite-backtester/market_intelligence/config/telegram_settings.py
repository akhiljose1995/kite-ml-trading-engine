import os
from .base_config import BaseConfig


class TelegramSettings(BaseConfig):
    def _validate(self) -> None:
        if "telegram" not in self._raw:
            raise ValueError("Missing 'telegram' section")

    def _load(self) -> None:
        cfg = self._raw["telegram"]

        self.enabled = cfg.get("enabled", False)
        self.bot_token = self._resolve_env(cfg.get("bot_token"))
        self.chat_id = self._resolve_env(cfg.get("chat_id"))
        self.parse_mode = cfg.get("parse_mode", "Markdown")
        self.rate_limit_sec = float(cfg.get("rate_limit_sec", 1.0))

    @staticmethod
    def _resolve_env(value: str) -> str:
        if value and value.startswith("<ENV:"):
            env_key = value.replace("<ENV:", "").replace(">", "")
            return os.getenv(env_key)
        return value