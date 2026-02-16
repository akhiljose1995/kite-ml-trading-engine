from .base_config import BaseConfig


class MarketSessionConfig(BaseConfig):
    def _validate(self) -> None:
        if "sessions" not in self._raw:
            raise ValueError("Missing 'sessions' section")

    def _load(self) -> None:
        self.sessions = self._raw["sessions"]

    def get_session(self, exchange: str) -> dict:
        if exchange not in self.sessions:
            raise KeyError(f"Exchange session not found: {exchange}")
        return self.sessions[exchange]