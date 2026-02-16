from datetime import time, datetime
from typing import List


class SessionContext:
    """
    Determines market session state for a given exchange.
    """

    def __init__(
        self,
        *,
        exchange: str,
        timezone: str,
        market_open: str,
        market_close: str,
        pre_open_start: str | None = None,
        holidays: List[str] | None = None
    ):
        self.exchange = exchange
        self.timezone = timezone
        self.market_open = self._parse_time(market_open)
        self.market_close = self._parse_time(market_close)
        self.pre_open_start = self._parse_time(pre_open_start) if pre_open_start else None
        self.holidays = set(holidays or [])

    @staticmethod
    def _parse_time(t: str) -> time:
        h, m = map(int, t.split(":"))
        return time(hour=h, minute=m)

    def is_holiday(self, date: datetime.date) -> bool:
        return date.isoformat() in self.holidays

    def is_pre_open(self, now: datetime) -> bool:
        if not self.pre_open_start:
            return False
        return self.pre_open_start <= now.time() < self.market_open

    def is_open(self, now: datetime) -> bool:
        return self.market_open <= now.time() <= self.market_close

    def is_closed(self, now: datetime) -> bool:
        return not self.is_pre_open(now) and not self.is_open(now)

    def session_state(self, now: datetime) -> str:
        if self.is_holiday(now.date()):
            return "HOLIDAY"
        if self.is_pre_open(now):
            return "PRE_OPEN"
        if self.is_open(now):
            return "OPEN"
        return "CLOSED"