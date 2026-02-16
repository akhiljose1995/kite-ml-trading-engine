from datetime import datetime
from zoneinfo import ZoneInfo


class TimeContext:
    """
    Provides timezone-aware current datetime utilities.
    """

    def __init__(self, timezone: str):
        self.timezone = ZoneInfo(timezone)
        self.now = datetime.now(tz=self.timezone)

    @property
    def date(self):
        return self.now.date()

    @property
    def time(self):
        return self.now.time()

    def is_same_day(self, other_dt: datetime) -> bool:
        return self.date == other_dt.date()
    
    def get_now(self) -> datetime:
        self.now = datetime.now(tz=self.timezone)
        return self.now