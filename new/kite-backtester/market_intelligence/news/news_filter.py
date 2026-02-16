from typing import List, Dict
from datetime import timedelta, datetime
from datetime import timezone


class NewsFilter:
    """
    Filters news items based on instrument relevance and recency.
    """

    def __init__(self, max_age_hours: int = 24):
        self.max_age = timedelta(hours=max_age_hours)

    def filter(
        self,
        *,
        news: List[Dict],
        instrument_ctx,
        now: datetime,
    ) -> List[Dict]:

        filtered = []

        for item in news:
            if not self._is_recent(item, now):
                continue

            if self._is_relevant(item, instrument_ctx):
                filtered.append(item)

        return filtered

    def _is_recent(self, item: Dict, now: datetime) -> bool:
        published_at = item.get("published_at")
        if not published_at:
            return False

        # Normalize timezone (defensive)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        return (now - published_at) <= self.max_age

    def _is_relevant(self, item: Dict, instrument_ctx) -> bool:
        title = (item.get("title") or "").lower()
        tags = [t.lower() for t in (item.get("tags") or [])]
        region = item.get("region")

        if isinstance(region, list):
            region = region[0] if region else ""
        elif region is None:
            region = ""

        region = region.lower()

        # Forex → global macro
        if instrument_ctx.is_forex():
            return True

        # Index → country + index mention
        if instrument_ctx.is_index():
            if instrument_ctx.symbol.lower() in title:
                return True
            return region in {"in", "us", "global"}

        # Stock → company-specific
        if instrument_ctx.is_stock():
            return instrument_ctx.symbol.lower() in title

        return False