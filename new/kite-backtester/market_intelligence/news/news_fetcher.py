import os
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
from sympy import pprint


class NewsFetcher:
    """
    Base class for news fetchers.
    """

    def fetch(self, **kwargs) -> List[Dict]:
        raise NotImplementedError


class NewsDataIOFetcher(NewsFetcher):
    """
    Fetches news using NewsData.io API.
    """

    BASE_URL = "https://newsdata.io/api/1/news"

    def __init__(
        self,
        api_key: Optional[str],
        max_results: int = 10,
        language: str = "en",
        timeout_sec: int = 10,
    ):
        if not api_key:
            raise ValueError("NewsData.io API key is missing")

        self.api_key = api_key
        self.max_results = max_results
        self.language = language
        self.timeout = timeout_sec

    def fetch(
        self,
        *,
        query: Optional[str] = None,
        country: Optional[str] = None,
        category: str = "business",
    ) -> List[Dict]:

        params = {
            "apikey": self.api_key,
            "language": self.language,
            "category": category,
            "size": self.max_results,
        }

        if query:
            params["q"] = query

        if country:
            params["country"] = country

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()
        payload = response.json()
        #print(payload['results'][0])  # DEBUG

        return self._normalize(payload.get("results", []))

    @staticmethod
    def _normalize(items):
        normalized = []

        for item in items:
            for i in item:
                print(f"{i}: {item[i]}")  # DEBUG
            country = item.get("country")

            # Normalize region to string
            if isinstance(country, list):
                region = country[0].lower() if country else None
            elif isinstance(country, str):
                region = country.lower()
            else:
                region = None

            normalized.append(
                {
                    "title": item.get("title"),
                    "source": item.get("source_id"),
                    "published_at": NewsDataIOFetcher._parse_date(
                        item.get("pubDate")
                    ),
                    "summary": item.get("description"),
                    "tags": item.get("keywords") or [],
                    "region": region,
                    "link": item.get("link"),
                }
            )

        return normalized

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None