import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import List, Dict


class InvestingComNiftyNewsFetcher:
    """
    Scrapes NIFTY-related news from Investing.com (India).
    Intended for low-frequency, personal-use scraping only.
    """

    URL = "https://in.investing.com/indices/s-p-cnx-nifty-news"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def fetch(self, max_items: int = 10) -> List[Dict]:
        response = requests.get(self.URL, headers=self.HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.select("article")[:max_items]

        news_items = []

        for article in articles:
            title_tag = article.select_one("a")
            time_tag = article.select_one("time")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            link = title_tag.get("href")

            if link and link.startswith("/"):
                link = "https://in.investing.com" + link

            published_at = None
            if time_tag and time_tag.has_attr("datetime"):
                try:
                    published_at = datetime.fromisoformat(
                        time_tag["datetime"].replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except Exception:
                    published_at = None

            news_items.append(
                {
                    "title": title,
                    "source": "investing.com",
                    "published_at": published_at,
                    "summary": None,  # LLM will summarize later
                    "tags": ["nifty", "india", "market"],
                    "region": "in",
                    "link": link,
                }
            )

        return news_items