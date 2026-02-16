import os
import pytest
from datetime import datetime
from pprint import pprint

from market_intelligence.news.news_fetcher import NewsDataIOFetcher
from market_intelligence.news.news_filter import NewsFilter
from market_intelligence.news.investing_news_fetcher import InvestingComNiftyNewsFetcher
from market_intelligence.context.instrument_context import InstrumentContext
from market_intelligence.context.time_context import TimeContext


@pytest.fixture(scope="module")
def news_fetcher():
    api_key = os.getenv("NEWSDATA_API_KEY")
    assert api_key, "NEWSDATA_API_KEY env variable not set"

    return NewsDataIOFetcher(
        api_key=api_key,
        max_results=10,
    )


@pytest.fixture(scope="module")
def time_ctx():
    return TimeContext("Asia/Kolkata")


@pytest.fixture(scope="module")
def instrument_contexts():
    return {
        "nifty": InstrumentContext(
            instrument_key="nifty50",
            symbol="NIFTY",
            exchange="NSE",
            instrument_type="index",
            timezone="Asia/Kolkata",
            tick_size=0.05,
        ),
        "reliance": InstrumentContext(
            instrument_key="reliance",
            symbol="RELIANCE",
            exchange="NSE",
            instrument_type="stock",
            timezone="Asia/Kolkata",
            tick_size=0.05,
        ),
        "eurusd": InstrumentContext(
            instrument_key="eurusd",
            symbol="EURUSD",
            exchange="FOREX",
            instrument_type="forex",
            timezone="UTC",
            tick_size=0.0001,
        ),
    }


""" def test_news_fetch_basic(news_fetcher):
    '''
    Validate NewsData.io fetch works and returns normalized output.
    '''
    news = news_fetcher.fetch(query="markets", country="in")

    print("\nFetched News:")
    pprint(news[:3])

    assert isinstance(news, list)
    assert len(news) > 0

    sample = news[0]
    assert "title" in sample
    assert "source" in sample
    assert "published_at" in sample
    assert "summary" in sample
    assert "region" in sample """


def test_news_filter_for_index(news_fetcher, time_ctx, instrument_contexts):
    """
    Validate filtering for index (NIFTY).
    """
    raw_news = news_fetcher.fetch(query="NIFTY", country="in")
    #fetcher = InvestingComNiftyNewsFetcher()
    #raw_news = fetcher.fetch(max_items=8)

    print("\nRaw Index News:")
    pprint(raw_news)

    filterer = NewsFilter(max_age_hours=24)

    filtered = filterer.filter(
        news=raw_news,
        instrument_ctx=instrument_contexts["nifty"],
        now=time_ctx.now,
    )

    print("\nFiltered Index News:")
    pprint(filtered)

    assert isinstance(filtered, list)
    # It's okay if some days have fewer relevant items
    for item in filtered:
        assert item["published_at"] is not None


""" def test_news_filter_for_stock(news_fetcher, time_ctx, instrument_contexts):
    '''
    Validate filtering for stock (RELIANCE).
    '''
    raw_news = news_fetcher.fetch(query="NIFTY50", country="in")

    filterer = NewsFilter(max_age_hours=24)

    filtered = filterer.filter(
        news=raw_news,
        instrument_ctx=instrument_contexts["reliance"],
        now=time_ctx.now,
    )

    print("\nFiltered Stock News:")
    pprint(filtered)

    assert isinstance(filtered, list)


def test_news_filter_for_forex(news_fetcher, time_ctx, instrument_contexts):
    '''
    Validate filtering for forex instruments.
    '''
    raw_news = news_fetcher.fetch(query="USD", country="us")

    filterer = NewsFilter(max_age_hours=24)

    filtered = filterer.filter(
        news=raw_news,
        instrument_ctx=instrument_contexts["eurusd"],
        now=time_ctx.now,
    )

    print("\nFiltered Forex News:")
    pprint(filtered)

    assert isinstance(filtered, list)
    assert len(filtered) > 0 """