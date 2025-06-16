import pytest
import datetime
from data.kite_loader import KiteDataLoader
import config
import ta
from indicators.ema import EMAIndicator
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.adx import ADXIndicator
from indicators.divergence_detector import DivergenceDetector
from indicators.candle_patterns import CandlePatternDetector
from strategy.candle_pattern import CandlePatternStrategy


@pytest.fixture(scope="module")
def kite_data():
    """
    Fixture to load historical data using KiteDataLoader
    for a small time range to test the strategy.
    """
    loader = KiteDataLoader(
        api_key=config.API_KEY,
        access_token=config.ACCESS_TOKEN,
        request_token=config.REQUEST_TOKEN,
        api_secret=config.API_SECRET
    )
    df = loader.get_data(
        instrument_token=config.INSTRUMENT_TOKEN,
        from_date=datetime.datetime(2025, 6, 12),
        to_date=datetime.datetime(2025, 6, 14),
        interval="15minute"
    )
    return df


def test_candle_pattern_strategy(kite_data):
    """
    Test function to validate candle pattern strategy output.
    """
    # Step 1: Detect patterns
    cdl_detector = CandlePatternDetector(kite_data.copy())
    pattern_df = cdl_detector.detect_patterns()

    # Step 2: Run candle strategy
    strategy = CandlePatternStrategy(pattern_df)
    result_df = strategy.run()

    # Step 3: Validate structure
    assert "candle" in result_df.columns
    assert "CDL_close" in result_df.columns
    assert "interval" in result_df.columns
    assert "type" in result_df.columns
    assert "price_dir" in result_df.columns
    assert "price_chg" in result_df.columns

    # Step 4: Check at least one pattern detected with metadata
    candle_detected = result_df[result_df["candle"].notna()]
    assert not candle_detected.empty, "No candles detected. Adjust your dataset or detection logic."

    # Step 5: Validate correct mapping from JSON
    example_row = candle_detected.iloc[0]
    assert example_row["type"] in ["bullish", "bearish", "neutral", "depends"]
    assert isinstance(example_row["price_chg"], float)
    assert example_row["price_dir"] in ["up", "down", "sideways"]

    # Save to CSV
    result_df.to_csv("candle_pattern_strategy_output.csv", index=False)
