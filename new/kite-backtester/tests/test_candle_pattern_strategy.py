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
from telegram_bot.telegram_bot import TelegramBot
import pandas as pd


# Show all columns, all rows, and full column content
pd.set_option('display.max_columns', None)       # Show all columns
pd.set_option('display.max_rows', None)          # Show all rows (optional)
pd.set_option('display.max_colwidth', None)      # Don't truncate column contents
pd.set_option('display.width', 0)                # Auto-detect console width

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
    token_name_map = loader.get_instrument_names()
    token_name = token_name_map.get(config.INSTRUMENT_TOKEN, f"Token_{config.INSTRUMENT_TOKEN}")
    print("#" * 50)
    print(f"Loading data for {token_name}...")
    print("#" * 50)
    interval=["60minute", "15minute", "5minute"]
    data_dict = {}
    for intv in interval:
        data_dict[intv] = None
        df = loader.get_data(
            instrument_token=config.INSTRUMENT_TOKEN,
            from_date=datetime.datetime(2025, 6, 22),
            to_date=datetime.datetime(2025, 7, 1),
            interval=intv)
        data_dict[intv] = df
    
    return data_dict


def test_candle_pattern_strategy(kite_data):
    """
    Test function to validate candle pattern strategy output.
    """
    for name, df in kite_data.items():
        # Step 1: Detect patterns
        cdl_detector = CandlePatternDetector(df)
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
        assert isinstance(example_row["price_chg"], float)
        assert example_row["price_dir"] in ["up", "down", "sideways"]

        # Save to CSV
        result_df.to_csv("candle_pattern_strategy_output.csv", index=False)

        # print last 25 rows for debugging
        print("-" * 50)
        print("INTERVAL:", name)
        print("-" * 25)
        print(result_df.tail(50)[["date", "single_candle", "candle", "cdl_implication", "cdl_strength", "cdl_direction", "price_dir"]])

    """ tgram_bot = TelegramBot()
    message = "Hii da myraa"
    response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS) """
