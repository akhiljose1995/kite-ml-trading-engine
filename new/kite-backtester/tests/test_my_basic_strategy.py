import pytest
import datetime
import pandas as pd
import os
from data.kite_loader import KiteDataLoader
import config
from strategy.my_basic import BasicTradingStrategy
from indicators.ema import EMAIndicator
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.adx import ADXIndicator
from indicators.divergence_detector import DivergenceDetector

@pytest.fixture(scope="module")
def kite_data():
    """
    Fixture to load historical data for multiple instruments
    """
    loader = KiteDataLoader(
        api_key=config.API_KEY,
        access_token=config.ACCESS_TOKEN,
        request_token=config.REQUEST_TOKEN,
        api_secret=config.API_SECRET
    )

    data_dict = {}
    for token in config.INSTRUMENT_TOKENS:
        df = loader.get_data(
            instrument_token=token,
            from_date=datetime.datetime(2025, 5, 1),
            to_date=datetime.datetime(2025, 6, 17),
            interval="60minute"
        )
        df.reset_index(inplace=True)
        data_dict[token] = df
    return data_dict


def get_instrument_names():
    """
    Load instrument names from instruments.csv
    """
    # Get project root (one level above tests/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "instruments.csv")
    df = pd.read_csv(csv_path)
    token_name_map = dict(zip(df["instrument_token"], df["name"]))
    return token_name_map


def test_basic_strategy_with_real_data(kite_data):
    token_name_map = get_instrument_names()
    all_signals = []

    for token, df in kite_data.items():
        rsi_period = [14]
        adx_period = [14]
        for period in rsi_period:
            #df[f"RSI_{period}"] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
            #df[f"RSI_{period}_Div_Type"] = DivergenceDetector.detect_rsi_divergence(df, period)
            # Run divergence detection
            
            rsi = RSIIndicator(period=period)
            df = rsi.compute(df)
            detector = DivergenceDetector(df)
            df = detector.detect_rsi_divergence(period=period, order=1)
            
        
        macd = MACDIndicator()
        df = macd.compute(df)
        detector = DivergenceDetector(df)
        df = detector.detect_macd_divergence(order=1)

        for period in adx_period:
            adx = ADXIndicator(period=period)
            df = adx.compute(df)

        stock_name = token_name_map.get(token, f"Token_{token}")
        strategy = BasicTradingStrategy(df, stock_name)
        signals = strategy.run()
        all_signals.append(signals)

    result_df = pd.DataFrame(all_signals)
    print(result_df)

    # You can also export to CSV for review if needed:
    result_df.to_csv("strategy_signals_output.csv", index=False)

    # Save the original data for reference
    df.to_csv("kite_data_output.csv", index=False)

    # Basic assertion just to ensure output exists
    assert not result_df.empty
