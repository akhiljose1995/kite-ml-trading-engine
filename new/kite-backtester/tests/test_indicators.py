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
#from indicators.adx import ADXIndicator

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
        to_date=datetime.datetime(2025, 6, 13),
        interval="15minute"
    )
    return df

""" def test_ema_indicator(kite_data):
    ema = EMAIndicator(period=20)
    result = ema.compute(kite_data)
    print(result.iloc[30:])
    col_name = f"EMA_{ema.period}"
    assert col_name in result.columns, f"Column {col_name} not found in the result."
    assert not result[col_name].isna().all()

def test_rsi_indicator(kite_data):
    rsi = RSIIndicator(period=14)
    result = rsi.compute(kite_data)
    print(result.iloc[:40])
    col_name = f"RSI_{rsi.period}"
    assert col_name in result.columns, f"Column {col_name} not found in the result."
    assert not result[col_name].isna().all()

def test_macd_indicator(kite_data):
    macd = MACDIndicator()
    result = macd.compute(kite_data)
    print(result.iloc[:40])
    col_names = [f"MACD", "MACD_Signal"]
    assert all(col in result.columns for col in col_names)
    #assert not result[col_names].isna().all() """

def test_adx_indicator(kite_data):
    adx_period = 14
    adx = ADXIndicator(period=adx_period)
    result = adx.compute(kite_data)
    col_names = [f"ADX_{adx_period}", f"PDI_{adx_period}", f"NDI_{adx_period}"]
    print(result.reset_index()[["date"] + col_names])
    assert all(col in result.columns for col in col_names)
    print(result.columns)
    #assert not result[col_names].isna().all()

""" def test_divergence_detector(kite_data):
    """
    #Tests the RSI and MACD divergence detection using real data.
"""
    # Add required indicators
    rsi_period = 14
    rsi_col = f"RSI_{rsi_period}"
    macd_col = "MACD"
    adx_period = 14
    rsi = RSIIndicator(period=rsi_period)
    macd = MACDIndicator()
    ema = EMAIndicator(period=20)

    df = rsi.compute(kite_data)
    df = macd.compute(df)
    df = ema.compute(df)

    # Ensure required columns are present
    assert rsi_col in df.columns
    assert macd_col in df.columns
    assert "low" in df.columns

    # Run divergence detection
    detector = DivergenceDetector(df)
    rsi_divs = detector.detect_rsi_divergence(period=14, order=1)
    macd_divs = detector.detect_macd_divergence(order=1)

    # Print for manual inspection
    #print("RSI Divergences:", rsi_divs)
    print("MACD Divergences:", macd_divs)

    # Check type and structure
    assert isinstance(rsi_divs, list)
    assert isinstance(macd_divs, list)

    for div in rsi_divs:
        assert "type" in div and div["type"] in ["Bullish", "Bearish"]
        assert "end index" in div and "rsi" in div and "price" in div

    for div in macd_divs:
        assert "type" in div and div["type"] in ["Bullish", "Bearish"]
        assert "end index" in div and "MACD" in div and "price" in div """