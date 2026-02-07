"""
This test file is to find support and resistance zones based on swing highs and lows.
"""

import pytest
import datetime
from data.kite_loader import KiteDataLoader
import config
from general_lib.convert_timeframe import TimeframeConverter
from general_lib.swing_detector import SwingDetector, SwingConfig, TimeframeConfig
from general_lib.zone_clustering import ZoneCluster, ZoneClusterConfig
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
    
    available_interval=["60minute", "15minute", "5minute"]
    interval=["1D"]  # For testing, use only 5-minute interval
    req_intv = "15minute"
    data_dict = {}
    data_limit_dict = {"1D": 200, "60minute": 400, "15minute": 200, "5minute": 100, "minute": 50}
    for token in config.INSTRUMENT_TOKENS:
        token_name = token_name_map.get(token, f"Token_{token}")
        print("#" * 50)
        print(f"Loading data for {token_name}...")
        print("#" * 50)
        data_dict[token] = {}
        for intv in interval:
            if intv not in available_interval:
                req_intv, intv = intv, "15minute" # Default to 15-minute if desired interval not available
            data_dict[token][intv] = None
            # Set start as 1 month ago from today
            days=30
            if req_intv == "1D":
                days=365
            #start_date=datetime.datetime.now() - datetime.timedelta(days=days)
            start_date=datetime.datetime(2024, 1, 1)
            # make end date as current date + 1 minute to ensure we get the latest data
            end_date=datetime.datetime.now() + datetime.timedelta(minutes=1)
            #end_date=datetime.datetime(2026, 1, 31)
            current_start = start_date
            all_df = []
            while current_start < end_date:
                current_end = min(current_start + datetime.timedelta(days=data_limit_dict[intv]), end_date)

                print(f"Fetching {intv} data for {token_name} from {current_start.date()} to {current_end.date()}")
                df = loader.get_data(
                    instrument_token=token,
                    from_date=current_start,
                    to_date=current_end,
                    interval=intv)
                # Check if date columns format is correct
                #if not df.empty:
                #    print("Checking date column format...")
                #    print("Date column dtype:", df["date"].dtype)
                #    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "Date column is not in datetime format."

                all_df.append(df)
                current_start = current_end + datetime.timedelta(days=1)

            combined_df = pd.concat(all_df, ignore_index=True)
            # if requested interval is not available, convert from 15-minute to desired interval
            if req_intv not in available_interval:
                print(f"Converting from {intv} to {req_intv}...")
                converter = TimeframeConverter(combined_df)
                combined_df = converter.convert(req_intv)
                combined_df.reset_index(inplace=True)  # Ensure 'date' column is available after conversion
            
            # Print the last few rows of the dataframe for debugging
            #print(f"Data for {token_name} at interval {intv}:")
            #print(combined_df)
            data_dict[token][intv] = combined_df
        
    return data_dict, token_name_map

def test_detect_sr_zones(kite_data):
    """
    Test function to validate support and resistance zone detection
    using swing highs and swing lows.
    """

    kite_data_dict, token_name_map = kite_data

    # ---------------------------------------------
    # Timeframe-specific swing configuration
    # ---------------------------------------------
    timeframe_swing_params = {
        "1D": TimeframeConfig(lookback=7, atr_period=14, atr_multiplier=0.7),
        "60minute": TimeframeConfig(lookback=6, atr_period=14, atr_multiplier=0.6),
        "15minute": TimeframeConfig(lookback=5, atr_period=14, atr_multiplier=0.5),
        "5minute": TimeframeConfig(lookback=4, atr_period=14, atr_multiplier=0.4),
        "minute": TimeframeConfig(lookback=3, atr_period=14, atr_multiplier=0.3),
    }

    # ---------------------------------------------
    # Iterate over instruments & intervals
    # ---------------------------------------------
    for token, interval_dict in kite_data_dict.items():
        token_name = token_name_map.get(token, token)

        for interval, df in interval_dict.items():
            if df is None or df.empty:
                print(f"Skipping {token_name} | {interval} (no data)")
                continue

            print("\n" + "=" * 100)
            print(f"Processing Swing Detection for {token_name} | Interval: {interval}")
            print("=" * 100)

            # ---------------------------------------------
            # Prepare dataframe
            # ---------------------------------------------
            df = df.copy()

            # Ensure datetime index
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            # ---------------------------------------------
            # Initialize Swing Detector
            # ---------------------------------------------
            swing_config = SwingConfig(
                interval=interval,
                timeframe_params=timeframe_swing_params,
                price_col_map={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                },
            )

            detector = SwingDetector(swing_config)

            # ---------------------------------------------
            # Detect swings
            # ---------------------------------------------
            result_df = detector.detect(df)

            # ---------------------------------------------
            # Filter only swing points
            # ---------------------------------------------
            swing_df = result_df[
                (result_df["swing_high"]) | (result_df["swing_low"])
            ].copy()

            # Add helper column for readability
            swing_df["swing_type"] = None
            swing_df.loc[swing_df["swing_high"], "swing_type"] = "SWING_HIGH"
            swing_df.loc[swing_df["swing_low"], "swing_type"] = "SWING_LOW"

            # ---------------------------------------------
            # PRINT RESULTS (IMPORTANT)
            # ---------------------------------------------
            print("\nDetected Swing Points:")
            print(swing_df[["swing_high_price", "swing_low_price", "swing_type"]])
            print(f"Total Swings Detected: {len(swing_df)}")

            # ---------------------------------------------
            # Basic Assertions (sanity checks)
            # ---------------------------------------------
            assert "swing_high" in result_df.columns
            assert "swing_low" in result_df.columns
            assert "atr" in result_df.columns

            # Optional: ensure at least some swings exist
            assert len(swing_df) > 0, (
                f"No swing points detected for {token_name} | {interval}"
            )

            # ---------------------------------------------
            # Cluster swing points into zones
            # ---------------------------------------------
            zc = ZoneCluster()
            zones_df = zc.cluster(swing_df)

            print(zones_df)