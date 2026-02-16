"""
This module contains tests for the candle pattern strategy.
It includes loading historical data, applying indicators, detecting patterns, and creating a complete dataframe
with all necessary information for further analysis. 
Save the results to CSV files for each stock and interval.
"""

import pytest
import datetime
from data.kite_loader import KiteDataLoader
import config
import ta
from indicators.ema import EMAIndicator
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.adx import ADXIndicator
from indicators.atr import ATRIndicator
from indicators.bbw import BBWIndicator
from indicators.divergence_detector import DivergenceDetector
from indicators.candle_patterns import CandlePatternDetector
from indicators.trendlines_with_breaks import TrendlinesWithBreaks
from strategy.candle_pattern import CandlePatternStrategy
from strategy.ema import EMA
from general_lib.convert_timeframe import TimeframeConverter
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
    
    available_interval=["60minute", "15minute", "10minute","5minute", "3minute", "minute", "1D"]
    interval=["15minute"]  # For testing, use only 5-minute interval
    data_dict = {}
    data_limit_dict = {"1D": 200, "60minute": 400, "15minute": 200, "5minute": 100, "minute": 50}
    for token in config.INSTRUMENT_TOKENS:
        token_name = token_name_map.get(token, f"Token_{token}")
        print("#" * 50)
        print(f"Loading data for {token_name}...")
        print("#" * 50)
        data_dict[token] = {}
        for intv in interval:
            data_dict[token][intv] = None
            # Set start as 1 month ago from today
            days=7
            if intv == "1D":
                days=365
            start_date=datetime.datetime.now() - datetime.timedelta(days=days)
            #start_date=datetime.datetime(2026, 2, 9)
            # make end date as current date + 1 minute to ensure we get the latest data
            end_date=datetime.datetime.now() + datetime.timedelta(minutes=1)
            #end_date=datetime.datetime(2026, 1, 31)
            
            print(f"Fetching {intv} data for {token_name} from {start_date.date()} to {end_date.date()}")
            df = loader.get_data(
                instrument_token=token,
                from_date=start_date,
                to_date=end_date,
                interval=intv)
            # Check if date columns format is correct
            #if not df.empty:
            #    print("Checking date column format...")
            #    print("Date column dtype:", df["date"].dtype)
            #    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "Date column is not in datetime format."

            
            # Print the last few rows of the dataframe for debugging
            #print(f"Data for {token_name} at interval {intv}:")
            #print(df)
            data_dict[token][intv] = df
        
    return data_dict, token_name_map


def test_indicator_and_pattern_strategy(kite_data):
    """
    Test function to validate candle pattern strategy output.
    """    
    all_signals = []
    kite_data_dict, token_name_map = kite_data
    for token, df_intv in kite_data_dict.items():
        stock_name = token_name_map.get(token, f"Token_{token}")
        all_dfs = []
        for intv, df in df_intv.items():
            if df.empty:
                print(f"No data for {token} at interval {intv}")
                continue

            """ # EMA 20
            ema_20 = EMAIndicator()
            df = ema_20.ema_calc(df, period=20)

            # Distance of price from EMA 20
            ema_20 = EMA(period=20)
            df = ema_20.ema_to_price_distance(df)

            # EMA 50
            ema_50 = EMAIndicator()
            df = ema_50.ema_calc(df, period=50)
            # Distance of price from EMA 50
            ema_50 = EMA(period=50)
            df = ema_50.ema_to_price_distance(df)

            # EMA 200
            ema_200 = EMAIndicator()
            df = ema_200.ema_calc(df, period=200)
            # Distance of price from EMA 200
            ema_200 = EMA(period=200)
            df = ema_200.ema_to_price_distance(df)

            # Distance between EMAs
            # EMA 20 and EMA 50
            ema = EMA()
            df = ema.distance_between_emas(df, 'EMA_20', 'EMA_50')
            # EMA 50 and EMA 200
            df = ema.distance_between_emas(df, 'EMA_50', 'EMA_200')
            # EMA 20 and EMA 200
            df = ema.distance_between_emas(df, 'EMA_20', 'EMA_200')
            
            rsi_period = [14]
            adx_period = [14]
            atr_period = [14]
            bbw_period = [20]

            # RSI
            for period in rsi_period:
                #df[f"RSI_{period}"] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
                #df[f"RSI_{period}_Div_Type"] = DivergenceDetector.detect_rsi_divergence(df, period)
                # Run divergence detection
                
                rsi = RSIIndicator(period=period)
                df = rsi.compute(df)
                detector = DivergenceDetector(df)
                df = detector.detect_rsi_divergence(period=period, order=1)
            
            # RSI Divergence Length
            detector = DivergenceDetector(df)
            df = detector.div_length_before_price_dir_change(column="RSI")
            
            # MACD
            macd = MACDIndicator()
            df = macd.compute(df)
            detector = DivergenceDetector(df)
            df = detector.detect_macd_divergence(order=1)
            df = detector.div_length_before_price_dir_change(column="MACD")

            # ADX
            for period in adx_period:
                adx = ADXIndicator(period=period)
                df = adx.compute(df)

            # ATR
            for period in atr_period:
                atr = ATRIndicator(period=period)
                df = atr.compute(df)

            # BBW
            for period in bbw_period:
                bbw = BBWIndicator(period=period)
                df = bbw.compute(df) """
            
            # 
            # Initialize indicator 
            tl = TrendlinesWithBreaks(df, length=3, mult=1.0, method='Atr') 
            # Compute trendlines and breakouts 
            result = tl.compute() 
            print("\n=== Trendline Output ===")
             
            print(result[['date', 'upper', 'lower', 'up_break', 'dn_break']]) 
            # Compute follow-through 
            follow = tl.breakout_followthrough(lookahead=3) 
            print("\n=== Breakout Follow-Through ===") 
            print(follow)

            # Plot 
            tl.plot()
            
            """ # Step 1: Detect patterns
            cdl_detector = CandlePatternDetector(df)
            pattern_df = cdl_detector.detect_patterns()

            # Step 2: Run candle strategy
            strategy = CandlePatternStrategy(pattern_df)
            result_df = strategy.run(trading_type = "swing")

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
            #assert example_row["price_dir"] in ["up", "down", "sideways"]

            # step 6: Check if date is in datetime format
            print("Checking date column format...")
            print("Date column dtype:", result_df["date"].dtype)
            assert pd.api.types.is_datetime64_any_dtype(result_df["date"]), "Date column is not in datetime format."

            print(f"Data for {stock_name} at interval {intv}:")
            print(result_df[["date", "candle", "cdl_implication", "cdl_strength", "cdl_direction", "price_chg", "price_dir"]].tail(25)) """

            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save to CSV
        stock_name_csv = stock_name.replace(" ", "_").replace("/", "_")
        combined_df.to_csv(f"{stock_name_csv}.csv", index=False)

        # print last 25 rows for debugging
        #print("=" * 50)
        #print("STOCK NAME:", stock_name)
        #print("-" * 25)
        #print("INTERVAL:", intv)
        #print("-" * 25)
        #print(combined_df.tail(50)[["date", "candle", "cdl_implication", "cdl_strength", "cdl_direction", "price_chg", "price_dir"]])

    """tgram_bot = TelegramBot()
    message = "Hii da"
    response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS)"""
