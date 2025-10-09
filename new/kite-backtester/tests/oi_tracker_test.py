
# Script to track option OI changes
# This script connects to the Kite API, fetches NFO option chain data for a specified underlying,
# calculates Open Interest (OI) changes over various intervals, and displays this information
# in live-updating tables in the console.
# ==============================
# Testing and Encoding Setup
# ==============================
import pytest
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ==============================
# Date and Time Utilities
# ==============================
from datetime import datetime, date, timedelta, timezone
from datetime import time as time1
import time  # For time.sleep() in live update loop

# ==============================
# OS and System Utilities
# ==============================
import os
import traceback
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================
# Data Science and Visualization
# ==============================
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Matplotlib setup for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import plotly.graph_objects as go

# ==============================
# Web and API Integration
# ==============================
import requests
from kiteconnect import KiteConnect, exceptions

# ==============================
# Web UI and Dashboard
# ==============================
import streamlit as st

# ==============================
# Rich Terminal Output
# ==============================
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

# ==============================
# HTML to Image Conversion
# ==============================
from html2image import Html2Image

# ==============================
# Custom Modules
# ==============================
import config
from telegram_bot.telegram_bot import TelegramBot
from general_lib.option_chain import OptionChainTracker
from general_lib.webdriver_tools import WebDriverSession

# ==============================================================================
# --- SCRIPT CONFIGURATION ---
# ==============================================================================
# Instructions for API Keys:
# 1. Set Environment Variables:
#    For Linux/macOS:
#      export KITE_API_KEY="your_api_key"
#      export KITE_API_SECRET="your_api_secret"
#    For Windows (PowerShell):
#      $env:KITE_API_KEY="your_api_key"
#      $env:KITE_API_SECRET="your_api_secret"
#    Replace "your_api_key" and "your_api_secret" with your actual Kite API credentials.
# 2. Alternatively, you can hardcode them below by changing API_KEY_DEFAULT and
#    API_SECRET_DEFAULT, but using environment variables is recommended for security.

# --- API Credentials ---
API_KEY_DEFAULT = config.API_KEY        # Default placeholder if env var not found
API_SECRET_DEFAULT = config.API_SECRET  # Default placeholder if env var not found

# --- Data Fetching Parameters ---
HISTORICAL_DATA_MINUTES = 800         # How many minutes of historical data to fetch for OI calculation
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30) # Past intervals (in minutes) to calculate OI change from latest OI

# --- Trading Parameters ---
UNDERLYING_SYMBOL = ""       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
ATM_TRADESYMBOL = ""        # To store the ATM trading symbol
ATM_LTP = 0.0                # To store the ATM LTP
FAKE_BREAKOUT_THRESHOLD = 5  # Threshold in points to consider a fake breakout
index_symbol = "nifty50"  # Default index to track; options: "nifty50", "niftybank", "sensex"
STRIKE_DIFFERENCE = 100               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
## --- Exchange Configuration ---
# Use KiteConnect attributes for exchange names
EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_BFO  # Exchange for NFO options contracts
EXCHANGE_LTP = KiteConnect.EXCHANGE_BSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50)

# --- Display and Logging ---
REFRESH_INTERVAL_SECONDS = 1        # How often to refresh the data and tables (in seconds)
LOG_FILE_NAME = "oi_tracker.log"     # Name of the log file
FILE_LOG_LEVEL = "DEBUG, INFO"              # Logging level for the log file (DEBUG, INFO, WARNING, ERROR, CRITICAL)
PCT_CHANGE_THRESHOLDS = {            # Thresholds for highlighting OI % change (interval_in_min: percentage)
     5: 8.0,
    10: 10.0,  # Highlight if 10-min % change > 10%
    15: 15.0,  # Highlight if 15-min % change > 15%
    30: 25.0   # Highlight if 30-min % change > 25%
}
PLOT_PRINT_TIME = time.time()
PLOT_PRINT_INTERVAL = 30

# Note: Console output is managed by Rich; file logging captures more detailed/background info.

# ==============================================================================
# --- END OF CONFIGURATION ---
# ==============================================================================

# --- Global Initializations ---
# Setup file logging according to configured level and file name
logging.basicConfig(
    level=getattr(logging, FILE_LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    filename=LOG_FILE_NAME,
    filemode='a'  # Append to the log file
)

# Derive underlying prefix (e.g., NIFTY from "NIFTY 50") for instrument searching

now = datetime.now()
current_year_two_digits = now.strftime("%y")
UNDERLYING_PREFIX = ""

# Retrieve API keys from environment variables or use default placeholders
api_key_to_use = API_KEY_DEFAULT
api_secret_to_use = API_SECRET_DEFAULT


# Global KiteConnect and Rich Console instances
# Initialize KiteConnect with reduced library logging to prevent console clutter from the library itself
kite = KiteConnect(api_key=api_key_to_use)
console = Console(record=True) # For Rich text and table display

# Initialize variable used for PCR calculation
pe_oi_dict = {}
ce_oi_dict = {}
PCR_STRIKE_COUNT = 1

# Global variable to store support and resistance created by OI
OI_SUPPORT = None
OI_RESISTANCE = None
OI_DIFF_THRESHOLD_UPPER = 4000000
OI_DIFF_THRESHOLD_LOWER = 2000000
OI_diff_between_PE_and_CE = None
OI_diff_between_PE_and_CE_PREV = {}
OI_diff_between_PE_and_CE_df = None
ATM_OI_DIFF_SLOPE = 0
ATM_OI_DIFF_INTERCEPT = 0
ATM_CE_OI_SLOPE = 0
ATM_PE_OI_SLOPE = 0
OI_DIFF_CROSS_TIME = 9999

# Telegram Message related
MESSAGE_TRIGGERED = False
MESSAGE_TRIGGER_TIMER = 0
PREVIOUS_TRADESYMBOL = ""

# --- Core Functions ---
def calculate_slope(df):
    # Ensure datetime index is numeric (convert to seconds)
    #x = np.array((df.index - df.index[0]).total_seconds()).reshape(-1, 1)
    x = np.arange(len(df)).reshape(-1, 1)
    y = df.values

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]   # slope of the line
    intercept = model.intercept_  # y-intercept of the line

    return slope, intercept

def initialize_index(index="nifty50"):
    """
    Depending on parameter UNDERLYING_SYMBOL, take in the symbol and set the 
    """  # e.g., "nifty"
    console.print(f"Initializing index parameters for: {index}")
    global UNDERLYING_SYMBOL, UNDERLYING_PREFIX, STRIKE_DIFFERENCE, OPTIONS_COUNT, EXCHANGE_NFO_OPTIONS, EXCHANGE_LTP, FAKE_BREAKOUT_THRESHOLD
    if index.lower() in ["nifty50", "niftybank"]:
        print(f"Running OI tracker for: {index}")
        if index.lower() == "nifty50":
            UNDERLYING_SYMBOL = "NIFTY 50"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
            
        elif index.lower() == "niftybank":
            UNDERLYING_SYMBOL = "NIFTY BANK"
        FAKE_BREAKOUT_THRESHOLD = 5
        STRIKE_DIFFERENCE = 50               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
        OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
        # --- Exchange Configuration ---
        # Use KiteConnect attributes for exchange names
        EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_NFO  # Exchange for NFO options contracts
        EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50)

    elif index.lower() == "sensex":
        # --- Trading Parameters ---
        # --  SENSEX Parameters --
        UNDERLYING_SYMBOL = "SENSEX"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
        FAKE_BREAKOUT_THRESHOLD = 10
        STRIKE_DIFFERENCE = 100               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
        OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
        ## --- Exchange Configuration ---
        # Use KiteConnect attributes for exchange names
        EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_BFO  # Exchange for NFO options contracts
        EXCHANGE_LTP = KiteConnect.EXCHANGE_BSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50)
    UNDERLYING_PREFIX = UNDERLYING_SYMBOL.split(" ")[0].upper()

def get_atm_strike(kite_obj: KiteConnect, underlying_sym: str, exch_for_ltp: str, strike_diff: int):
    """
    Fetches the Last Traded Price (LTP) for the underlying symbol and calculates the At-The-Money (ATM) strike.

    Args:
        kite_obj: Initialized KiteConnect object.
        underlying_sym: The underlying symbol (e.g., "NIFTY 50").
        exch_for_ltp: The exchange to fetch LTP from (e.g., "NSE").
        strike_diff: The difference between consecutive option strikes.

    Returns:
        The calculated ATM strike as a float, or None if LTP cannot be fetched or an error occurs.
    """
    try:
        # Construct the instrument identifier for LTP (e.g., "NSE:NIFTY 50")
        ltp_instrument = f"{exch_for_ltp}:{underlying_sym}"
        ltp_data = kite_obj.ltp(ltp_instrument)
        print("LTP Data:",ltp_data)

        # Validate LTP data structure
        if not ltp_data or ltp_instrument not in ltp_data or 'last_price' not in ltp_data[ltp_instrument]:
            logging.error(f"LTP data not found or incomplete for {ltp_instrument}. Response: {ltp_data}")
            return None
        
        ltp = ltp_data[ltp_instrument]['last_price']
        # Calculate ATM strike by rounding LTP to the nearest strike difference
        atm_strike = round(ltp / strike_diff) * strike_diff
        logging.debug(f"LTP for {underlying_sym}: {ltp}, Calculated ATM strike: {atm_strike}")
        return atm_strike, ltp
    except Exception as e:
        logging.error(f"Error in get_atm_strike for {underlying_sym}: {e}", exc_info=True)
        return None

def get_nearest_weekly_expiry(instruments: list, underlying_prefix_str: str):
    """
    Finds the nearest future weekly expiry date for the given underlying symbol prefix from a list of instruments.

    Args:
        instruments: A list of instrument dictionaries from Kite API.
        underlying_prefix_str: The prefix of the underlying symbol (e.g., "NIFTY").

    Returns:
        The nearest weekly expiry date as a datetime.date object, or None if no suitable expiry is found.
    """
    today = date.today()
    possible_expiries = set()
    logging.info(f"Searching for nearest weekly expiry for {underlying_prefix_str} among {len(instruments)} instruments.")

    #print(instruments)
    #print(underlying_prefix_str)
    trading_symbol = {} 

    for inst in instruments:
        # Filter for options of the specified underlying
        if inst['name'] == underlying_prefix_str and inst['exchange'] == EXCHANGE_NFO_OPTIONS:
            # Ensure expiry is a date object and is in the future or today
            if isinstance(inst['expiry'], date) and inst['expiry'] >= today:
                possible_expiries.add(inst['expiry'])
                trading_symbol[inst['expiry']] = inst['tradingsymbol']

    #print(possible_expiries) 
    if not possible_expiries:
        logging.error(f"No future expiries found for {underlying_prefix_str}.")
        return None

    # Sort expiries and return the closest one
    nearest_expiry = sorted(list(possible_expiries))[0]
    trading_symbol_of_nearest_expiry = trading_symbol[nearest_expiry]

    symbol_prefix = trading_symbol_of_nearest_expiry[0:len(underlying_prefix_str)+5]
    #print("Trading_symbol = "+symbol_prefix)
    logging.info(f"Nearest weekly expiry for {underlying_prefix_str}: {nearest_expiry}")
    return {"expiry":nearest_expiry, "symbol_prefix":symbol_prefix}

def get_relevant_option_details(instruments: list, atm_strike_val: float, expiry_dt: date, 
                                strike_diff_val: int, opt_count: int, underlying_prefix_str: str, symbol_prefix: str):
    """
    Identifies relevant ITM, ATM, and OTM Call/Put option contract details (tradingsymbol, instrument_token, strike)
    for a given ATM strike and expiry date.

    Args:
        instruments: List of all NFO instrument dictionaries.
        atm_strike_val: The current At-The-Money strike.
        expiry_dt: The expiry date for the options.
        strike_diff_val: The difference between option strikes.
        opt_count: Number of ITM/OTM strikes to fetch on each side of ATM.
        underlying_prefix_str: The prefix of the underlying (e.g., "NIFTY").

    Returns:
        A dictionary where keys are like "atm_ce", "itm1_pe", etc., and values are
        dictionaries containing 'tradingsymbol', 'instrument_token', and 'strike'.
        Returns an empty dictionary if critical inputs are missing.
    """
    relevant_options = {}
    if not expiry_dt or atm_strike_val is None:
        logging.error("Expiry date or ATM strike is None, cannot fetch option details.")
        return relevant_options

    # Format expiry date for Zerodha's trading symbol convention (e.g., NIFTY23OCT19500CE)
    # Year: last two digits. Month: 3-letter uppercase. Day: two digits.
    expiry_str_part = expiry_dt.strftime("%y%b%d").upper() 
    logging.debug(f"Searching for options with expiry: {expiry_dt}, ATM strike: {atm_strike_val}")

    # Iterate from -opt_count (deep ITM for calls / deep OTM for puts) to +opt_count
    for i in range(-opt_count, opt_count + 1):
        current_strike = atm_strike_val + (i * strike_diff_val)
        
        # Construct core part of trading symbols for CE and PE to aid matching
        ce_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"
        pe_symbol_pattern_core = f"{symbol_prefix}{int(current_strike)}"

        #print(ce_symbol_pattern_core)
        
        found_ce, found_pe = None, None
        # Search through all instruments for matches
        for inst in instruments:
            # Match instrument by name, strike, expiry date, and echange 
            if inst['name'] == underlying_prefix_str and \
               inst['strike'] == current_strike and \
               inst['expiry'] == expiry_dt and \
               inst['exchange'] == EXCHANGE_NFO_OPTIONS:
                
                # Further match by instrument type (CE/PE) and ensure core symbol pattern is present
                if inst['instrument_type'] == 'CE' and ce_symbol_pattern_core in inst['tradingsymbol']:
                    found_ce = inst
                elif inst['instrument_type'] == 'PE' and pe_symbol_pattern_core in inst['tradingsymbol']:
                    found_pe = inst
            
            # Optimization: if both CE and PE found for this strike, no need to search further for this strike
            if found_ce and found_pe:
                break 
        
        # Determine key suffix (atm, itm1, otm1, etc.) based on position relative to ATM
        if i == 0: key_suffix = "atm"
        elif i < 0: key_suffix = f"itm{-i}" # e.g., i=-1 -> itm1 (strike < ATM)
        else: key_suffix = f"otm{i}"       # e.g., i=1  -> otm1 (strike > ATM)
        
        if found_ce:
            relevant_options[f"{key_suffix}_ce"] = {
                'tradingsymbol': found_ce['tradingsymbol'], 
                'instrument_token': found_ce['instrument_token'], 
                'strike': current_strike
            }
        else:
            logging.warning(f"CE option not found for strike {current_strike}, expiry {expiry_dt}")
        
        if found_pe:
            relevant_options[f"{key_suffix}_pe"] = {
                'tradingsymbol': found_pe['tradingsymbol'], 
                'instrument_token': found_pe['instrument_token'], 
                'strike': current_strike
            }
        else:
            logging.warning(f"PE option not found for strike {current_strike}, expiry {expiry_dt}")
            
    logging.debug(f"Relevant option details identified: {len(relevant_options)} contracts.")
    return relevant_options

def fetch_historical_oi_data(kite_obj: KiteConnect, option_details_dict: dict, 
                             minutes_of_data: int = HISTORICAL_DATA_MINUTES):
    """
    Fetches historical OI data (minute interval) for the provided option contracts.

    Args:
        kite_obj: Initialized KiteConnect object.
        option_details_dict: Dictionary of option contracts (from get_relevant_option_details).
        minutes_of_data: The duration in minutes for which to fetch historical data.

    Returns:
        A dictionary where keys are option keys (e.g., "atm_ce") and values are lists
        of historical candle data (each candle is a dict). Returns empty list for a contract on error.
    """
    historical_oi_store = {}
    if not option_details_dict:
        logging.warning("No option details provided to fetch_historical_oi_data.")
        return historical_oi_store

    # Calculate date range for historical data API call
    # Kite API expects "YYYY-MM-DD HH:MM:SS" format, and times are usually in UTC.
    to_date = datetime.now()  # Current local time 
    
    """ # Define your custom date and time string
    date_str = "31-07-2025 3:00 PM"
    # Parse it to a datetime object
    to_date = datetime.strptime(date_str, "%d-%m-%Y %I:%M %p") """
    from_date = to_date - timedelta(minutes=minutes_of_data)
    from_date_str = from_date.strftime("%Y-%m-%d %H:%M:00")
    to_date_str = to_date.strftime("%Y-%m-%d %H:%M:00")

    logging.debug(f"Fetching historical data from {from_date_str} to {to_date_str} (UTC)")

    for option_key, details in option_details_dict.items():
        instrument_token = details.get('instrument_token')
        tradingsymbol = details.get('tradingsymbol')

        if not instrument_token:
            logging.warning(f"Missing instrument_token for {option_key} ({tradingsymbol}). Skipping historical data fetch.")
            historical_oi_store[option_key] = []  # Store empty list for consistency
            continue
        
        try:
            logging.debug(f"Fetching historical OI for {tradingsymbol} (Token: {instrument_token})")
            # Fetch minute-interval data including Open Interest (oi=True)
            #print(instrument_token)
            #print(from_date_str)
            #print(to_date_str)
            data = kite_obj.historical_data(instrument_token, from_date_str, to_date_str, interval="minute", oi=True)
            #print(option_key)
            #print(details)
            #print(data)
            #print("-" * 50)
            historical_oi_store[option_key] = data
            logging.debug(f"Fetched {len(data)} records for {tradingsymbol}")
        except Exception as e:
            logging.error(f"Error fetching historical OI for {tradingsymbol} (Token: {instrument_token}): {e}", exc_info=True)
            historical_oi_store[option_key] = []  # Store empty list on error to prevent crashes downstream

    return historical_oi_store

def fetch_historical_premium_data(kite_obj: KiteConnect, option_details_dict: dict, minutes_of_data: int = HISTORICAL_DATA_MINUTES):
    """
    Fetches historical LTP (premium) data for the provided option contracts.

    Returns:
        A dictionary: { option_key: [candles with 'close'] }
    """
    premium_data_store = {}
    to_date = datetime.now()
    from_date = to_date - timedelta(minutes=minutes_of_data)
    from_str = from_date.strftime("%Y-%m-%d %H:%M:00")
    to_str = to_date.strftime("%Y-%m-%d %H:%M:00")

    for option_key, details in option_details_dict.items():
        token = details.get('instrument_token')
        symbol = details.get('tradingsymbol')

        if not token:
            premium_data_store[option_key] = []
            continue

        try:
            data = kite_obj.historical_data(token, from_str, to_str, interval="minute")
            premium_data_store[option_key] = data
        except Exception as e:
            logging.error(f"Error fetching LTP for {symbol}: {e}")
            premium_data_store[option_key] = []

    return premium_data_store

def find_oi_at_timestamp(historical_candles: list, target_time: datetime, 
                          latest_oi_and_time: tuple):
    """
    Finds Open Interest (OI) at or just before a specific target_time from a list of historical candles.
    The historical_candles are assumed to be sorted oldest to newest.

    Args:
        historical_candles: List of candle dictionaries (from Kite API, containing 'date' and 'oi').
        target_time: The target datetime object (timezone-aware) to find OI for.
        latest_oi_and_time: Optional tuple (latest_oi, latest_timestamp). If provided, ensures
                            that the selected candle is not later than this latest_timestamp.
                            This prevents looking "into the future" if target_time is very recent
                            and slightly ahead of the last available candle.

    Returns:
        The Open Interest (int) at the target time, or None if no suitable candle is found.
    """
    if not historical_candles:
        return None

    # Iterate backwards through candles to find the most recent one at or before target_time
    for candle in reversed(historical_candles):
        candle_time = candle['date']  # 'date' field from Kite API is already a timezone-aware datetime object

        if candle_time <= target_time:
            # If latest_oi_and_time is provided, ensure we don't pick a candle
            # whose timestamp is later than the latest_oi_timestamp from the most current data point.
            if latest_oi_and_time and candle_time > latest_oi_and_time[1]:
                continue  # This candle is too new compared to the reference latest OI point
            return candle.get('oi')
            
    # If loop completes, no candle was found at or before target_time (or before the first candle)
    return None

def calculate_oi_differences(raw_historical_data_store: dict, intervals_min: tuple):
    """
    Calculates OI differences between the latest OI and OI at specified past intervals.

    Args:
        raw_historical_data_store: Dictionary of historical candle data for various option contracts.
        intervals_min: A tuple of time intervals in minutes (e.g., (10, 15, 30)) for which to calculate OI change.

    Returns:
        A dictionary structured by option_key, containing 'latest_oi', 'latest_oi_timestamp',
        and 'diff_Xm' for each interval X.
    """
    oi_differences_report = {}
    # Use a consistent, timezone-aware current time for all calculations in this batch
    current_processing_time = datetime.now(timezone.utc)
    logging.debug(f"Calculating OI differences based on current time: {current_processing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for option_key, candles_list in raw_historical_data_store.items():
        oi_differences_report[option_key] = {}
        
        latest_oi, latest_oi_timestamp = None, None
        if candles_list:
            # Candles are sorted oldest to newest by API; the last one is the latest.
            latest_candle = candles_list[-1]
            latest_oi = latest_candle.get('oi')
            latest_oi_timestamp = latest_candle.get('date') # This is a datetime object
        
        oi_differences_report[option_key]['latest_oi'] = round(int(latest_oi)/100000,2)
        oi_differences_report[option_key]['latest_oi_timestamp'] = latest_oi_timestamp

        # If latest_oi is None (e.g., no data for the contract), cannot calculate differences
        if latest_oi is None:
            for interval in intervals_min:
                oi_differences_report[option_key][f'abs_diff_{interval}m'] = None
                oi_differences_report[option_key][f'pct_diff_{interval}m'] = None
            continue # Move to the next option contract

        # Calculate OI at different past intervals
        for interval in intervals_min:
            target_past_time = current_processing_time - timedelta(minutes=interval)
            
            past_oi = find_oi_at_timestamp(
                candles_list,
                target_past_time,
                latest_oi_and_time=(latest_oi, latest_oi_timestamp) # Pass current latest OI info
            )
            
            abs_oi_diff = None
            pct_oi_change = None
            if past_oi is not None:
                abs_oi_diff = latest_oi - past_oi
                if past_oi != 0: # Avoid division by zero for percentage change
                    pct_oi_change = (abs_oi_diff / past_oi) * 100
                # else: pct_oi_change remains None if past_oi is 0 but abs_oi_diff is not (Infinite change)
            else:
                logging.debug(f"Could not find past OI for {option_key} at {interval}m prior ({target_past_time.strftime('%H:%M:%S %Z')}). abs_oi_diff and pct_oi_change will be None.")
            
            oi_differences_report[option_key][f'abs_diff_{interval}m'] = abs_oi_diff
            oi_differences_report[option_key][f'pct_diff_{interval}m'] = pct_oi_change
            
    logging.debug("OI differences calculation complete.")
    return oi_differences_report

def live_plotting_loop(fig, ax, raw_historical_data_store: dict, contract_details: dict, current_atm_strike: float, 
                            strike_step: int, num_strikes_each_side: int):
    
    try:
        global OI_diff_between_PE_and_CE_df
        # Use a consistent current time
        current_processing_time = datetime.now(timezone.utc)
        logging.debug(
                f"Plotting CE vs PE OIs at: {current_processing_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )

        # Create interactive plot
        fig1 = go.Figure()

        # Collect CE/PE dataframes
        dataframes = {}
        strikes = {}
        for option_key, candles_list in raw_historical_data_store.items():
            df = pd.DataFrame(candles_list)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            dataframes[option_key] = df
            strikes[option_key]= contract_details[option_key]['strike']

        ax.clear()

        # --- Plot ATM ---
        if "atm_ce" in dataframes and "atm_pe" in dataframes:
            
            ax.plot(dataframes["atm_ce"].index, dataframes["atm_ce"]["oi"], 
                    label=strikes["atm_ce"], color="red")
            ax.plot(dataframes["atm_pe"].index, dataframes["atm_pe"]["oi"], 
                    label=strikes["atm_pe"], color="green")
            # ATM CE
            fig1.add_trace(go.Scatter(
                x=dataframes["atm_ce"].index, 
                y=dataframes["atm_ce"]["oi"], 
                mode="lines+markers",
                name=strikes["atm_ce"],
                line=dict(color="red"),
                hovertemplate="Time: %{x}<br>OI CE: %{y}<extra></extra>"
            ))
            # ATM PE
            fig1.add_trace(go.Scatter(
            x=dataframes["atm_pe"].index, 
            y=dataframes["atm_pe"]["oi"], 
            mode="lines+markers",
            name=strikes["atm_pe"],
            line=dict(color="green"),
            hovertemplate="Time: %{x}<br>OI PE: %{y}<extra></extra>"
            ))

        # --- Plot OTM +1 ---
        if "otm1_ce" in dataframes and "otm1_pe" in dataframes:
            ax.plot(dataframes["otm1_ce"].index, dataframes["otm1_ce"]["oi"], 
                    label=strikes["otm1_ce"], linestyle="--", color="red")
            ax.plot(dataframes["otm1_pe"].index, dataframes["otm1_pe"]["oi"], 
                    label=strikes["otm1_pe"], linestyle="--", color="green")

            # OTM1 CE
            fig1.add_trace(go.Scatter(
                x=dataframes["otm1_ce"].index, 
                y=dataframes["otm1_ce"]["oi"], 
                mode="lines+markers",
                name=strikes["otm1_ce"],
                line=dict(color="coral", dash="dash"),
                hovertemplate="Time: %{x}<br>OI CE: %{y}<extra></extra>"
            ))
            # OTM1 PE
            fig1.add_trace(go.Scatter(
                x=dataframes["otm1_pe"].index, 
                y=dataframes["otm1_pe"]["oi"], 
                mode="lines+markers",
                name=strikes["otm1_pe"],
                line=dict(color="chartreuse", dash="dash"),
                hovertemplate="Time: %{x}<br>OI PE: %{y}<extra></extra>"
            ))

        # --- Plot ITM -1 ---
        if "itm1_ce" in dataframes and "itm1_pe" in dataframes:
            ax.plot(dataframes["itm1_ce"].index, dataframes["itm1_ce"]["oi"], 
                    label=strikes["itm1_ce"], linestyle=":", color="red")
            ax.plot(dataframes["itm1_pe"].index, dataframes["itm1_pe"]["oi"], 
                    label=strikes["itm1_pe"], linestyle=":", color="green")
            # ITM1 CE
            fig1.add_trace(go.Scatter(
                x=dataframes["itm1_ce"].index,
                y=dataframes["itm1_ce"]["oi"],
                mode="lines+markers",
                name=strikes["itm1_ce"],
                line=dict(color="burlywood", dash="dot"),
                hovertemplate="Time: %{x}<br>OI CE: %{y}<extra></extra>"
            ))
            # ITM1 PE
            fig1.add_trace(go.Scatter(
                x=dataframes["itm1_pe"].index,
                y=dataframes["itm1_pe"]["oi"],
                mode="lines+markers",
                name=strikes["itm1_pe"],
                line=dict(color="cadetblue", dash="dot"),
                hovertemplate="Time: %{x}<br>OI PE: %{y}<extra></extra>"
            ))

        # Formatting
        ax.set_title("Live OI: CE vs PE (ATM & ±1 strikes)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Open Interest")
        ax.legend()
        ax.grid(True)

        # Save the current plot as image (e.g., every refresh)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        global index_symbol
        plt.savefig(f"oi_plots/{index_symbol}/oi_plot_{timestamp}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)   # free memory
        plt.pause(1)  # refresh every second (non-blocking)
        #time.sleep(1) # optional, avoid CPU overuse
        # Layout
        
        # Interactive Plotly Layout
        fig1.update_layout(
            title="ATM CE vs PE OI (Interactive)",
            xaxis_title="Time",
            yaxis_title="Open Interest",
            template="plotly_dark"
        )
        # Save as interactive HTML
        fig1.write_html(f"oi_plots/{index_symbol}/oi_plot_{timestamp}.html")
    
    except Exception as e:
        print(f"Error in live_plotting_loop: {e}")
        logging.error(f"Error in live_plotting_loop: {e}", exc_info=True)

def set_oi_support_and_resistance(kite_obj, web_oi_data: dict, raw_historical_data_store: dict, contract_details: dict):
    """
    Sets global OI_SUPPORT and OI_RESISTANCE based on the highest OI(lastest value)
    """
    try:
        global OI_SUPPORT, OI_RESISTANCE, OI_DIFF_THRESHOLD_UPPER, OI_diff_between_PE_and_CE, OI_diff_between_PE_and_CE_PREV, \
            OI_diff_between_PE_and_CE_df, ATM_OI_DIFF_SLOPE, ATM_OI_DIFF_INTERCEPT, ATM_CE_OI_SLOPE, ATM_PE_OI_SLOPE, \
                OI_DIFF_THRESHOLD_LOWER, ATM_TRADESYMBOL, ATM_LTP, OI_DIFF_CROSS_TIME
    
        #print(raw_historical_data_store)
        #print(f"web data at ATM:\n{web_oi_data[str(contract_details['atm_pe']['strike'])][-5:]}")
        atm_pe_data = list(web_oi_data[str(contract_details['atm_pe']['strike'])]['Put OI'])
        atm_ce_data = list(web_oi_data[str(contract_details['atm_ce']['strike'])]['Call OI'])
        #print("ATM PE OI data:",atm_pe_data)
        #print("ATM CE OI data:",atm_ce_data)
        #latest_PE_OI = raw_historical_data_store["atm_pe"][-1]['oi'] if raw_historical_data_store.get("atm_pe") else None
        #latest_CE_OI = raw_historical_data_store["atm_ce"][-1]['oi'] if raw_historical_data_store.get("atm_ce") else None
        latest_PE_OI = atm_pe_data[-1] if len(atm_pe_data)>=1 else None
        latest_CE_OI = atm_ce_data[-1] if len(atm_ce_data)>=1 else None

        try:
            #ATM_PE_OI_PREV = raw_historical_data_store["atm_pe"][-2]['oi'] if raw_historical_data_store.get("atm_pe") else None
            #ATM_CE_OI_PREV = raw_historical_data_store["atm_ce"][-2]['oi'] if raw_historical_data_store.get("atm_ce") else None
            ATM_PE_OI_PREV = atm_pe_data[-2] if len(atm_pe_data)>=2 else None
            ATM_CE_OI_PREV = atm_ce_data[-2] if len(atm_ce_data)>=2 else None
        except:
            ATM_PE_OI_PREV = 0
            ATM_CE_OI_PREV = 0
        #if latest_PE_OI is None or latest_CE_OI is None:
        #   logging.warning("Cannot set OI_SUPPORT/RESISTANCE: Missing latest OI for ATM PE or CE.")
        #    return

        OI_diff_between_PE_and_CE = round(abs(latest_PE_OI - latest_CE_OI), 2)
        ATM_OI_DIFF_PREV = round(abs(ATM_PE_OI_PREV - ATM_CE_OI_PREV), 2) if ATM_PE_OI_PREV and ATM_CE_OI_PREV else 0
        #OI_diff_between_PE_and_CE = abs(int(pe_oi_dict.get('atm_pe', 0)) - int(ce_oi_dict.get('atm_ce', 0)))
        print("CE OI:",latest_CE_OI," PE OI:",latest_PE_OI, \
              " OI diff:",OI_diff_between_PE_and_CE, " Prev OI diff:",ATM_OI_DIFF_PREV)

        if OI_diff_between_PE_and_CE > 0:
            # Collect CE/PE dataframes
            dataframes = {}
            strikes = {}
            for option_key, candles_list in raw_historical_data_store.items():
                #if option_key not in ["atm_ce","atm_pe"]:
                #    continue
                df = pd.DataFrame(candles_list)
                if df.empty:
                    continue
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                dataframes[option_key] = df
                strikes[option_key]= contract_details[option_key]['strike']
            #for option_key, candles_list in raw_historical_data_store.items():
                

            #dataframes["atm_ce"] = dataframes["itm1_ce"]
            #dataframes["atm_pe"] = dataframes["itm1_pe"]
            #print ("atm_ce tail 5:\n",dataframes["atm_ce"].tail())
            #print ("atm_pe tail 5:\n",dataframes["atm_pe"].tail())
            #start = pd.Timestamp("2025-09-18 12:33:00", tz="Asia/Kolkata")
            #end   = pd.Timestamp("2025-09-18 12:37:00", tz="Asia/Kolkata")
            #print ("atm_ce tail 5:\n",dataframes["atm_ce"].loc[start:end])
            #print ("atm_pe tail 5:\n",dataframes["atm_pe"].loc[start:end])

            #filtered_df = OI_diff_between_PE_and_CE_df.loc[start:end]
            dataframes["atm_pe"]["oi"] = (dataframes["atm_pe"]["oi"]/100000).round(2)
            dataframes["atm_ce"]["oi"] = (dataframes["atm_ce"]["oi"]/100000).round(2)
            OI_diff_between_PE_and_CE_df = abs(dataframes["atm_pe"]["oi"] - dataframes["atm_ce"]["oi"])
            #print(">>>OI diff df>>:",OI_diff_between_PE_and_CE_df)
            #filtered_df = OI_diff_between_PE_and_CE_df.loc[start:end]
            
            latest_timestamp = web_oi_data[str(contract_details['atm_pe']['strike'])]['date'].iloc[-1]
            # Check if the Series is empty or the new timestamp is newer than the last one
            if OI_diff_between_PE_and_CE_df.empty or latest_timestamp > OI_diff_between_PE_and_CE_df.index[-1]:
                print("Appending new OI diff to df")
                OI_diff_between_PE_and_CE_df.at[latest_timestamp] = OI_diff_between_PE_and_CE
            filtered_df = OI_diff_between_PE_and_CE_df.tail(3) #last 15 minutes data
            print(">>>OI diff>>:",filtered_df)
            print(web_oi_data[str(contract_details['atm_pe']['strike'])].tail(1))

            # Calculate slope of OI difference over last 15 minutes
            ATM_OI_DIFF_SLOPE, ATM_OI_DIFF_INTERCEPT = calculate_slope(filtered_df)
            #ATM_OI_DIFF_SLOPE = round(ATM_OI_DIFF_SLOPE/1000000, 2)
            #ATM_CE_OI_SLOPE = calculate_slope(atm_ce_oi)
            #ATM_PE_OI_SLOPE = calculate_slope(atm_pe_oi)
            print("Slope:", ATM_OI_DIFF_SLOPE)
        
        # If there is no item named contract_details["atm_pe"]['strike'] in OI_diff_between_PE_and_CE_PREV{}, initialize
        try:
            if str(contract_details["atm_pe"]['strike']) not in OI_diff_between_PE_and_CE_PREV:
                OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])] = [0, 0]
        except:
            pass
        
        print ("OI_diff_between_PE_and_CE_PREV",OI_diff_between_PE_and_CE_PREV)
        # Check if there is sudden change in OI diff. If yes, send a message using telegram bot
        #if OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])][0]!=0 and \
        if ATM_OI_DIFF_PREV != OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])][0] or  \
            OI_diff_between_PE_and_CE != OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])][1]:
            #ATM_OI_DIFF_PREV != OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])]:
            print("### Inside sudden change check ###")

            # Calculate percentage change in OI diff
            OI_diff_change = 0
            if ATM_OI_DIFF_PREV != 0:
                OI_diff_change = (OI_diff_between_PE_and_CE - ATM_OI_DIFF_PREV) * 100 / ATM_OI_DIFF_PREV
            print(f"OI diff change: {OI_diff_change}%")
            
            if abs(OI_diff_change) > 2:  # check magnitude, but keep sign in value
                pe_inst = f"{EXCHANGE_NFO_OPTIONS}:{contract_details["atm_pe"]['tradingsymbol']}"
                pe_ltp = kite_obj.ltp(pe_inst)[pe_inst]['last_price']
                ce_inst = f"{EXCHANGE_NFO_OPTIONS}:{contract_details["atm_ce"]['tradingsymbol']}"
                ce_ltp = kite_obj.ltp(ce_inst)[ce_inst]['last_price']
                message = (f"⚠️ Sudden OI Change Alert! ⚠️\n\n"
                f"ATM {contract_details["atm_pe"]['strike']} CE OI: {latest_CE_OI}\n"
                f"ATM {contract_details["atm_pe"]['strike']} PE OI: {latest_PE_OI}\n"
                f"OI Difference: {OI_diff_between_PE_and_CE} \n"
                f"Previous OI Diff: {ATM_OI_DIFF_PREV}\n"
                f"(Change: {OI_diff_change:+.2f}%)\n"
                f"ATM {ce_inst} LTP: {ce_ltp}\n"
                f"ATM {pe_inst} LTP: {pe_ltp}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                tgram_bot = TelegramBot()
                response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS)
            #OI_diff_between_PE_and_CE_PREV = OI_diff_between_PE_and_CE
            OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])][0] = ATM_OI_DIFF_PREV
            OI_diff_between_PE_and_CE_PREV[str(contract_details["atm_pe"]['strike'])][1] = OI_diff_between_PE_and_CE
            print("OI_diff_between_PE_and_CE_PREV updated:",OI_diff_between_PE_and_CE_PREV)

        #plot OI_diff_between_PE_and_CE_df
        # Plotting (non-blocking, in background)
        global PLOT_PRINT_TIME, PLOT_PRINT_INTERVAL
        curr_time = time.time()
        elapsed = curr_time - PLOT_PRINT_TIME
        if elapsed > PLOT_PRINT_INTERVAL:

            import matplotlib.pyplot as plt1
            
            # Prepare x-axis as numeric for regression line
            x_vals = np.arange(len(filtered_df))  # convert time index to sequential numbers
            y_vals = ATM_OI_DIFF_SLOPE * x_vals + ATM_OI_DIFF_INTERCEPT
            plt1.figure(figsize=(10,5))

            #plt1.plot(filtered_df.index, filtered_df, marker='o')
            # Plot actual data
            plt1.plot(filtered_df.index, filtered_df, marker='o', label="OI Diff (PE - CE)")

            # Plot regression line (align with index)
            plt1.plot(filtered_df.index, y_vals, color="red", linestyle="--", label=f"Slope={ATM_OI_DIFF_SLOPE:.2f}")

            plt1.title("OI Difference between ATM PE and CE (Last 15 minutes)")
            plt1.xlabel("Time")
            plt1.ylabel("OI Difference (PE - CE)")
            plt1.legend()
            plt1.grid(True)
            plt1.tight_layout()
            plt1.savefig(f"oi_plots/{index_symbol}/oi_diff_pe_ce_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
            plt1.close() # free memory

        # Estimate time to OI diff cross zero
        OI_DIFF_CROSS_TIME = estimate_oi_diff_cross_time()
        print(f"Time to OI diff cross zero: {OI_DIFF_CROSS_TIME} minutes")

        if latest_PE_OI > latest_CE_OI:
            if OI_DIFF_CROSS_TIME and OI_DIFF_CROSS_TIME < 15:
                OI_RESISTANCE = contract_details["atm_pe"]['strike']
                OI_SUPPORT = contract_details["itm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_pe"]['tradingsymbol']
            else:
                OI_SUPPORT = contract_details["atm_pe"]['strike']
                OI_RESISTANCE = contract_details["otm1_ce"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_ce"]['tradingsymbol']
        else:
            if OI_DIFF_CROSS_TIME and OI_DIFF_CROSS_TIME < 15:
                OI_SUPPORT = contract_details["atm_ce"]['strike']
                OI_RESISTANCE = contract_details["otm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_ce"]['tradingsymbol']
            else:
                OI_RESISTANCE = contract_details["atm_ce"]['strike']
                OI_SUPPORT = contract_details["itm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_pe"]['tradingsymbol']

        """ if ATM_CE_OI_SLOPE < -800000:
            OI_DIFF_THRESHOLD_LOWER = 4000000
        elif ATM_CE_OI_SLOPE < -600000:
            OI_DIFF_THRESHOLD_LOWER = 3000000
        elif ATM_CE_OI_SLOPE < -400000:
            OI_DIFF_THRESHOLD_LOWER = 2000000

        if latest_PE_OI > latest_CE_OI:
            if ATM_OI_DIFF_SLOPE < -200000 and OI_diff_between_PE_and_CE<OI_DIFF_THRESHOLD_LOWER:
                OI_RESISTANCE = contract_details["atm_pe"]['strike']
                OI_SUPPORT = contract_details["itm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_pe"]['tradingsymbol']
            else:
                OI_SUPPORT = contract_details["atm_pe"]['strike']
                OI_RESISTANCE = contract_details["otm1_ce"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_ce"]['tradingsymbol']
        else:
            if ATM_OI_DIFF_SLOPE < -200000 and OI_diff_between_PE_and_CE<OI_DIFF_THRESHOLD_LOWER:
                OI_SUPPORT = contract_details["atm_ce"]['strike']
                OI_RESISTANCE = contract_details["otm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_ce"]['tradingsymbol']
            else:
                OI_RESISTANCE = contract_details["atm_ce"]['strike']
                OI_SUPPORT = contract_details["itm1_pe"]['strike']
                ATM_TRADESYMBOL = contract_details["atm_pe"]['tradingsymbol']
             """

        instrument = f"{EXCHANGE_NFO_OPTIONS}:{ATM_TRADESYMBOL}"
        ATM_LTP = kite_obj.ltp(instrument)[instrument]['last_price']
        """ if not OI_RESISTANCE:
            OI_RESISTANCE = contract_details["otm1_ce"]['strike']
        if not OI_SUPPORT:
            OI_SUPPORT = contract_details["itm1_pe"]['strike'] """

        print(f"OI_SUPPORT: {OI_SUPPORT}, OI_RESISTANCE: {OI_RESISTANCE}, OI_diff_between_PE_and_CE: {OI_diff_between_PE_and_CE}")
        print(f"ATM_TRADESYMBOL: {ATM_TRADESYMBOL}, ATM_LTP: {ATM_LTP}")

    except Exception as er:
        logging.error(f"Error in set_oi_support_and_resistance: {er}", exc_info=True)
        print(f"Error in set_oi_support_and_resistance. Check logs for details:{traceback.format_exc()} <<<")

def calculate_premium_differences(premium_data: dict, intervals_min: tuple):
    """
    Calculates % change in LTP over given intervals.

    Returns:
        Dict: { option_key: { 'latest_ltp': float, 'pct_diff_5m': float, ... } }
    """
    premium_diff_report = {}
    now_utc = datetime.now(timezone.utc)

    for key, candles in premium_data.items():
        latest_ltp = None
        latest_time = None
        if candles:
            latest = candles[-1]
            latest_ltp = latest['close']
            latest_time = latest['date']
        premium_diff_report[key] = {
            'latest_ltp': latest_ltp,
            'latest_ltp_time': latest_time
        }

        for min_val in intervals_min:
            target_time = now_utc - timedelta(minutes=min_val)
            past = None
            for candle in reversed(candles):
                if candle['date'] <= target_time:
                    past = candle['close']
                    break
            if past:
                pct_change = ((latest_ltp - past) / past) * 100 if past != 0 else None
                premium_diff_report[key][f'pct_diff_{min_val}m'] = pct_change
            else:
                premium_diff_report[key][f'pct_diff_{min_val}m'] = None

    return premium_diff_report

def _get_key_suffix(index_from_atm: int, total_options_one_side: int) -> str:
    """
    Helper function to determine the option key suffix (atm, itmX, otmX)
    based on the strike's index relative to the At-The-Money (ATM) strike.
    This mirrors the key generation logic in `get_relevant_option_details`.

    Args:
        index_from_atm: Integer representing the strike's position from ATM.
                        0 for ATM, negative for lower strikes, positive for higher strikes.
        total_options_one_side: Not directly used in current logic but kept for context.

    Returns:
        A string suffix like "atm", "itm1", "otm2".
    """
    if index_from_atm == 0:
        return "atm"
    elif index_from_atm < 0: # Strikes less than ATM
        return f"itm{-index_from_atm}" # e.g., index -1 is itm1
    else: # Strikes greater than ATM
        return f"otm{index_from_atm}"  # e.g., index 1 is otm1

def generate_options_tables(oi_report: dict, contract_details: dict, current_atm_strike: float, 
                            strike_step: int, num_strikes_each_side: int, 
                            change_intervals_list: tuple):
    """
    Generates two Rich Tables (one for Calls, one for Puts) displaying the OI analysis.
    If `current_atm_strike` is None, it returns an error Panel.

    Args:
        oi_report: Dictionary containing calculated OI data (from calculate_oi_differences).
        contract_details: Dictionary containing details of identified option contracts.
        current_atm_strike: The current At-The-Money strike. If None, an error panel is returned.
        strike_step: The difference between option strikes.
        num_strikes_each_side: Number of ITM/OTM strikes to display.
        change_intervals_list: Tuple of intervals (e.g., (10, 15, 30)) for OI change columns.

    Returns:
        A Rich Group object containing the Call and Put tables, or a Rich Panel with an error message.
    """
    if current_atm_strike is None:
        logging.error("Cannot generate tables: current_atm_strike is None.")
        return Panel("[bold red]ATM Strike could not be determined. Tables cannot be generated.[/bold red]", title="Error", border_style="red")

    time_now_str = datetime.now().strftime('%H:%M:%S') # Timestamp for table titles
    
    # Create Call options table
    ce_data = oi_report.get("atm_ce", {})
    ce_latest_oi_time = ce_data.get('latest_oi_timestamp')
    call_table_title = f"CALL Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str} \
        for {ce_latest_oi_time.strftime('%Y-%m-%d %H:%M:%S %Z') if ce_latest_oi_time else 'N/A'}"
    #call_table = Table(title=call_table_title, show_lines=True, expand=True)
    call_table = Table(title=call_table_title, show_lines=True)
    
    # Create Put options table
    pe_data = oi_report.get("atm_pe", {})
    pe_latest_oi_time = pe_data.get('latest_oi_timestamp')
    put_table_title = f"PUT Options OI ({UNDERLYING_SYMBOL} - ATM: {int(current_atm_strike)}) @ {time_now_str} \
        for {pe_latest_oi_time.strftime('%Y-%m-%d %H:%M:%S %Z') if pe_latest_oi_time else 'N/A'}"
    #put_table = Table(title=put_table_title, show_lines=True, expand=True)
    put_table = Table(title=put_table_title, show_lines=True)

    # Define common columns for both tables
    #cols = ["Strike", "Symbol", "Latest OI(Lakhs)", "OI Time"]
    cols = ["Symbol", "Current OI(Lakhs)"] # Updated column header for clarity
    for interval in change_intervals_list: # Dynamically add OI change columns
        cols.append(f"OI %Chg ({interval}m)") # Updated column header
    
    for col_name in cols:
        call_table.add_column(col_name, width=20, no_wrap=True, justify="right")
        put_table.add_column(col_name, width=20, no_wrap=True, justify="right")

    # Iterate through strike levels relative to ATM (-num to +num)
    total_call_threshold_breached = 0
    total_call_cells = 0

    total_put_threshold_breached = 0
    total_put_cells = 0
    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strike_val = current_atm_strike + (i * strike_step)
        key_suffix = _get_key_suffix(i, num_strikes_each_side) # Get "atm", "itmX", "otmX"

        # --- Populate Call Option Row ---
        option_key_ce = f"{key_suffix}_ce" # e.g., "atm_ce", "itm1_ce"
        ce_data = oi_report.get(option_key_ce, {}) # Get data for this call option
        ce_contract = contract_details.get(option_key_ce, {}) # Get contract details
        
        ce_strike_display = str(int(ce_contract.get('strike', strike_val))) # Use actual strike from contract if available
        
        # Style strike price: ATM (cyan), ITM for Calls (lower strikes - green), OTM for Calls (higher strikes - red)
        ce_strike_style = "cyan" if i == 0 else ("green" if i < 0 else "red")
        ce_strike_oi_style = "" 
        if i == 0:
            ce_strike_oi_style = "bold cyan" # Cyan for ATM, normal for ITM/OTM
        elif i == -1 or i == 1:
            ce_strike_oi_style = "bold yellow"
             
        
        ce_latest_oi = ce_data.get('latest_oi')

        # Store latest OI in global dictionary for later use
        ce_oi_dict[option_key_ce] = ce_latest_oi

        ce_latest_oi_time = ce_data.get('latest_oi_timestamp')

        # Prepare row data for call table
        """ ce_row_data = [
            Text(ce_strike_display, style=ce_strike_style),
            ce_contract.get('tradingsymbol', 'N/A'),
            Text(f"{ce_latest_oi:,}" if ce_latest_oi is not None else "N/A", style=ce_strike_oi_style), # Format OI with comma
            ce_latest_oi_time.strftime("%H:%M:%S %Z") if ce_latest_oi_time else "N/A" # Format time
        ] """
        ce_row_data = [
            Text(ce_contract.get('tradingsymbol', 'N/A'), style=ce_strike_style),
            Text(f"{ce_latest_oi:,}" if ce_latest_oi is not None else "N/A", style=ce_strike_oi_style), # Format OI with comma
        ]

        for interval in change_intervals_list: # Add OI change values
            total_call_cells = total_call_cells+1
            pct_oi_change = ce_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"
            
            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                #if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                if pct_oi_change > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                    cell_text.stylize("bold red") # Apply style if threshold exceeded
                    total_call_threshold_breached = total_call_threshold_breached+1
                elif pct_oi_change < -PCT_CHANGE_THRESHOLDS[interval]: # Check negative change against threshold
                    cell_text.stylize("bold green") # Apply style if threshold exceeded
                    total_call_threshold_breached = total_call_threshold_breached+1
            ce_row_data.append(cell_text)
        call_table.add_row(*ce_row_data)

        # --- Populate Put Option Row ---
        option_key_pe = f"{key_suffix}_pe" # e.g., "atm_pe", "itm1_pe"
        pe_data = oi_report.get(option_key_pe, {}) # Get data for this put option
        pe_contract = contract_details.get(option_key_pe, {}) # Get contract details

        pe_strike_display = str(int(pe_contract.get('strike', strike_val)))
        
        # Style strike price: ATM (cyan), ITM for Puts (higher strikes - green), OTM for Puts (lower strikes - red)
        pe_strike_style = "cyan" if i == 0 else ("green" if i > 0 else "red")
        pe_strike_oi_style = ""
        if i == 0:
            pe_strike_oi_style = "bold cyan" # Cyan for ATM, normal for ITM/OTM
        elif i == -1 or i == 1:
            pe_strike_oi_style = "bold yellow"
        
        pe_latest_oi = pe_data.get('latest_oi')

        # Store latest OI in global dictionary for later use
        pe_oi_dict[option_key_pe] = pe_latest_oi

        pe_latest_oi_time = pe_data.get('latest_oi_timestamp')

        # Prepare row data for put table
        """ pe_row_data = [
            Text(pe_strike_display, style=pe_strike_style),
            pe_contract.get('tradingsymbol', 'N/A'),
            Text(f"{pe_latest_oi:,}" if pe_latest_oi is not None else "N/A", style=pe_strike_oi_style),
            pe_latest_oi_time.strftime("%H:%M:%S %Z") if pe_latest_oi_time else "N/A"
        ] """
        pe_row_data = [
            Text(pe_contract.get('tradingsymbol', 'N/A'), style=pe_strike_style),
            Text(f"{pe_latest_oi:,}" if pe_latest_oi is not None else "N/A", style=pe_strike_oi_style), # Format OI with comma
        ]
        for interval in change_intervals_list: # Add OI percentage change values
            total_put_cells = total_put_cells+1
            pct_oi_change = pe_data.get(f'pct_diff_{interval}m')
            formatted_pct_str = f"{pct_oi_change:+.2f}%" if pct_oi_change is not None else "N/A"

            cell_text = Text(formatted_pct_str)
            if pct_oi_change is not None and interval in PCT_CHANGE_THRESHOLDS:
                #if abs(pct_oi_change) > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                if pct_oi_change > PCT_CHANGE_THRESHOLDS[interval]: # Check absolute change against threshold
                    cell_text.stylize("bold green") # Apply style if threshold exceeded
                    total_put_threshold_breached = total_put_threshold_breached+1
                elif pct_oi_change < -PCT_CHANGE_THRESHOLDS[interval]: # Check negative change against threshold
                    cell_text.stylize("bold red") # Apply style if threshold exceeded
                    total_put_threshold_breached = total_put_threshold_breached+1
            pe_row_data.append(cell_text)
        put_table.add_row(*pe_row_data)
    if (float(total_put_threshold_breached)/float(total_put_cells) > 0.5) or (float(total_call_threshold_breached)/float(total_call_cells) > 0.5):
        os.system('afplay /Users/vibhu/zd/siren-alert-96052.mp3')    


    return Group(call_table, put_table) # Group tables for simultaneous display in Live

def generate_premium_change_table(premium_report: dict, contract_details: dict, current_atm_strike: float,
                                   strike_step: int, num_strikes_each_side: int, intervals_min: tuple):
    """
    Generates two Rich tables (Calls & Puts) displaying premium % change.

    Returns:
        Group of Rich Tables
    """
    time_now_str = datetime.now().strftime('%H:%M:%S')

    ce_table = Table(title=f"CALL Premium Change % (ATM: {int(current_atm_strike)}) @ {time_now_str}", show_lines=True)
    pe_table = Table(title=f"PUT Premium Change % (ATM: {int(current_atm_strike)}) @ {time_now_str}", show_lines=True)

    headers = ["Symbol", "Last Premium"]
    for interval in intervals_min:
        headers.append(f"%Chg ({interval}m)")

    for h in headers:
        ce_table.add_column(h, justify="right", width=15, no_wrap=True)
        pe_table.add_column(h, justify="right", width=15, no_wrap=True)

    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        key_suffix = _get_key_suffix(i, num_strikes_each_side)

        for option_type in ["ce", "pe"]:
            option_key = f"{key_suffix}_{option_type}"
            report = premium_report.get(option_key, {})
            contract = contract_details.get(option_key, {})

            symbol = contract.get('tradingsymbol', 'N/A')
            latest_ltp = report.get('latest_ltp')
            row = [
                Text(symbol),
                Text(f"{latest_ltp:.2f}" if latest_ltp is not None else "N/A")
            ]

            for interval in intervals_min:
                pct = report.get(f'pct_diff_{interval}m')
                formatted = f"{pct:+.2f}%" if pct is not None else "N/A"
                cell = Text(formatted)
                if pct is not None:
                    if pct >= 10:
                        cell.stylize("bold green")
                    elif pct <= -10:
                        cell.stylize("bold red")
                row.append(cell)

            if option_type == "ce":
                ce_table.add_row(*row)
            else:
                pe_table.add_row(*row)

    return Group(ce_table, pe_table)

def run_analysis_iteration(kite_conn: KiteConnect, oi_tracker: object, nfo_instr: list, nearest_exp_date: date, symbol_prefix: str):
    """
    Performs one complete iteration of fetching data, calculating differences, and generating tables.
    This function is called repeatedly by the live update loop.

    Args:
        kite_conn: Initialized KiteConnect object.
        nfo_instr: List of all NFO instruments (fetched once at startup).
        nearest_exp_date: The nearest weekly expiry date (determined once at startup).

    Returns:
        A Rich Group object containing the Call and Put tables for display,
        or a Rich Panel with an error/warning message if issues occur.
    """
    try:
        global OI_SUPPORT, OI_RESISTANCE, OI_diff_between_PE_and_CE
        logging.debug("Starting new analysis iteration.")
        # 1. Get current ATM strike
        current_atm_strike, ltp = get_atm_strike(kite_conn, UNDERLYING_SYMBOL, EXCHANGE_LTP, STRIKE_DIFFERENCE)

        if not current_atm_strike: # Critical if ATM cannot be determined for this iteration
            logging.error("Could not determine ATM strike for this iteration.")
            return Panel("[bold red]Error: Could not determine ATM strike. Check logs. Waiting for next refresh.[/bold red]", title="Update Error", border_style="red")
        
        # This check should ideally be redundant if main() ensures nearest_exp_date is valid before starting loop
        if not nearest_exp_date:
             logging.error("Nearest expiry date is not available (should not happen if pre-checked).")
             return Panel("[bold red]Error: Nearest expiry date not available. Critical error.[/bold red]", title="Update Error", border_style="red")

        # 2. Identify relevant option contracts around the new ATM strike
        option_contract_details = get_relevant_option_details(
            nfo_instr, current_atm_strike, nearest_exp_date,
            STRIKE_DIFFERENCE, OPTIONS_COUNT, UNDERLYING_PREFIX, symbol_prefix
        )
        
        # If no contracts are found (e.g., due to market close or issues with instrument list for that ATM)
        if not option_contract_details:
            logging.warning(f"Could not retrieve relevant option contracts for ATM {int(current_atm_strike)}.")
            return Panel(f"[bold yellow]Warning: Could not retrieve relevant option contracts for ATM {int(current_atm_strike)}. Waiting for next refresh.[/bold yellow]", title="Update Warning", border_style="yellow")

        # 3. Fetch historical OI data for these contracts
        raw_historical_oi_data = fetch_historical_oi_data(kite_conn, option_contract_details)
        
        # 4. Fetch OI data directly from web
        status, web_oi_data = oi_tracker.run_once(option_contract_details)
        #print(f"Status:{status}, Web OI data:{web_oi_data}")
        
        # Set global OI_SUPPORT and OI_RESISTANCE based on latest OI values
        set_oi_support_and_resistance(kite_conn, web_oi_data, raw_historical_oi_data, option_contract_details)

        # When price reach near or do fake breakout on support or resistance, send signal to telegram bot
        telegram_bot_send_signal(ltp=ltp)

        # Plotting (non-blocking, in background)
        global PLOT_PRINT_TIME, PLOT_PRINT_INTERVAL
        curr_time = time.time()
        elapsed = curr_time - PLOT_PRINT_TIME
        if elapsed > PLOT_PRINT_INTERVAL:
            logging.warning(f"Data fetch took too long ({elapsed:.2f}s) exceeding refresh interval. Skipping plotting for this iteration.")
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots(figsize=(12,6))
            live_plotting_loop(fig, ax, raw_historical_oi_data, option_contract_details, current_atm_strike, STRIKE_DIFFERENCE, OPTIONS_COUNT)
            PLOT_PRINT_TIME = time.time()

        # 4. Calculate OI differences
        oi_change_data = calculate_oi_differences(raw_historical_oi_data, OI_CHANGE_INTERVALS_MIN)
        
        # 5. Generate Rich tables for display
        table_group = generate_options_tables(
            oi_change_data, option_contract_details, current_atm_strike, 
            STRIKE_DIFFERENCE, OPTIONS_COUNT, OI_CHANGE_INTERVALS_MIN
        )
        logging.debug("Analysis iteration completed successfully.")
        return table_group

    except Exception as e: # Catch any other unexpected errors during the iteration
        logging.error(f"Exception during analysis iteration: {e}", exc_info=True)
        print(f"Exception during analysis iteration. Check logs for details:{traceback.format_exc()} <<<")
        return Panel(f"[bold red]An error occurred during data refresh: {e}. Check logs.[/bold red]", title="Update Error", border_style="red")

def telegram_bot_send_table(table):
    """
    Sends a Rich Table to a Telegram bot using the configured bot token and chat ID.
    The table is converted to HTML format for Telegram compatibility.

    Args:
        table: A Rich Table object to be sent.

    Returns:
        None
    """
    try:
        html_render = console.export_html(inline_styles=True)
        hti = Html2Image(output_path=".", custom_flags=["--no-sandbox"])  # Important for Linux
        html_file = "rich_table.html"
        image_file = "rich_table.png"

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_render)

        hti.screenshot(html_file=html_file, save_as=image_file, size=(1500, 1300),)

        # Step 4: Send image to Telegram
        BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
        CHAT_IDS = config.CHAT_IDS

        for id in CHAT_IDS:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            files = {'photo': open(image_file, 'rb')}
        
            data = {f"chat_id": id, "caption": f"Live OI Tracker Update for {UNDERLYING_SYMBOL}"}
            response = requests.post(url, files=files, data=data)

        # Optional cleanup
        os.remove(html_file)
        os.remove(image_file)

        print("Image sent to Telegram." if response.ok else "Failed to send image.")

    except Exception as e:
        logging.error(f"Error sending table to Telegram: {e}", exc_info=True)

def estimate_oi_diff_cross_time():
    """
    Estimate the time when OI difference between PE and CE will cross zero based on current slope and intercept.
    """
    time_to_cross = 9999  # Default large value indicating no estimate
    # Hyperparameters
    T_min = 5
    T_max = 25
    k = 0.035
    try:
        if ATM_OI_DIFF_SLOPE < 0: # Only estimate if slope is negative (decreasing)
            time_to_cross = - OI_diff_between_PE_and_CE/ ATM_OI_DIFF_SLOPE  # in minutes
    except Exception as e:
        logging.error(f"Error estimating OI diff cross time: {e}", exc_info=True)
        print(f"Error estimating OI diff cross time. \nError:{traceback.format_exc()} <<<")
    return round(time_to_cross, 1)

def telegram_bot_send_signal(ltp=None):
    """
    Check if the ltp has broken the support or resistance and send a signal to telegram bot
    """
    global OI_SUPPORT, OI_RESISTANCE, OI_diff_between_PE_and_CE, ATM_OI_DIFF_SLOPE, OI_DIFF_THRESHOLD_UPPER, \
        OI_DIFF_THRESHOLD_LOWER, FAKE_BREAKOUT_THRESHOLD, ATM_TRADESYMBOL, ATM_LTP, \
        MESSAGE_TRIGGERED, MESSAGE_TRIGGER_TIMER, PREVIOUS_TRADESYMBOL, OI_DIFF_CROSS_TIME
    try:
        """ print("FAKE_BREAKOUT_THRESHOLD:",FAKE_BREAKOUT_THRESHOLD)
        print("ltp:",ltp)
        print("OI_SUPPORT:",OI_SUPPORT)
        print("OI_RESISTANCE:",OI_RESISTANCE)
        print("<15 support",(-FAKE_BREAKOUT_THRESHOLD<ltp-int(OI_RESISTANCE)<FAKE_BREAKOUT_THRESHOLD))
        print("<15 resistance:",(-FAKE_BREAKOUT_THRESHOLD<int(OI_SUPPORT)-ltp<FAKE_BREAKOUT_THRESHOLD))
        print(">15 supprt:", (-FAKE_BREAKOUT_THRESHOLD<ltp-int(OI_SUPPORT)<FAKE_BREAKOUT_THRESHOLD))
        print(">15 resistance:", (-FAKE_BREAKOUT_THRESHOLD<int(OI_RESISTANCE)-ltp<FAKE_BREAKOUT_THRESHOLD))
        print("(str(OI_SUPPORT) in str(ATM_TRADESYMBOL)):", (str(OI_SUPPORT) in str(ATM_TRADESYMBOL)))
        print("(str(OI_RESISTANCE) in str(ATM_TRADESYMBOL)):", (str(OI_RESISTANCE) in str(ATM_TRADESYMBOL)))
        print("OI_DIFF_CROSS_TIME:",OI_DIFF_CROSS_TIME) """
        message = None
        if int(OI_DIFF_CROSS_TIME) > 15:
            #print("OI_DIFF_CROSS_TIME>15")
            if (str(OI_SUPPORT) in str(ATM_TRADESYMBOL)) and (-FAKE_BREAKOUT_THRESHOLD<ltp-int(OI_SUPPORT)<FAKE_BREAKOUT_THRESHOLD):
                #print("1")
                message = (f"🚨 Index value nearing or below support {OI_SUPPORT} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"with slope {ATM_OI_DIFF_SLOPE:.2f}/min \n")
            elif (str(OI_RESISTANCE) in str(ATM_TRADESYMBOL)) and (-FAKE_BREAKOUT_THRESHOLD<int(OI_RESISTANCE)-ltp<FAKE_BREAKOUT_THRESHOLD):
                #print("2")
                message = (f"🚨 Index value nearing or above resistance {OI_RESISTANCE} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"with slope {ATM_OI_DIFF_SLOPE:.2f}/min \n")
        elif int(OI_DIFF_CROSS_TIME) < 15:
            #print("OI_DIFF_CROSS_TIME<15")
             # Check if the ATM_TRADESYMBOL contains the OI_SUPPORT or OI_RESISTANCE value as substring
            if (str(OI_RESISTANCE) in ATM_TRADESYMBOL) and (-FAKE_BREAKOUT_THRESHOLD<ltp-int(OI_RESISTANCE)<FAKE_BREAKOUT_THRESHOLD):
                #print("3")
                message = (f"⚠️ Index value nearing or above new forming resistance {OI_RESISTANCE} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"OI Difference decreasing for last 5mins {ATM_OI_DIFF_SLOPE:.2f}/min \n")
            elif (str(OI_SUPPORT) in ATM_TRADESYMBOL) and (-FAKE_BREAKOUT_THRESHOLD<int(OI_SUPPORT)-ltp<FAKE_BREAKOUT_THRESHOLD):
                #print("4")
                message = (f"⚠️ Index value nearing or below new forming support {OI_SUPPORT} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"OI Difference decreasing for last 5mins {ATM_OI_DIFF_SLOPE:.2f}/min \n")

        """ if ltp-int(OI_SUPPORT)<FAKE_BREAKOUT_THRESHOLD and OI_diff_between_PE_and_CE>OI_DIFF_THRESHOLD_UPPER:
            if ATM_OI_DIFF_SLOPE>-120000:
                message = (f"🚨 Index value nearing or below support {OI_SUPPORT} for {UNDERLYING_SYMBOL} at {ltp}\n")
            elif ATM_OI_DIFF_SLOPE<-200000 and OI_diff_between_PE_and_CE<OI_DIFF_THRESHOLD_LOWER:
                message = (f"⚠️ Index value nearing or above resistance {OI_RESISTANCE} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"OI Difference decreasing for last 15mins {ATM_OI_DIFF_SLOPE:.2f}\n")
                
        elif int(OI_RESISTANCE)-ltp<FAKE_BREAKOUT_THRESHOLD and OI_diff_between_PE_and_CE>OI_DIFF_THRESHOLD_UPPER:
            if ATM_OI_DIFF_SLOPE>-120000:
                message = (f"🚨 Index value nearing or above resistance {OI_RESISTANCE} for {UNDERLYING_SYMBOL} at {ltp}\n")
            elif ATM_OI_DIFF_SLOPE<-200000 and OI_diff_between_PE_and_CE<OI_DIFF_THRESHOLD_LOWER:
                message = (f"⚠️ Index value nearing or below support {OI_SUPPORT} for {UNDERLYING_SYMBOL} at {ltp}\n"
                           f"OI Difference decreasing for last 15mins {ATM_OI_DIFF_SLOPE:.2f}\n")
         """
        
        cooldown = 300  # 5 min
        if message:
            message += (
            f"Estimated time to OI diff cross zero: {OI_DIFF_CROSS_TIME} minutes\n"
            f"📉 PE OI: {pe_oi_dict.get('atm_pe', 'N/A')}\n"
            f"📈 CE OI: {ce_oi_dict.get('atm_ce', 'N/A')}\n"
            f"💡 Consider buying {ATM_TRADESYMBOL}, current premium at {ATM_LTP}"
            f"\n\n[This is an automated message. Please check charts before trading.]")
            #print("message:",message)
            now = time.time()
            date = datetime.now().replace(microsecond=0).isoformat()  # ISO format with timezone if needed
            send_the_message = False
            if ATM_TRADESYMBOL == PREVIOUS_TRADESYMBOL:
                if not MESSAGE_TRIGGERED or (now - MESSAGE_TRIGGER_TIMER > cooldown):
                    # Send signal to Telegram
                    send_the_message = True                
            else:
                send_the_message = True

            if send_the_message:
                # Prepare row as dictionary
                row = {
                    "index_symbol": index_symbol,
                    "datetime": date,
                    "tradingsymbol": ATM_TRADESYMBOL,
                    "ltp": ATM_LTP
                }

                # File path
                file_path = "entry_signals.csv"

                # Write or append
                df = pd.DataFrame([row])
                if not os.path.exists(file_path):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_csv(file_path, mode="a", header=False, index=False)
                tgram_bot = TelegramBot()
                response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS)
                PREVIOUS_TRADESYMBOL = ATM_TRADESYMBOL
                MESSAGE_TRIGGER_TIMER = now
                MESSAGE_TRIGGERED = True


    except Exception as e:
        print(f"Error in telegram_bot_send_signal. \nError:{traceback.format_exc()} <<<")
        logging.error(f"Error sending signal to Telegram: {e}", exc_info=True)

def calculate_pcr_for_near_strikes(num_strikes_each_side, pe_oi_data, ce_oi_data):
    """
    Calculate PCR for ATM, 1 strike above, and 1 strike below.

    :param option_data: DataFrame with columns ['strike', 'CE_OI', 'PE_OI']
    :param fut_price: Current futures price
    :return: dict with PCR values for 'ATM', 'ATM+1', 'ATM-1'
    """
    ce_oi_sum = 0
    pe_oi_sum = 0
    for i in range(-num_strikes_each_side, num_strikes_each_side+1):
        key_suffix = _get_key_suffix(i, num_strikes_each_side) # Get "atm", "itmX", "otmX"

        # --- Populate Call Option Row ---
        option_key_ce = f"{key_suffix}_ce" # e.g., "atm_ce", "itm1_ce"
        ce_oi_sum += int(ce_oi_data[option_key_ce])
        option_key_pe = f"{key_suffix}_pe" # e.g., "atm_pe", "itm1_pe"
        pe_oi_sum += int(pe_oi_data[option_key_pe])
    
    if ce_oi_sum != 0:
        pcr = round((pe_oi_sum / ce_oi_sum), 2)  # Calculate PCR
        if pcr < 1:
            console.print(f"[bold red]PCR for {UNDERLYING_SYMBOL} - {num_strikes_each_side} strikes (CE: {ce_oi_sum}, PE: {pe_oi_sum}) = {pcr:.2f}[/bold red]")
        else:
            console.print(f"[bold green]PCR for {UNDERLYING_SYMBOL} - {num_strikes_each_side} strikes (CE: {ce_oi_sum}, PE: {pe_oi_sum}) = {pcr:.2f}[/bold green]")
    else:
        pcr = None
        console.print(f"[bold red]PCR for {UNDERLYING_SYMBOL} (CE: {ce_oi_sum}, PE: {pe_oi_sum}) = Undefined (CE OI is zero)[/bold red]")

def main():
    """
    Main function to run the OI Tracker script.
    Handles initial setup (API connection, instrument fetching) and then enters the live update loop.
    """
    global index_symbol
    index_symbol = "nifty50"  # Default index symbol, can be overridden by command line argument
    if len(sys.argv) < 1:
        console.print("Usage: python oi_tracker_test.py 'nifty50', if you want. Currently using 'nifty50' as default.")

    index_symbol = sys.argv[1]

    initialize_index(index_symbol)  # Initialize index symbols if needed

    console.print(f"[bold blue]Starting OI Tracker Script (Log file: {LOG_FILE_NAME})[/bold blue]")

    # Check if API keys are default placeholders and warn user
    if api_key_to_use == API_KEY_DEFAULT or api_secret_to_use == API_SECRET_DEFAULT:
        console.print(f"[bold yellow]Warning: Using default placeholder API Key/Secret.[/bold yellow]")
        console.print(f"[yellow]Please set 'API_KEY' and 'API_SECRET' environment variables for live data.[/yellow]")
        logging.warning("Using default placeholder API Key/Secret. User prompted to set environment variables.")
        # The script might continue but will likely fail at API calls if keys are not valid.

    try:
        # --- Initial Setup ---
        # 1. User Login and Session Generation
        try:
            login_url = kite.login_url()
            console.print(f"Kite Login URL: [link={login_url}]{login_url}[/link]")
            kite.set_access_token(config.ACCESS_TOKEN)

            console.print("[bold green]Kite API session generated and access token set successfully![/bold green]")
            profile = kite.profile() # Verify connection by fetching profile
            console.print(f"[green]Successfully connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")

        except Exception as e:
            console.print(f"[bold red]Error during Kite API login: {e}[/bold red]")
            login_url = kite.login_url()
            console.print(f"Kite Login URL: [link={login_url}]{login_url}[/link]")
            request_token = console.input("[bold cyan]Enter Request Token from the above URL: [/bold cyan]").strip()
            if not request_token:
                console.print("[bold red]No request token entered. Exiting.[/bold red]")
                sys.exit(1) # Critical error, cannot proceed

            data = kite.generate_session(request_token, api_secret=api_secret_to_use)
            kite.set_access_token(data["access_token"])
            print("[INFO] Session data:", data)
            access_token = data["access_token"]
            print("[INFO] Access Token:", access_token)
            console.print("[bold green]Kite API session generated and access token set successfully![/bold green]")
        
            profile = kite.profile() # Verify connection by fetching profile
            console.print(f"[green]Successfully connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")

        # 2. Fetch NFO instruments list (done once at startup)
        console.print(f"Fetching NFO instruments list for {EXCHANGE_NFO_OPTIONS} (once)...")
        nfo_instruments = kite.instruments(EXCHANGE_NFO_OPTIONS)
        if not nfo_instruments:
            console.print(f"[bold red]Failed to fetch NFO instruments from {EXCHANGE_NFO_OPTIONS}. Exiting.[/bold red]")
            logging.critical(f"Failed to fetch NFO instruments from {EXCHANGE_NFO_OPTIONS}.")
            sys.exit(1) # Critical error
        logging.info(f"Fetched {len(nfo_instruments)} NFO instruments.")

        # 3. Determine nearest weekly expiry (done once at startup)
        # Note: If the script is run over multiple days, this expiry might become outdated.
        # For simplicity, it's fetched once. A more advanced version might re-check periodically.
        return_arr = get_nearest_weekly_expiry(nfo_instruments, UNDERLYING_PREFIX)
        nearest_expiry_date = return_arr['expiry']
        symbol_prefix = return_arr['symbol_prefix']

        if not nearest_expiry_date:
            console.print(f"[bold red]Could not determine nearest weekly expiry for {UNDERLYING_PREFIX}. Exiting.[/bold red]")
            logging.critical(f"Could not determine nearest weekly expiry for {UNDERLYING_PREFIX}.")
            sys.exit(1) # Critical error
        console.print(f"Tracking options for expiry: [bold magenta]{nearest_expiry_date.strftime('%d-%b-%Y')}[/bold magenta]")
        
        console.print(f"Starting live updates. Refresh interval: {REFRESH_INTERVAL_SECONDS} seconds. Press Ctrl+C to exit.")
        console.print(f"Underlying: [bold cyan]{UNDERLYING_SYMBOL}[/bold cyan], Strike Difference: [bold cyan]{STRIKE_DIFFERENCE}[/bold cyan], Options Count per side: [bold cyan]{OPTIONS_COUNT}[/bold cyan]")

        # --- Live Update Loop ---
        # refresh_per_second for Live is for UI animation smoothness if any;
        # auto_refresh=False means we control update timing with time.sleep()

        # Send signal to Telegram
        tgram_bot = TelegramBot()
        message = (f"🚀 START: OI Tracker started for {UNDERLYING_SYMBOL}!")
        response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS)

        # Initialize OptionChainTracker for background OI data fetching
        web_session_obj = WebDriverSession()
        web_session_obj.login_zerodha()
        web_session = web_session_obj.get_driver()
        oi_tracker = OptionChainTracker(driver=web_session, index_symbol=index_symbol)
        
        with Live(console=console, refresh_per_second=10, auto_refresh=False) as live: 
            while True:
                console.print("Entering live update loop...")
                console.print("-"*50)
                console.print("Current Time:" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                console.print("-"*50)
                logging.info("Starting new live update cycle.")
                # Perform one iteration of analysis
                display_content = run_analysis_iteration(kite, oi_tracker, nfo_instruments, nearest_expiry_date, symbol_prefix)
                # Update the live display with the new tables or error panel
                live.update(display_content, refresh=True)
                logging.info(f"Live display updated. Waiting for {REFRESH_INTERVAL_SECONDS} seconds.")
                
                # Send the table to Telegram
                #telegram_bot_send_table(display_content)
                #st.write(f"Live OI Tracker Update for {UNDERLYING_SYMBOL} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                #st.write(display_content)
                # Wait for the configured refresh interval

                calculate_pcr_for_near_strikes(PCR_STRIKE_COUNT, pe_oi_dict, ce_oi_dict)
                calculate_pcr_for_near_strikes(PCR_STRIKE_COUNT+1, pe_oi_dict, ce_oi_dict)
                calculate_pcr_for_near_strikes(PCR_STRIKE_COUNT+2, pe_oi_dict, ce_oi_dict)

                # If current time is past 3:30 PM, exit the script gracefully
                if datetime.now().time() >= time1(15, 31):
                    console.print("[bold yellow]Market close time reached (3:30 PM). Exiting script.[/bold yellow]")
                    logging.info("Market close time reached. Exiting script.")
                    break
                time.sleep(REFRESH_INTERVAL_SECONDS)

    # Handle specific Kite API exceptions during setup
    except exceptions.TokenException as te:
        console.print(f"[bold red]Token Exception: {te}. This usually means an invalid or expired request_token or session issues. Please restart and re-login.[/bold red]")
        logging.critical(f"Token Exception: {te}", exc_info=True)
        sys.exit(1)
    except exceptions.InputException as ie:
        console.print(f"[bold red]Input Error: {ie}. This could be due to an incorrect API Key/Secret or other parameters. Check configuration.[/bold red]")
        logging.critical(f"Input Exception: {ie}", exc_info=True)
        sys.exit(1)
    # Handle user interruption (Ctrl+C)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Script terminated by user (Ctrl+C).[/bold yellow]")
        logging.info("Script terminated by user.")
    # Catch any other unexpected critical errors during the main setup
    except Exception as e: 
        console.print(f"[bold red]An unexpected critical error occurred in the main setup: \
                      \n{traceback.format_exc()}[/bold red]")
        logging.critical(f"Unexpected critical error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup actions
        web_session_obj.quit()

        # This block executes whether an exception occurred or not (unless sys.exit was called)
        console.print("[bold blue]OI Tracker script finished.[/bold blue]")
        message = (f"🛑 STOP: OI Tracker script for {UNDERLYING_SYMBOL} has stopped.")
        response = tgram_bot.send_message(message, chat_id=config.CHAT_IDS)
        logging.info("oi_tracker.py script execution process ended.")

if __name__ == "__main__":
    # Entry point of the script
    main()