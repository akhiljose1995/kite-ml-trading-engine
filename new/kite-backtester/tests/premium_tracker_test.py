# Script to track option OI changes
# This script connects to the Kite API, fetches NFO option chain data for a specified underlying,
# calculates Open Interest (OI) changes over various intervals, and displays this information
# in live-updating tables in the console.
import pytest
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import os
import sys  # For sys.exit() for critical error handling
import time # For time.sleep() in the live update loop
from kiteconnect import KiteConnect
from datetime import datetime, date, timedelta, timezone
import config

# Rich library imports for enhanced terminal output
from rich.console import Console, Group
from html2image import Html2Image
import requests
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

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

# --- Trading Parameters ---
# -- NIFTY Parameters --
UNDERLYING_SYMBOL = "NIFTY 50"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
STRIKE_DIFFERENCE = 50               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
# --- Exchange Configuration ---
# Use KiteConnect attributes for exchange names
EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_NFO  # Exchange for NFO options contracts
EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50)


# --- Trading Parameters ---
# --  SENSEX Parameters --
#UNDERLYING_SYMBOL = "SENSEX"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
#STRIKE_DIFFERENCE = 100               # Difference between consecutive strikes (50 for NIFTY, 100 for BANKNIFTY)
#OPTIONS_COUNT = 5                    # Number of ITM/OTM strikes to fetch on each side of ATM (e.g., 2 means 2 ITM, 1 ATM, 2 OTM = 5 levels total)
## --- Exchange Configuration ---
## Use KiteConnect attributes for exchange names
#EXCHANGE_NFO_OPTIONS = KiteConnect.EXCHANGE_BFO  # Exchange for NFO options contracts
#EXCHANGE_LTP = KiteConnect.EXCHANGE_BSE      # Exchange for fetching LTP of the underlying (e.g., NSE for NIFTY 50)
 


# --- Data Fetching Parameters ---
HISTORICAL_DATA_MINUTES = 40         # How many minutes of historical data to fetch for OI calculation
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30) # Past intervals (in minutes) to calculate OI change from latest OI

# --- Display and Logging ---
REFRESH_INTERVAL_SECONDS = 10        # How often to refresh the data and tables (in seconds)
LOG_FILE_NAME = "oi_tracker.log"     # Name of the log file
FILE_LOG_LEVEL = "DEBUG, INFO"              # Logging level for the log file (DEBUG, INFO, WARNING, ERROR, CRITICAL)
PCT_CHANGE_THRESHOLDS = {            # Thresholds for highlighting OI % change (interval_in_min: percentage)
     5: 8.0,
    10: 10.0,  # Highlight if 10-min % change > 10%
    15: 15.0,  # Highlight if 15-min % change > 15%
    30: 25.0   # Highlight if 30-min % change > 25%
}

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
UNDERLYING_PREFIX = UNDERLYING_SYMBOL.split(" ")[0].upper()

# Retrieve API keys from environment variables or use default placeholders
api_key_to_use = API_KEY_DEFAULT
api_secret_to_use = API_SECRET_DEFAULT


# Global KiteConnect and Rich Console instances
# Initialize KiteConnect with reduced library logging to prevent console clutter from the library itself
kite = KiteConnect(api_key=api_key_to_use)
console = Console(record=True) # For Rich text and table display


# --- Core Functions ---
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

