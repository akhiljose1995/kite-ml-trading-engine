import os
import pandas as pd
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from datetime import datetime, timedelta
from datetime import time as time1
from kiteconnect import KiteConnect
import traceback
import config
from multiprocessing import Process, Manager
import time
from general_lib.pnl_simulator import PnLSimulator_OI_Tracker
from strategy.manually_entered_trading import SignalTrailTrader

def main():
    try:
        qty = 75  # Set your desired quantity here
        ATM_TRADESYMBOL = None
        ATM_LTP = None
        kite = KiteConnect(api_key=config.API_KEY)
        kite.set_access_token(config.ACCESS_TOKEN)
        print("Kite connection established.")
        """ simulator = PnLSimulator_OI_Tracker(kite, signal_file="oi_data/entry_signals.csv")
        simulator.run() """
        #kite = KiteConnect(api_key="your_api_key")
        #kite.set_access_token("your_access_token")

        #manager = SignalTrailManager(kite, signal_file="oi_data/entry_signals.csv")
        #manager.run()
        # Date time IST
        while True:

            # Manually take input
            ATM_TRADESYMBOL = input("Enter the ATM Trading Symbol (or type 'exit' to quit): ")
            if ATM_TRADESYMBOL.lower() == 'exit':
                print("Exiting manual signal entry.")
                continue

            # Get LTP from kite
            ATM_LTP = kite.ltp(f"NFO:{ATM_TRADESYMBOL}")[f"NFO:{ATM_TRADESYMBOL}"]['last_price']
            now = time.time()
            date = datetime.now().replace(microsecond=0).isoformat()  # ISO format with timezone if needed
                
            row = {
                "index_symbol": "nifty50",
                "datetime": date,
                "tradingsymbol": ATM_TRADESYMBOL,
                "ltp": ATM_LTP
            }

            # File path
            file_path = "oi_data/manual_entry_signals.csv"

            # Write or append
            df = pd.DataFrame([row])
            if not os.path.exists(file_path):
                df.to_csv(file_path, index=False)
                
            else:
                df.to_csv(file_path, mode="a", header=False, index=False)

    except Exception:
        print("Error in main:\n", traceback.format_exc())
if __name__ == "__main__":
    main()