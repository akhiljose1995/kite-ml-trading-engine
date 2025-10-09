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
from general_lib.pnl_simulator import PnLSimulator_OI_Tracker

def main():
    try:
        kite = KiteConnect(api_key=config.API_KEY)
        kite.set_access_token(config.ACCESS_TOKEN)

        simulator = PnLSimulator_OI_Tracker(kite, signal_file="entry_signal.csv")
        simulator.run()
    except Exception:
        print("Error in main:\n", traceback.format_exc())
if __name__ == "__main__":
    main()