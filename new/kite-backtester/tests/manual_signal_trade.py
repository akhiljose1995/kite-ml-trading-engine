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
        kite = KiteConnect(api_key=config.API_KEY)
        kite.set_access_token(config.ACCESS_TOKEN)
        print("Kite connection established.")
        """ simulator = PnLSimulator_OI_Tracker(kite, signal_file="oi_data/entry_signals.csv")
        simulator.run() """
        #kite = KiteConnect(api_key="your_api_key")
        #kite.set_access_token("your_access_token")

        #manager = SignalTrailManager(kite, signal_file="oi_data/entry_signals.csv")
        #manager.run()

        manager = Manager()
        signal_dict = manager.dict()

        tracker = SignalTrailTrader(kite, signal_file="oi_data/manual_entry_signals.csv", signal_dict=signal_dict, qty=qty)

        p1 = Process(target=tracker.track_entry_signals, args=(signal_dict,))
        p2 = Process(target=tracker.dispatch_sl_trailing, args=(signal_dict,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    except Exception:
        print("Error in main:\n", traceback.format_exc())
if __name__ == "__main__":
    main()