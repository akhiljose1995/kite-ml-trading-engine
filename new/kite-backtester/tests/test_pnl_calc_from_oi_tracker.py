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
from strategy.trailing_sl_trading import SignalTrailTrader

# ==============================
# Rich Terminal Output
# ==============================
from rich.console import Console, Group
console = Console(record=True) # For Rich text and table display
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

def wait_till_market_open():
    """
    Waits until the market opens (9:15 AM) before proceeding.
    Checks the current time and sleeps in intervals until market open time is reached.
    """
    market_open_time = time1(9, 45)  # 9:45 AM
    console.print(f"[bold yellow]Waiting until market opens and settles. {market_open_time.strftime('%H:%M')}...[/bold yellow]")
    while True:
        now = datetime.now().time()
        if now >= market_open_time:
            console.print(f"[bold green]Proceeding with operations as market is open.[/bold green]")
            break
        else:
            time_to_open = datetime.combine(datetime.today(), market_open_time) - datetime.combine(datetime.today(), now)
            minutes, seconds = divmod(time_to_open.seconds, 60)
            console.print(f"[yellow]Time until market open: {minutes} minutes and {seconds} seconds.[/yellow]", end='\r')
            time.sleep(2)  # Sleep for 30 seconds before checking again

def main():
    try:
        qty = 75  # Set your desired quantity here
        kite = KiteConnect(api_key=config.API_KEY)
        kite.set_access_token(config.ACCESS_TOKEN)
        print("Kite connection established.")
        """ simulator = PnLSimulator_OI_Tracker(kite, signal_file="oi_data/entry_signals.csv")
        simulator.run() """

        # Wait till 9:45 AM to start tracking signals
        wait_till_market_open()

        #kite = KiteConnect(api_key="your_api_key")
        #kite.set_access_token("your_access_token")

        #manager = SignalTrailManager(kite, signal_file="oi_data/entry_signals.csv")
        #manager.run()

        manager = Manager()
        signal_dict = manager.dict()
        prev_candle = manager.dict({"open": None, "close": None})

        tracker = SignalTrailTrader(kite, signal_file="oi_data/entry_signals.csv", signal_dict=signal_dict, \
                                    qty=qty, prev_candle=prev_candle)

        p1 = Process(target=tracker.track_entry_signals, args=(signal_dict,))
        p2 = Process(target=tracker.dispatch_sl_trailing, args=(signal_dict,))
        p3 = Process(target=tracker.read_last_candle, args=(256265, "minute"))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

    except Exception:
        print("Error in main:\n", traceback.format_exc())
if __name__ == "__main__":
    main()