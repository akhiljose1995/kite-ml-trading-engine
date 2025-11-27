import os
import pandas as pd
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from datetime import datetime, timedelta
from datetime import time as time1
from kiteconnect import KiteConnect
import traceback

class PnLSimulator_OI_Tracker:
    def __init__(self, kite, signal_file):
        try:
            self.kite = kite
            self.signal_file = signal_file
            self.last_processed_time = None
        except Exception:
            print("Error in __init__:\n", traceback.format_exc())

    def read_latest_signal(self):
        try:
            df = pd.read_csv(self.signal_file, parse_dates=["datetime"])
            if df.empty:
                return None
            latest = df.iloc[-1].to_dict()
            return df.to_dict(orient="records")
            if self.last_processed_time is None or latest["datetime"] > self.last_processed_time:
                return latest
        except Exception:
            print("Error in read_latest_signal:\n", traceback.format_exc())
        return None

    def resolve_instrument_token(self, tradingsymbol):
        try:
            instruments = self.kite.instruments("NFO")
            df = pd.DataFrame(instruments)
            match = df[df["tradingsymbol"] == tradingsymbol]
            if not match.empty:
                return int(match["instrument_token"].values[0])
        except Exception:
            print(f"Error resolving token for {tradingsymbol}:\n", traceback.format_exc())
        return None

    def append_result_to_file(self, index_symbol, row_dict):
        try:
            filename = f"oi_data/inaccurate/{index_symbol}_pnl_{datetime.now().date()}.csv"
            df = pd.DataFrame([row_dict])
            if not os.path.exists(filename):
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, mode="a", header=False, index=False)
            print(f"Appended latest signal: {row_dict}")
        except Exception:
            print("Error in append_result_to_file:\n", traceback.format_exc())

    def track_trade(self, signal):
        try:
            index_symbol = signal["index_symbol"]
            tradingsymbol = signal["tradingsymbol"]
            timestamp = signal["datetime"]
            entry_ltp = signal["ltp"]
            end_time = datetime.fromisoformat(signal["end time"])
            entry_ltp_minus_1 = entry_ltp - 1

            token = self.resolve_instrument_token(tradingsymbol)
            if not token:
                return

            if end_time > timestamp + timedelta(minutes=5):
                end_time = timestamp + timedelta(minutes=5)
            ohlc = self.kite.historical_data(token, timestamp, end_time, "minute")
            df = pd.DataFrame(ohlc)
            if df.empty:
                return

            low_row = df.loc[df["low"].idxmin()]
            high_row = df.loc[df["high"].idxmax()]

            result = {
                "start_time": timestamp,
                "tradingsymbol": tradingsymbol,
                "entry_ltp": entry_ltp,
                "entry_ltp_minus_1": entry_ltp_minus_1,
                "low_time": low_row["date"],
                "low_ltp": low_row["low"],
                "high_time": high_row["date"],
                "high_ltp": high_row["high"],
                "close_ltp": df.iloc[-1]["close"]
            }

            self.append_result_to_file(index_symbol, result)
            self.last_processed_time = timestamp

        except Exception:
            print("Error in track_trade:\n", traceback.format_exc())

    def run(self):
        """
        """
        try:
            signals = self.read_latest_signal()
            for signal in signals:
                #signal["datetime"] = datetime.fromisoformat(signal["datetime"])
                # continue if signal["datetime"] is not equal to today's date
                if signal["datetime"].date() != datetime.now().date():
                    continue
                try:
                    end_time = signal.get("end time")
                    if not end_time:
                        end_time = signal["datetime"] + timedelta(minutes=5)                        
                except:
                    end_time = signal["datetime"] + timedelta(minutes=5)
                # Convert end_time string to datetime if it's a string
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time)
                if end_time - signal["datetime"] > timedelta(minutes=5):
                    end_time = signal["datetime"] + timedelta(minutes=5)
                self.track_trade(signal)
        except Exception:
            print("Error in run:\n", traceback.format_exc())

    """ def run(self):
        try:
            signal, prev_signal = {}, {}
            prev_tradingsymbol, appended_signal = None, None
            prev_datetime = None
            track_signal_flag = False
            exit_flag = False
            end_time = None
            prev_signal_tracked = False
            signal_list = []
            
            while not exit_flag:
                prev_signal = signal
                while True:
                    signal = self.read_latest_signal()
                    prev_signal["end time"] = signal["datetime"] if signal else None
                    signal_list.append(signal)
                    if signal:
                        # Fetch tradingsymbol and datetime from signal
                        if (signal["tradingsymbol"] != prev_tradingsymbol or
                            signal["datetime"] != prev_datetime):
                            print(f"New signal detected: {signal}")
                            prev_tradingsymbol = signal["tradingsymbol"]
                            prev_datetime = signal["datetime"]
                            track_signal_flag = True
                            end_time = datetime.now()
                            break
                        # If current time is past 3:30 PM, exit the script gracefully
                    if datetime.now().time() >= time1(15, 31):
                        print("[bold yellow]Market close time reached (3:30 PM). Exiting script.[/bold yellow]")
                        exit_flag = True
                        prev_signal["end time"] = datetime.now()
                        break
                                     
                    time.sleep(1)  # Adjust polling interval as needed
                if track_signal_flag:
                    print(f"Tracked: {prev_signal}")
                    # If end_time is not set, track for 5 minutes from signal time
                    #if end_time > prev_signal["datetime"] + timedelta(minutes=5):
                    #    end_time = prev_signal["datetime"] + timedelta(minutes=5)
                    self.track_trade(prev_signal)
                    track_signal_flag = False
                    
                # If current time is past 3:30 PM, exit the script gracefully
                if datetime.now().time() >= time1(15, 31):
                    print("[bold yellow]Market close time reached (3:30 PM). Exiting script.[/bold yellow]")
                    break

        except Exception:
            print("Error in run:\n", traceback.format_exc()) """
    
    def calculate_pnl(self, signal_file):
        try:
            # Convert signal_file to dataframe
            df = pd.read_csv(signal_file, parse_dates=["start_time", "low_time", "high_time"])
            if df.empty:
                return 0.0
            
            # Calculate new low ltp column
            df["new_low_ltp"] = df["low_ltp"] - 2

            # Calculate new high ltp column. 
            # If entry_ltp - high_ltp-2 > 2, then new_high_ltp = entry_ltp - 2
            # else new_high_ltp = high_ltp -2
            df["new_high_ltp"] = df.apply(lambda row: row["entry_ltp"] - 2 if (row["entry_ltp"] - (row["high_ltp"]-2) > 2) else row["high_ltp"] - 2, axis=1)

        except Exception:
            print("Error in calculate_pnl:\n", traceback.format_exc())
        return 0.0  
# Example usage
def main():
    try:
        kite = KiteConnect(api_key=config.API_KEY)
        kite.set_access_token(config.ACCESS_TOKEN)

        simulator = PnLSimulator_OI_Tracker(kite, signal_file="entry_signals.csv")
        simulator.run()
    except Exception:
        print("Error in main:\n", traceback.format_exc())

if __name__ == "__main__":
    main()
