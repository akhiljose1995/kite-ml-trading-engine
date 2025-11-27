from kiteconnect import KiteConnect, KiteTicker
from multiprocessing import Process, Manager
import time
from datetime import time as time1
import os
import threading
import pandas as pd
import time
from datetime import datetime
from kiteconnect import KiteConnect
import json
import traceback

class SignalTrailTrader:
    def __init__(self, kite, signal_file, signal_dict, qty=75):
        self.kite = kite
        self.qty = qty
        self.signal_file = signal_file
        self.signal_dict = signal_dict
        self.last_signal_time = None
        self.signal_counter = 0
        self.exit_time = time1(15, 15)
    
    # ----------------------------
    # 1. Place BUY order at Market
    # ----------------------------
    def place_buy_order(self, symbol):
        print(f"[INFO] Placing BUY order for {symbol}")
        buy_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_BUY,
            quantity=self.qty,
            product=self.kite.PRODUCT_NRML,
            order_type=self.kite.ORDER_TYPE_MARKET
        )
        print("[INFO] Buy order placed. Waiting for execution...")
        avg_price = self.wait_for_execution(buy_order_id)
        return buy_order_id, avg_price

    # ----------------------------
    # 2. Place SELL order at Market
    # ----------------------------
    def place_sell_order(self, symbol):
        print(f"[INFO] Placing SELL order for {symbol}")
        sell_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=self.qty,
            product=self.kite.PRODUCT_NRML,
            order_type=self.kite.ORDER_TYPE_MARKET
        )
        print("[INFO] SELL order placed. Waiting for execution...")
        avg_price = self.wait_for_execution(sell_order_id)
        return sell_order_id, avg_price

    # ----------------------------
    # 3. Wait for order execution
    # ----------------------------
    def wait_for_execution(self, order_id, status="COMPLETE"):
        while True:
            order = self.kite.order_history(order_id)
            #print(f"[INFO] Checking order status: {order[-1]}")
            if order[-1]["status"] == status:
                print(f"[INFO] Buy order executed @ {order[-1]}")
                return order[-1]['average_price']
            time.sleep(0.1)

    # ----------------------------
    # 4. Check status of order
    # ----------------------------
    def get_order_data(self, order_id, value="status"):
        """
        'average_price', 'filled_quantity', 'status', 'quantity', 'price', 'trigger_price'
        "full data" - return entire order dict
        """
        if value == "full data":
            order = self.kite.order_history(order_id)
            return order[-1]
        order = self.kite.order_history(order_id)
        return order[-1][value]

    # ----------------------------
    # 5. Place initial SL-M order
    # ----------------------------
    def place_initial_sl(self, symbol, initial_sl):
        print(f"[INFO] Placing initial SL-M @ {initial_sl}")
        limit_price = round(initial_sl - 0.10, 1)  # 10 paisa below trigger (for SELL SL)
        sl_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=self.qty,
            product=self.kite.PRODUCT_NRML,
            #order_type=self.kite.ORDER_TYPE_SL_MARKET,
            order_type="SL",                  # Use SL, not SL-M
            price=limit_price,                # Limit price < Trigger
            trigger_price=initial_sl
        )
        print(f"[INFO] SL-M placed at {initial_sl}. ID: {sl_order_id}")
        order = self.kite.order_history(sl_order_id)
        print (f"[INFO] SL Order details: {order[-1]}")
        return sl_order_id

    # ----------------------------
    # 6. Modify SL dynamically
    # ----------------------------
    def modify_sl(self, sl_order_id, new_sl):
        #if not self.sl_order_id:
        #    return
        try:
            limit_price = round(new_sl - 0.10, 1)
            new_sl = round(new_sl, 1)
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=sl_order_id,
                price=limit_price,
                trigger_price=new_sl
            )
            print(f"[UPDATE] SL modified to {new_sl}")
            #order = self.kite.order_history(sl_order_id)
            #print (f"[INFO] Updated SL Order details: {order[-1]}")
            #self.last_sl = new_sl
        except Exception as e:
            print(f"[ERROR] Failed to modify SL: {e}")
            if "Maximum allowed order modifications exceeded." in str(e):
                return False, "Exit at market"
        return True, "COMPLETE"

    # Cancel an order
    def cancel_the_order(self, order_id):
        try:
            cancelled = self.kite.cancel_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id
            )
            print(f"[INFO] Order {order_id} cancelled successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to cancel order {order_id}: {e}")

    def read_latest_signal(self):
        try:
            df = pd.read_csv(self.signal_file, parse_dates=["datetime"])
            if df.empty:
                return None
            latest = df.iloc[-1].to_dict()
            #with open(self.signal_file, "r") as f:
            #    latest = json.loads(f.read())
            #    latest["datetime"] = datetime.fromisoformat(latest["datetime"])
            #print(f"[INFO] Latest signal read: {latest}")
            if (self.last_signal_time is None or latest["datetime"] > self.last_signal_time) and \
                (((datetime.now() - latest["datetime"]).total_seconds() < 10) and \
                datetime.now().date() == latest["datetime"].date()):
                self.last_signal_time = latest["datetime"]
                return latest
        except Exception as e:
            print(f"[ERROR] reading signal: {e}")
        return None

    def get_current_ltp(self, symbol):
        try:
            ltp_data = self.kite.ltp(f"NFO:{symbol}")
            return ltp_data[f"NFO:{symbol}"]["last_price"]
        except Exception as e:
            print(f"[ERROR] Failed to fetch LTP for {symbol}: {e}")
            return None

    def detect_entry_ltp(self, symbol, signal_ltp, actual_trading=False):
        ltp_history = []
        prev_ltp = None
        final_entry_ltp = None
        buy_order_id, sl_order_id, avg_price = None, None, None
        while True:
            ltp = self.get_current_ltp(symbol)
            if ltp and ltp != prev_ltp:
                ltp_history.append(ltp)
                prev_ltp = ltp
                print(f"[LTP] {symbol}: {ltp}")
                if len(ltp_history) >= 3:
                    #if ltp > signal_ltp:
                    #    print(f"[ENTRY] LTP {ltp} crossed signal LTP {signal_ltp}")
                    #    final_entry_ltp = ltp
                    #    break
                    if (ltp_history[-1] > ltp_history[-2] > ltp_history[-3]):
                        entry_ltp = ltp_history[-1]
                        print(f"[ENTRY] Reversal detected. Entry LTP: {entry_ltp}")
                        final_entry_ltp = entry_ltp
                        break
            time.sleep(0.1)
        if final_entry_ltp:
            buy_order_id, avg_price = self.place_buy_order(symbol)
            if buy_order_id and avg_price:
                print(f"Immediately placing SL for {symbol} at {final_entry_ltp - 2}")
                sl_order_id = self.place_initial_sl(symbol, initial_sl=int(avg_price) - 2)
        return buy_order_id, avg_price, sl_order_id

    def trail_sl_until_exit(self, signal_id, signal_dict, actual_trading=False, exit=False):
        signal = signal_dict[signal_id]
        print(f"[START TRAIL] Signal ID {signal_id}: {signal}")
        symbol = signal["tradingsymbol"]
        entry_ltp = signal["entry_ltp"]
        sl_order_id = signal["sl_order_id"]
        exit_order_status = signal["exit_order_status"]
        modify_sl_limit_count = 3
        if "sl" in signal and signal["sl"] is not None:
            sl = signal["sl"]
        else:
            signal["sl"] = entry_ltp - 2
            sl = signal["sl"]
        prev_ltp = entry_ltp

        while True:
            ltp = self.get_current_ltp(symbol)
            if ltp is None:
                time.sleep(0.1)
                continue

            #print(f"[TRACK] {symbol} LTP: {ltp}, SL: {sl}")

            # Exit when ltp reaches above prev_ltp+2
            if ltp > entry_ltp+2:
                # First cancel already open order
                print(f"[EXIT CHECK] LTP {ltp} crossed exit threshold {prev_ltp+2}")
                cancelled = self.cancel_the_order(sl_order_id)
                self.wait_for_execution(sl_order_id, status="CANCELLED")
                cancelled_order_status = self.get_order_data(order_id=sl_order_id, value="status")
                print(f"[CANCELLED] {symbol} SL Order Status: {cancelled_order_status}")
                print(f"[EXIT CHECK] Exiting at market as LTP crossed exit threshold.")
                sell_order_id, avg_price = self.place_sell_order(symbol)
                sl_order_id = sell_order_id
                self.wait_for_execution(sell_order_id, status="COMPLETE")
                exit_order_status = self.get_order_data(order_id=sell_order_id, value="status")
                print(f"[CHECK EXIT] {symbol} SL Order Status: {exit_order_status}")
                

            elif ltp > prev_ltp:
                #delta = int(ltp - prev_ltp)
                sl = round(ltp-1, 1)
                signal["sl"] = sl
                exit_order_status = self.get_order_data(order_id=sl_order_id, value="status")
                if exit_order_status != "COMPLETE":
                    status, message = self.modify_sl(sl_order_id, sl)
                    if status:
                        print(f"[SL TRAIL] New SL: {sl}")
                    elif not status and "Exit at market" in message:
                        modify_sl_limit_count += 1
                        if modify_sl_limit_count >= 3:
                            print(f"[SL CANCEL] Max modifications reached. Cancelling SL order.")
                            cancelled = self.cancel_the_order(sl_order_id)
                            print(f"[SL CANCEL] Exiting at market as max modifications reached.")
                            sell_order_id, avg_price = self.place_sell_order(symbol)
                prev_ltp = ltp

            exit_order_status = self.get_order_data(order_id=sl_order_id, value="status")
            print(f"[CHECK EXIT] {symbol} SL Order Status: {exit_order_status}")
            if datetime.now().time() >= self.exit_time or \
                exit_order_status == "COMPLETE":
            #if ltp < sl or \
            #    datetime.now().time() >= self.exit_time or \
            #    exit_order_status == "COMPLETE":
                # Print exit time order details
                print(f"[EXIT ORDER] SL Order details: {self.get_order_data(order_id=sl_order_id, value='full data')}")
                exit_ltp = self.get_order_data(order_id=sl_order_id, value="average_price")
                signal["exit_ltp"] = exit_ltp
                signal["pnl"] = round(exit_ltp - entry_ltp, 2)
                signal["track_status"] = "completed"
                signal["exit_order_status"] = exit_order_status
                print(f"[EXIT] {symbol} exited at SL: {sl}, PnL: {signal['pnl']}")
                index_symbol = signal["index_symbol"]
                filename = f"oi_data/manual/{index_symbol}_pnl_{datetime.now().date()}.csv"
                df = pd.DataFrame([signal])  # signal is a dictionary with all required fields

                if not os.path.exists(filename):
                    df.to_csv(filename, index=False)
                else:
                    df.to_csv(filename, mode="a", header=False, index=False)

                print(f"[LOG] Signal exit data saved to {filename}")
                if datetime.now().time() >= self.exit_time:
                    print("Market close time reached (3:30 PM). Exiting script.")
                break
            time.sleep(0.1)

    def run(self):
        while True:
            if datetime.now().time() >= self.exit_time:
                # Print message as Exiting script
                print("Market close time reached (3:30 PM). Exiting script.")
                break
            latest_signal = self.read_latest_signal()
            if latest_signal is None:
                time.sleep(2)
                continue

            symbol = latest_signal["tradingsymbol"]
            already_tracked = any(
                s["tradingsymbol"] == symbol and s["track_status"] == "in_progress"
                for s in self.signal_dict.values()
            )

            if not already_tracked:
                self.signal_counter += 1
                signal_id = self.signal_counter
                signal_entry = {
                    "index_symbol": latest_signal["index_symbol"],
                    "datetime": latest_signal["datetime"],
                    "tradingsymbol": symbol,
                    "signal_ltp": latest_signal["ltp"],
                    "entry_ltp": None,
                    "sl": None,
                    "track_status": "in_progress",
                    "exit_ltp": None,
                    "pnl": None
                }
                self.signal_dict[signal_id] = signal_entry
                print(f"[NEW SIGNAL] Tracking signal ID {signal_id} for {latest_signal}")
                entry_ltp = self.detect_entry_ltp(symbol, latest_signal["ltp"])
                self.signal_dict[signal_id]["entry_ltp"] = entry_ltp

                self.trail_sl_until_exit(signal_id)

            time.sleep(0.1)

    def track_entry_signals(self, signal_dict):
        last_signal_time = None
        signal_counter = 0
        #self.signal_dict = signal_dict

        while True:
            try:
                if datetime.now().time() >= self.exit_time:
                    # Print message as Exiting script
                    print("Market close time reached (3:30 PM). Exiting track_entry_signals() script.")
                    break
                latest_signal = self.read_latest_signal()
                if latest_signal:
                    signal_counter += 1
                    signal_id = str(signal_counter)
                    signal_dict[signal_id] = {
                        **latest_signal,
                        "signal_ltp": latest_signal["ltp"],
                        "entry_ltp": None,
                        "buy_order_id": None,
                        "sl_order_id": None,
                        "exit_order_status": None,
                        "sl": None,
                        "track_status": "in_progress",
                        "exit_ltp": None,
                        "pnl": None
                    }
                    self.last_signal_time = latest_signal["datetime"]
                time.sleep(3)
            except TimeoutError:
                print(f"[ERROR] in track_entry_signals: {traceback.format_exc()}")
                time.sleep(1)

    def dispatch_sl_trailing(self, signal_dict, actual_trading=False):
        active_signals = set()
        self.signal_dict = signal_dict
        buy_order_id, avg_price, sl_order_id = None, None, None
        exit = False

        while True:
            if datetime.now().time() >= self.exit_time:
                # Print message as Exiting script
                print("Market close time reached (3:30 PM). Exiting dispatch_sl_trailing() script.")
                break
            for signal_id, signal in signal_dict.items():
                if signal["track_status"] == "in_progress" and signal_id not in active_signals:
                    print(f"[NEW SIGNAL] Tracking signal ID {signal_id} for {signal}")
                    # Wait till the current minute becomes greater than signal time minute
                    # Get minute of current time and signal time
                    current_minute = datetime.now().minute
                    signal_minute = signal["datetime"].minute
                    #while current_minute <= signal_minute:
                    #    time.sleep(0.1)
                    #    current_minute = datetime.now().minute

                    buy_order_id, avg_price, sl_order_id = self.detect_entry_ltp(signal["tradingsymbol"] , signal["signal_ltp"], actual_trading=actual_trading)
                    updated_signal = dict(signal)  # make a copy
                    updated_signal["entry_ltp"] = avg_price
                    updated_signal["buy_order_id"] = buy_order_id
                    updated_signal["sl_order_id"] = sl_order_id
                    signal_dict[signal_id] = updated_signal  # reassign to manager dict
                    if not (buy_order_id and avg_price and sl_order_id):
                        exit = True
                        signal["track_status"] = "completed"
                        continue
                    p = Process(target=self.trail_sl_until_exit, args=(signal_id, signal_dict, actual_trading, exit))
                    p.start()
                    active_signals.add(signal_id)
        
            time.sleep(0.1)


# ==========================================================
# Example usage
# ==========================================================
""" if __name__ == "__main__":
    trader = TrailingStopLossTrader(
        api_key="your_api_key",
        access_token="your_access_token",
        symbol="NIFTY25O1425200PE",
        qty=75,
        entry_price=65,
        initial_sl=63,
        trailing_gap=2,   # SL stays 2 points below LTP
        trail_step=1      # update SL every +1 move
    )

    trader.place_buy_order()
    trader.place_initial_sl()
    trader.start_trailing() """
def main():
    kite = KiteConnect(api_key="your_api_key")
    kite.set_access_token("your_access_token")

    manager = SignalTrailTrader(kite, signal_file="entry_signals.csv")
    manager.run()

if __name__ == "__main__":
    main()
