from kiteconnect import KiteConnect, KiteTicker
from multiprocessing import Process, Manager, Event, Value
import time
from datetime import time as time1
import os
import threading
import traceback
import pandas as pd
import time
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import json

"""# ----------------------------
    # 7️. Check if position still open
    # ----------------------------
    def check_position_open(self):
        positions = self.kite.positions()["net"]
        pos = next((p for p in positions if p["tradingsymbol"] == self.symbol), None)
        return bool(pos and pos["quantity"] != 0)
    """
class SignalTrailTrader:
    def __init__(self, kite, signal_file, signal_dict, qty=75, prev_candle=None, trade_method="buy", \
                 actual_trade=False, options_and_ltp=None):
        self.kite = kite
        self.qty = qty
        self.signal_file = signal_file
        self.signal_dict = signal_dict
        self.prev_candle = prev_candle if prev_candle else {"date": None, "open": None, "close": None}
        self.last_signal_time = None
        self.signal_counter = 0
        self.exit_time = time1(15, 15)
        self.trade_method = trade_method  # "buy" or "sell"
        self.id_counter = 0
        self.actual_trade = actual_trade
        self.options_and_ltp = options_and_ltp
        self.initial_sl = 2
        self.diff_for_sl_to_match_entry_ltp = 3  # Initial difference between entry ltp and sl to set sl=entry_ltp
        self.diff_to_exit = 5  # Difference between entry ltp and current ltp to exit the trade
    
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
        print(f"[INFO] Placing Sell order for {symbol}")
        sell_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=self.qty,
            product=self.kite.PRODUCT_NRML,
            order_type=self.kite.ORDER_TYPE_MARKET
        )
        print("[INFO] Sell order placed. Waiting for execution...")
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
                print(f"[INFO] Order executed @ {order[-1]}")
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
        if self.trade_method == "buy":
            limit_price = round(initial_sl - 0.10, 1)  # 10 paisa above trigger (for BUY SL)
            transaction_type = self.kite.TRANSACTION_TYPE_SELL
        else:
            limit_price = round(initial_sl + 0.10, 1)  # 10 paisa below trigger (for SELL SL)
            transaction_type = self.kite.TRANSACTION_TYPE_BUY

        sl_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
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
    
    # Cancel and Exit the trade
    def cancel_and_exit_trade(self, symbol, order_id):
        try:
            print(f"[SL CANCEL] Max modifications reached. Cancelling SL order.")
            self.cancel_the_order(order_id)
            self.wait_for_execution(order_id, status="CANCELLED")
            cancelled_order_status = self.get_order_data(order_id=order_id, value="status")
            print(f"[CANCELLED] {symbol} SL Order Status: {cancelled_order_status}")
            print(f"[SL CANCEL] Exiting at market price.")
            if self.trade_method == "buy":
                exit_order_id, avg_price = self.place_sell_order(symbol)
            else:
                exit_order_id, avg_price = self.place_buy_order(symbol)
            self.wait_for_execution(exit_order_id, status="COMPLETE")
            exit_order_status = self.get_order_data(order_id=exit_order_id, value="status")
            print(f"[CHECK EXIT] {symbol} SL Order Status: {exit_order_status}")
            return exit_order_status, exit_order_id, avg_price
        except Exception as e:
            print(f"[ERROR] Failed to cancel and exit trade for {symbol}: {e}")    
            return None, None, None

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

    def get_current_ltp(self, exchange="NFO", symbol=""):
        retry_count = 10
        try:
            while retry_count > 0:
                try:
                    ltp_data = self.kite.ltp(f"NFO:{symbol}")
                    return ltp_data[f"{exchange}:{symbol}"]["last_price"], "Success"
                except Exception as e:
                    if "Too many requests" in str(e) or "Read timed out" in str(e):
                        retry_count -= 1
                        time.sleep(10)
                    else:
                        print(f"[ERROR] Failed to fetch LTP for {symbol}: {traceback.format_exc()}")
                        return None, e
        except Exception as e:
            print(f"[ERROR] Failed to fetch LTP for {symbol}: {traceback.format_exc()}")
            return None, e
    
    def get_current_ltp_in_loop(self, options_and_ltp, exchange="NFO", symbol=""):
        updated_signal = dict(options_and_ltp[symbol])  # make a copy
        stop_event = options_and_ltp[symbol]["stop_event"]
        while not stop_event:
            try:
                #print(f"[LTP LOOP] Fetching LTP for {symbol}")
                ltp, status = self.get_current_ltp(exchange=exchange, symbol=symbol)
                if ltp is None and status == "Too many requests":
                    time.sleep(10)
                    continue
                updated_signal["ltp"] = ltp
                options_and_ltp[symbol] = updated_signal # reassign to manager dict
                #print(f"[LTP LOOP] {symbol} LTP updated to {ltp} and stored {options_and_ltp[symbol]['ltp']}")
                time.sleep(0.2)

                # past self.exit_time, break the loop
                if datetime.now().time() >= self.exit_time:
                    print("Market close time reached (3:30 PM). Exiting LTP fetch loop.")
                    break
            except Exception as e:
                print(f"[ERROR] Failed to fetch LTP for {symbol} in loop: {e}")
            
    def detect_entry_ltp(self, symbol, signal_ltp):
        ltp_history = []
        prev_ltp = None
        final_entry_ltp = None
        entry_order_id, exit_order_id, avg_price, entry_time = None, None, None, None

        if self.trade_method == "buy":
            while True:
                ltp, status_string = self.get_current_ltp(symbol=symbol)
                if ltp and ltp != prev_ltp:
                    ltp_history.append(ltp)
                    prev_ltp = ltp
                    print(f"[LTP] {symbol}: {ltp}")
                    if len(ltp_history) >= 3:
                        #if ltp > signal_ltp:
                        #    print(f"[ENTRY] LTP {ltp} crossed signal LTP {signal_ltp}")
                        #    final_entry_ltp = ltp
                        #    break
                        if (ltp_history[-1] > ltp_history[-2] > ltp_history[-3]) or\
                            (int(ltp_history[-1]) - int(ltp_history[-2]) >=2):
                            entry_ltp = ltp_history[-1]
                            print(f"[ENTRY] Reversal detected. Entry LTP: {entry_ltp}")
                            final_entry_ltp = entry_ltp
                            break
                time.sleep(0.1)
            if final_entry_ltp:
                if self.actual_trade:
                    entry_order_id, avg_price = self.place_buy_order(symbol)
                else:
                    entry_order_id, avg_price = f"TESTBUY{int(time.time())}", final_entry_ltp
                if entry_order_id and avg_price:
                    entry_time = datetime.now()
                    print(f"Immediately placing SL for {symbol} at {final_entry_ltp - 2}")
                    if self.actual_trade:
                        exit_order_id = self.place_initial_sl(symbol, initial_sl=int(avg_price) - 2)
                    else:
                        exit_order_id = f"TESTSL{int(time.time())}"

        elif self.trade_method == "sell":
            while True:
                ltp, status_string = self.get_current_ltp(symbol=symbol)
                if ltp and ltp != prev_ltp:
                    ltp_history.append(ltp)
                    prev_ltp = ltp
                    print(f"[LTP] {symbol}: {ltp}")
                    if len(ltp_history) >= 3:
                        #if ltp < signal_ltp:
                        #    print(f"[ENTRY] LTP {ltp} crossed signal LTP {signal_ltp}")
                        #    final_entry_ltp = ltp
                        #    break
                        if (ltp_history[-1] < ltp_history[-2] < ltp_history[-3]) or\
                            (int(ltp_history[-2]) - int(ltp_history[-1]) >=2):
                            entry_ltp = ltp_history[-1]
                            print(f"[ENTRY] Reversal detected. Entry LTP: {entry_ltp}")
                            final_entry_ltp = entry_ltp
                            break
                time.sleep(0.1)
            if final_entry_ltp:
                if self.actual_trade:
                    entry_order_id, avg_price = self.place_sell_order(symbol)
                else:
                    entry_order_id, avg_price = f"TESTSELL{int(time.time())}", final_entry_ltp
                if entry_order_id and avg_price:
                    entry_time = datetime.now()
                    if self.trade_method == "sell":
                        print(f"Immediately placing SL for {symbol} at {final_entry_ltp + self.initial_sl}")
                    elif self.trade_method == "buy":
                        print(f"Immediately placing SL for {symbol} at {final_entry_ltp - self.initial_sl}")
                    if self.actual_trade:
                        exit_order_id = self.place_initial_sl(symbol, initial_sl=int(avg_price) + self.initial_sl)
                    else:
                        exit_order_id = f"TESTSL{int(time.time())}"
        return entry_order_id, avg_price, exit_order_id, entry_time
    
    def read_last_candle(self, token=256265, interval="minute"):
        """
        Have a while loop and keep on fetch the last candle for the given symbol and interval. Then update open and close values 
        in self.prev_candle
        """
        try:
            print(f"[START] Starting read_last_candle for token {token} at interval {interval}")
            while True:
                try:
                    if datetime.now().time() >= self.exit_time:
                        # Print message as Exiting script
                        print("Market close time reached (3:30 PM). Exiting read_last_candle() script.")
                        break
                    #print(f"[INFO] Fetching last candle for token {token} at interval {interval}")
                    from_date = datetime.now() - timedelta(minutes=30)
                    to_date = datetime.now()
                    #start_time = time.time()
                    candles = self.kite.historical_data(
                        instrument_token=token,
                        from_date=from_date,
                        to_date=to_date,
                        interval=interval
                    )
                    #end_time = time.time()
                    #print(f"Time taken to fetch last candle: {end_time - start_time} seconds")
                    #print(f"[INFO] Fetched candles:\n")
                    #for can in candles[-5:]:
                    #    print(can)
                    if candles:
                        last_candle = candles[-2]
                        self.prev_candle["date"] = last_candle["date"]
                        self.prev_candle["open"] = last_candle["open"]
                        self.prev_candle["close"] = last_candle["close"]
                        #print(f"[CANDLE] Updated last candle for {token}: {self.prev_candle}")
                    time.sleep(5)
                except Exception as e:
                    print(f"[ERROR] fetching last candle for {token}: {traceback.format_exc()}")
                    time.sleep(5)
        except Exception as e:
            print(f"[ERROR] reading last candle: {e}")

    def check_entry_breakout(self, signal):
        try:
            index_token = signal["index_symbol"]
            strike_token = signal["tradingsymbol"]
            if signal["index_symbol"].lower() == "nifty50":
                index_token = "256265"       # Underlying instrument (e.g., "NIFTY 50", "NIFTY BANK")
                symbol = "NIFTY 50"
            elif signal["index_symbol"].lower() == "niftybank":
                index_token = "260105"
            elif signal["index_symbol"].lower() == "sensex":
                index_token = "265"
            signal_time = signal["datetime"]
            timeout_time = signal_time + timedelta(minutes=3)

            # Wait until strike open > prev_max or timeout
            while datetime.now() < timeout_time:
                try:
                    signal_minute = datetime.now().minute                   
                    current_minute = datetime.now().minute
                    stream_minute = datetime.now().minute
                    retry_count = 10
                    while current_minute == stream_minute:
                        while retry_count > 0:
                            try:
                                current_ltp = self.kite.ltp(index_token)[index_token]["last_price"]
                                break
                            except Exception as e:
                                if "Too many requests" in str(e) or "Read timed out" in str(e):
                                    retry_count -= 1
                                    time.sleep(10)
                                else:
                                    print(f"[ERROR] Failed to fetch LTP for {index_token}: {traceback.format_exc()}")
                                    return "exit"
                        if ("CE" in strike_token and self.trade_method == "buy") or \
                           ("PE" in strike_token and self.trade_method == "sell"):
                            prev_actual = round(float(max(self.prev_candle["open"], self.prev_candle["close"])), 2)
                            prev = (1-0.00001) * prev_actual  # 0.01% slippage
                        elif ("PE" in strike_token and self.trade_method == "buy") or \
                             ("CE" in strike_token and self.trade_method == "sell"):
                            prev_actual = round(float(min(self.prev_candle["open"], self.prev_candle["close"])), 2)
                            prev = (1+0.00001) * prev_actual  # 0.01% slippage

                        #print(f"[INFO] Signal {strike_token}, {signal["index_symbol"]} Previous (open/close): {prev_actual}, Current ltp: {current_ltp}")
                        if self.trade_method == "buy":
                            if ("CE" in strike_token and current_ltp > prev) or \
                                ("PE" in strike_token and current_ltp < prev):
                                print(f"[INFO] Entry condition met for {strike_token} | curr ltp:{current_ltp} | prev(with slippage):{prev}")
                                return "enter"
                        elif self.trade_method == "sell":
                            if ("CE" in strike_token and current_ltp < prev) or \
                            ("PE" in strike_token and current_ltp > prev):
                                print(f"[INFO] Entry condition met for {strike_token} | curr ltp:{current_ltp} | prev(with slippage):{prev}")
                                return "enter"
                        time.sleep(0.5)
                        stream_minute = datetime.now().minute
                except Exception as e:
                    print(f"[ERROR] Exception while checking breakout: {traceback.format_exc()}")
                    if "Read timed out" in str(e):
                        print(f"[WARN] Read timed out while fetching data. Retrying...")
                        time.sleep(0.5)
                        continue
                    if "TimeoutError" in str(e):
                        print(f"[WARN] TimeoutError: The read operation timed out while fetching data. Retrying...")
                        time.sleep(0.5)
                        continue
                    else:
                        return "exit"

                current_minute = datetime.now().minute
                while current_minute <= signal_minute:
                    time.sleep(0.1)
                    current_minute = datetime.now().minute

            print(f"[INFO] Timeout: No breakout within 3 minutes.")
            return "exit"

        except Exception as e:
            print(f"[ERROR] Exception in check_entry_breakout: {e}")
            return "exit"

    def trail_sl_until_exit(self, signal_id, options_and_ltp, exit=False,):
        trade_exit_time = None
        sl_set_to_break_even = False
        signal = self.signal_dict[signal_id]
        print(f"[START TRAIL] Signal ID {signal_id}: {signal}")
        symbol = signal["tradingsymbol"]
        if symbol not in options_and_ltp:
            options_and_ltp_def = {
                        "ltp": 0.0,                # plain float is OK here
                        "stop_event": True      # REAL Event
                    }
            options_and_ltp[symbol] = options_and_ltp_def
        #print(f"options_and_ltp before starting LTP fetch: {options_and_ltp}")
        options_and_ltp_init = dict(options_and_ltp[symbol])  # make a copy
        options_and_ltp_init["ltp"], status_string = self.get_current_ltp(symbol=symbol)
        options_and_ltp_init["stop_event"] = False
        options_and_ltp[symbol] = options_and_ltp_init # reassign to manager dict

        # Start a process to continuously fetch LTP. Also use exit flag to stop it when needed.
        p_ltp = Process(
            target=self.get_current_ltp_in_loop,
            args=(options_and_ltp, "NFO", symbol)
        )
        p_ltp.start()

        entry_ltp = signal["entry_ltp"]
        exit_order_id = signal["exit_order_id"]
        exit_order_status = signal["exit_order_status"]
        modify_sl_limit_count = 3
        exit_ltp = None
        if "sl" in signal and signal["sl"] is not None:
            sl = signal["sl"]
        else:
            if self.trade_method == "buy":
                signal["sl"] = entry_ltp - self.initial_sl
            elif self.trade_method == "sell":
                signal["sl"] = entry_ltp + self.initial_sl
            sl = signal["sl"]   
        prev_ltp = entry_ltp

        while True:
            ltp = options_and_ltp[symbol]["ltp"]
            if ltp == 0.0:
                time.sleep(0.1)
                continue
            #print(f"[LTP FETCH] {symbol} LTP: {ltp} | Entry LTP: {entry_ltp} | SL: {sl}")
            # max_ltp and min_ltp tracking
            signal_max_key = "max_ltp"
            signal_min_key = "min_ltp"
            signal[signal_max_key] = max(int(signal.get(signal_max_key, ltp)), ltp)
            signal[signal_min_key] = min(int(signal.get(signal_min_key, ltp)), ltp)

            #print(f"[TRACK] {symbol} LTP: {ltp}, SL: {sl}")
            if self.trade_method == "sell" and ltp < entry_ltp:
                #delta = int(prev_ltp - ltp)
                if sl_set_to_break_even and entry_ltp-ltp >= self.diff_to_exit:
                    print(f"@@@@@\n[EXIT CONDITION] LTP moved {self.diff_to_exit} points from entry LTP. Exiting trade.\n@@@@@")
                    if self.actual_trade:
                        exit_order_status, exit_order_id, avg_price = self.cancel_and_exit_trade(symbol, exit_order_id)
                    else:
                        exit_order_status = "COMPLETE"
                        exit_ltp = ltp

                if (ltp <= entry_ltp - self.diff_for_sl_to_match_entry_ltp) and not sl_set_to_break_even:
                    print(f">>>>>\n[SL MATCH] LTP moved {self.diff_for_sl_to_match_entry_ltp} points from entry LTP. Setting SL = Entry LTP\n>>>>>")
                    sl = entry_ltp
                    signal["sl"] = sl
                    if self.actual_trade:
                        exit_order_status = self.get_order_data(order_id=exit_order_id, value="status")
                    
                        if exit_order_status != "COMPLETE":
                            status, message = self.modify_sl(exit_order_id, sl)
                            if status:
                                print(f"[SL TRAIL] New SL: {sl}")
                                sl_set_to_break_even = True
                            elif not status and "Exit at market" in message:
                                modify_sl_limit_count += 1
                                if modify_sl_limit_count >= 3:
                                    print(f"[SL CANCEL] Max modifications reached. Cancelling SL order.")
                                    exit_order_status, exit_order_id, avg_price = self.cancel_and_exit_trade(symbol, exit_order_id)
                    else:
                        if ltp >= sl:
                            exit_order_status = "COMPLETE"
                            exit_ltp = sl
                        else:
                            print(f"[SL TRAIL] New SL: {sl}")
                            sl_set_to_break_even = True
                            time.sleep(0.1)
                
            elif self.trade_method == "buy" and ltp > entry_ltp:
                #delta = int(ltp - prev_ltp)
                if sl_set_to_break_even and ltp - entry_ltp >= self.diff_to_exit:
                    print(f"@@@@@\n[EXIT CONDITION] LTP moved {self.diff_to_exit} points from entry LTP. Exiting trade.\n@@@@@")
                    if self.actual_trade:
                        exit_order_status, exit_order_id, avg_price = self.cancel_and_exit_trade(symbol, exit_order_id)
                    else:
                        exit_order_status = "COMPLETE"
                        exit_ltp = ltp

                if (ltp >= entry_ltp + self.diff_for_sl_to_match_entry_ltp) and not sl_set_to_break_even:
                    print(f">>>>>\n[SL MATCH] LTP moved {self.diff_for_sl_to_match_entry_ltp} points from entry LTP. Setting SL = Entry LTP\n>>>>>")
                    sl = entry_ltp
                    signal["sl"] = sl
                    if self.actual_trade:
                        exit_order_status = self.get_order_data(order_id=exit_order_id, value="status")
                    
                        if exit_order_status != "COMPLETE":
                            status, message = self.modify_sl(exit_order_id, sl)
                            if status:
                                print(f"[SL TRAIL] New SL: {sl}")
                                sl_set_to_break_even = True
                            elif not status and "Exit at market" in message:
                                modify_sl_limit_count += 1
                                if modify_sl_limit_count >= 3:
                                    print(f"[SL CANCEL] Max modifications reached. Cancelling SL order.")
                                    exit_order_status, exit_order_id, avg_price = self.cancel_and_exit_trade(symbol, exit_order_id)
                    else:
                        if ltp <= sl:
                            exit_order_status = "COMPLETE"
                            exit_ltp = sl
                        else:
                            print(f"[SL TRAIL] New SL: {sl}")
                            sl_set_to_break_even = True
                            time.sleep(0.1)

            if self.actual_trade:
                exit_order_status = self.get_order_data(order_id=exit_order_id, value="status")
            else:
                if self.trade_method == "buy" and ltp <= sl:
                    exit_order_status = "COMPLETE"
                    exit_ltp = ltp
                elif self.trade_method == "sell" and ltp >= sl:
                    exit_order_status = "COMPLETE"
                    exit_ltp = ltp
            if datetime.now().time() >= self.exit_time or \
                exit_order_status == "COMPLETE":
                trade_exit_time = datetime.now()
                if not self.actual_trade:
                    exit_order_status = "COMPLETE"
                
                # Set event to stop LTP fetching process
                options_and_ltp[symbol]["stop_event"] = True

            #if ltp < sl or \
            #    datetime.now().time() >= self.exit_time or \
            #    exit_order_status == "COMPLETE":
                # Print exit time order details
                if self.actual_trade:
                    print(f"[EXIT ORDER] SL Order details: {self.get_order_data(order_id=exit_order_id, value='full data')}")
                    exit_ltp = self.get_order_data(order_id=exit_order_id, value="average_price")
                else:
                    if exit_ltp is None:
                        exit_ltp = sl
                signal["exit_ltp"] = exit_ltp
                if self.trade_method == "buy":
                    signal["pnl"] = round(exit_ltp - entry_ltp, 2)
                elif self.trade_method == "sell":
                    signal["pnl"] = round(entry_ltp - exit_ltp, 2)
                signal["track_status"] = "completed"
                signal["exit_order_status"] = exit_order_status
                signal["exit_time"] = trade_exit_time

                # Update this in original dict as well
                self.signal_dict[signal_id] = signal

                print(f"[EXIT] {symbol} exited at SL: {sl}, PnL: {signal['pnl']}")
                index_symbol = signal["index_symbol"]
                filename = f"oi_data/real/{index_symbol}_pnl_{datetime.now().date()}.csv"
                df = pd.DataFrame([signal])  # signal is a dictionary with all required fields

                if not os.path.exists(filename):
                    df.to_csv(filename, index=False)
                else:
                    df.to_csv(filename, mode="a", header=False, index=False)

                print(f"[LOG] Signal exit data saved to {filename}")
                if datetime.now().time() >= self.exit_time:
                    print("Market close time reached (3:30 PM). Exiting script.")
                break

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

    def track_entry_signals(self):
        last_signal_time = None
        signal_counter = 0
        #self.signal_dict = signal_dict
        print("[START] Tracking entry signals...")
        while True:
            if datetime.now().time() >= self.exit_time:
                # Print message as Exiting script
                print("Market close time reached (3:30 PM). Exiting track_entry_signals() script.")
                break
            latest_signal = self.read_latest_signal()
            if latest_signal:
                signal_counter += 1
                signal_id = str(signal_counter)
                self.signal_dict[signal_id] = {
                    **latest_signal,
                    "signal_ltp": latest_signal["ltp"],
                    "entry_time": None,
                    "entry_ltp": None,
                    "entry_order_id": None,
                    "exit_order_id": None,
                    "exit_order_status": None,
                    "sl": None,
                    "track_status": "in_progress",
                    "exit_time": None,
                    "exit_ltp": None,
                    "max_ltp": 0,
                    "min_ltp": 999999,
                    "pnl": None
                }
                self.last_signal_time = latest_signal["datetime"]
            time.sleep(3)

    def dispatch_sl_trailing(self, options_and_ltp):
        active_signals = set()
        #self.signal_dict = signal_dict
        entry_order_id, avg_price, exit_order_id = None, None, None
        exit = False
        active_signal_last_check_time = None
        print("[START] Dispatching SL trailing for signals...")
        while True:
            if datetime.now().time() >= self.exit_time:
                # Print message as Exiting script
                print("Market close time reached (3:30 PM). Exiting dispatch_sl_trailing() script.")
                break

            # Every 30 seconds, count in progress signals
            if active_signal_last_check_time is None or \
               (datetime.now() - active_signal_last_check_time).total_seconds() >= 30:
                in_progress_signals = [sid for sid, sig in self.signal_dict.items() if sig
                ["track_status"] == "in_progress"]
                print(f"[INFO] In-progress signals count: {len(in_progress_signals)}")
                active_signal_last_check_time = datetime.now()

                # print all signals being tracked and their status
                print("############### Signal Status ###############")
                for signal_id, signal in self.signal_dict.items():
                    print(f"[STATUS] Signal ID {signal_id}: {signal['datetime']} {signal['tradingsymbol']} | Status: {signal['track_status']} | Entry LTP: {signal['entry_ltp']} | Exit LTP: {signal['exit_ltp']} | PnL: {signal['pnl']}")
                print("#############################################")

            # If the same trade symbol is already being tracked, skip adding a new one
            for signal_id, signal in self.signal_dict.items():
                if signal_id in active_signals:
                    continue
                for sid in active_signals:
                    #if signal["tradingsymbol"] == self.signal_dict[sid]["tradingsymbol"]:
                    if self.signal_dict[sid]["track_status"] == "in_progress":
                        active_signals.add(signal_id)
                        signal["track_status"] = "skipped"
                        self.signal_dict[signal_id] = signal
                        print(f"[SKIP] Signal ID {signal_id} for {self.signal_dict[signal_id]}. Currently another trade in progrees.")
                        break

            for signal_id, signal in self.signal_dict.items():
                if signal["track_status"] == "in_progress" and signal_id not in active_signals:
                    try:
                        active_signals.add(signal_id)
                        print(f"[NEW SIGNAL] Tracking signal ID {signal_id} for {signal}")
                        # Wait till the current minute becomes greater than signal time minute
                        # Get minute of current time and signal time
                        current_minute = datetime.now().minute
                        signal_minute = signal["datetime"].minute
                        #while current_minute <= signal_minute:
                        #    time.sleep(0.1)
                        #    current_minute = datetime.now().minute

                        # Check previous candle max(open/close) and compare with current candle open of the strike from index value. 
                        # Wait till a candle open greater than previous candle max is detected or 3 minutes have passed
                        flag = self.check_entry_breakout(signal)
                        if flag == "exit":
                            active_signals.add(signal_id)
                            signal["track_status"] = "completed"
                            self.signal_dict[signal_id] = signal
                            continue

                        entry_order_id, avg_price, exit_order_id, entry_time = self.detect_entry_ltp(signal["tradingsymbol"] , signal["signal_ltp"])
                        updated_signal = dict(signal)  # make a copy
                        updated_signal["entry_ltp"] = avg_price
                        updated_signal["entry_order_id"] = entry_order_id
                        updated_signal["exit_order_id"] = exit_order_id
                        updated_signal["entry_time"] = entry_time
                        self.signal_dict[signal_id] = updated_signal  # reassign to manager dict
                        if not (entry_order_id and avg_price and exit_order_id):
                            exit = True
                            updated_signal["track_status"] = "completed"
                            self.signal_dict[signal_id] = updated_signal
                            continue
                        p = Process(target=self.trail_sl_until_exit, args=(signal_id, options_and_ltp, exit))
                        p.start()
                    except Exception as e:
                        if "Insufficient funds" in str(e):
                            print(f"[ERROR] Insufficient funds for signal ID {signal_id}. Continuing to next signal.")
                        current_ltp_stop_event = dict(options_and_ltp[signal["tradingsymbol"]])
                        current_ltp_stop_event["stop_event"] = True
                        options_and_ltp[signal["tradingsymbol"]] = current_ltp_stop_event
        
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
