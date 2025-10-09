from kiteconnect import KiteConnect, KiteTicker
import time
import threading

class TrailingStopLossTrader:
    def __init__(self, api_key, access_token, symbol, qty, entry_price, initial_sl, trailing_gap=2, trail_step=1):
        self.api_key = api_key
        self.access_token = access_token
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.initial_sl = initial_sl
        self.trailing_gap = trailing_gap
        self.trail_step = trail_step
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)
        self.sl_order_id = None
        self.buy_order_id = None
        self.last_sl = initial_sl
        self.current_price = entry_price
        self.trail_active = True

    # ----------------------------
    # 1. Place BUY order
    # ----------------------------
    def place_buy_order(self):
        print(f"[INFO] Placing BUY order @ {self.entry_price}")
        self.buy_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=self.symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_BUY,
            quantity=self.qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_LIMIT,
            price=self.entry_price
        )
        print("[INFO] Buy order placed. Waiting for execution...")
        self.wait_for_execution(self.buy_order_id)

    # ----------------------------
    # 2. Wait for order execution
    # ----------------------------
    def wait_for_execution(self, order_id):
        while True:
            order = self.kite.order_history(order_id)
            if order[-1]["status"] == "COMPLETE":
                print(f"[INFO] Buy order executed @ {order[-1]['average_price']}")
                break
            time.sleep(2)

    # ----------------------------
    # 3. Place initial SL-M order
    # ----------------------------
    def place_initial_sl(self):
        print(f"[INFO] Placing initial SL-M @ {self.initial_sl}")
        self.sl_order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            exchange=self.kite.EXCHANGE_NFO,
            tradingsymbol=self.symbol,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL,
            quantity=self.qty,
            product=self.kite.PRODUCT_MIS,
            order_type=self.kite.ORDER_TYPE_SL_MARKET,
            trigger_price=self.initial_sl
        )
        print(f"[INFO] SL-M placed at {self.initial_sl}. ID: {self.sl_order_id}")

    # ----------------------------
    # 4. Modify SL dynamically
    # ----------------------------
    def modify_sl(self, new_sl):
        if not self.sl_order_id:
            return
        try:
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=self.sl_order_id,
                trigger_price=new_sl
            )
            print(f"[UPDATE] SL modified to {new_sl}")
            self.last_sl = new_sl
        except Exception as e:
            print(f"[ERROR] Failed to modify SL: {e}")

    # ----------------------------
    # 5. Track price and trail SL
    # ----------------------------
    def start_trailing(self):
        print("[INFO] Starting trailing stop loss monitoring...")
        kws = KiteTicker(self.api_key, self.access_token)

        def on_ticks(ws, ticks):
            if not ticks:
                return
            self.current_price = ticks[0]['last_price']
            self.handle_trailing_logic()

        def on_connect(ws, response):
            token = self.kite.ltp(f"NFO:{self.symbol}")[f"NFO:{self.symbol}"]["instrument_token"]
            ws.subscribe([token])
            ws.set_mode(ws.MODE_LTP, [token])
            print(f"[INFO] Subscribed to live LTP for {self.symbol}")

        def on_close(ws, code, reason):
            print("[INFO] WebSocket closed:", code, reason)
            self.trail_active = False

        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close

        # Run websocket in a separate thread
        wst = threading.Thread(target=kws.connect, daemon=True)
        wst.start()

        while self.trail_active:
            time.sleep(1)
            if not self.check_position_open():
                print("[INFO] Position exited. Stopping trailing.")
                self.trail_active = False
                break

    # ----------------------------
    # 6. Handle trailing logic
    # ----------------------------
    def handle_trailing_logic(self):
        new_sl = round(self.current_price - self.trailing_gap, 1)
        if new_sl >= self.last_sl + self.trail_step:
            self.modify_sl(new_sl)

    # ----------------------------
    # 7️. Check if position still open
    # ----------------------------
    def check_position_open(self):
        positions = self.kite.positions()["net"]
        pos = next((p for p in positions if p["tradingsymbol"] == self.symbol), None)
        return bool(pos and pos["quantity"] != 0)


# ==========================================================
# Example usage
# ==========================================================
if __name__ == "__main__":
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
    trader.start_trailing()
