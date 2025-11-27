import datetime
import time
import traceback
from typing import Optional

import pandas as pd

from live.fetcher import LiveFetcher
from live.feature_builder import FeatureBuilder
from models.predictor.predict_engine import PredictEngine
from models.preprocessing.preprocessor_predict import PreprocessorPredict

try:
    from telegram_bot.telegram_bot import TelegramBot
except Exception:
    TelegramBot = None


class LivePredictLoop:
    """
    Continuous live prediction loop:
    Fetch → Feature Build → Preprocess → Predict → Sleep until next candle.
    Guarantees accurate candle scheduling for 1m/5m/15m/60m intervals.
    """

    INTERVAL_MAP = {
        "1minute": 1,
        "3minute": 3,
        "5minute": 5,
        "15minute": 15,
        "60minute": 60
    }

    def __init__(
        self,
        instrument_token: int,
        interval: str,
        model_path: str,
        encoder_dir: str = "models/encoders",
        scaler_dir: str = "models/scalers",
        telegram_enabled: bool = False,
        telegram_chat_id: Optional[str] = None,
    ):
        self.instrument_token = instrument_token
        self.interval = interval

        # Core engine
        self.fetcher = LiveFetcher(instrument_token, interval)
        self.builder = FeatureBuilder()
        self.predict_engine = PredictEngine(model_path=model_path)
        self.preprocessor = PreprocessorPredict(
            encoder_dir=encoder_dir,
            scaler_dir=scaler_dir
        )

        # Telegram
        self.telegram_enabled = telegram_enabled
        self.bot = TelegramBot() if (telegram_enabled and TelegramBot) else None
        self.chat_id = telegram_chat_id

        if self.interval not in self.INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {self.interval}")

    # =====================================================================
    # TIME / SCHEDULING
    # =====================================================================

    def get_next_candle_time(self, last_ts: datetime.datetime) -> datetime.datetime:
        """Compute next candle close time based on the interval."""
        step = self.INTERVAL_MAP[self.interval]
        return last_ts + datetime.timedelta(minutes=step)

    def sleep_until(self, target_time: datetime.datetime):
        """
        Sleep until target_time with drift correction and 1-sec granularity.
        Handles microsecond precision.
        """
        while True:
            now = datetime.datetime.now()
            delta = (target_time - now).total_seconds()

            if delta <= 0:
                return

            # Avoid long blocking sleep; correct drift in smaller hops
            time.sleep(min(delta, 1))

    # =====================================================================
    # PREDICTION LOGIC
    # =====================================================================

    def run_once(self, raw_df: pd.DataFrame) -> Optional[dict]:
        """
        Executes a single prediction cycle using provided raw candle DataFrame.
        """
        try:
            # Step 1: Feature engineering
            feat_df = self.builder.build(raw_df)

            # Step 2: Predict last completed candle
            last_row_df = feat_df.tail(1).copy()

            # Step 3: Preprocess
            X = self.preprocessor.transform_for_prediction(last_row_df)

            # Step 4: Prediction
            pred_label, pred_prob = self.predict_engine.predict(X)

            ts = (
                last_row_df["date"].iloc[-1]
                if "date" in last_row_df.columns
                else None
            )

            result = {
                "timestamp": ts,
                "prediction": pred_label,
                "probability": pred_prob,
                "interval": self.interval,
            }

            # Print nicely
            print("\n============================")
            print("🔮 LIVE PREDICTION RESULT")
            print("============================")
            print(result)
            print("============================\n")

            # Telegram alert
            if self.telegram_enabled and self.bot and self.chat_id:
                self.bot.send_message(self.chat_id, f"Prediction:\n{result}")

            return result

        except Exception as e:
            print("❌ ERROR in run_once:", e)
            print(traceback.format_exc())
            return None

    # =====================================================================
    # MAIN LOOP
    # =====================================================================

    def run_forever(self):
        """
        Continuous:
            - Fetch last completed candle
            - Predict it
            - Sleep until next candle close
        """
        print("\n🚀 Live Prediction Loop Started...\n")

        while True:
            try:
                # ==========================================================
                # Fetch recent OHLC candles
                # ==========================================================
                raw_df = self.fetcher.get_recent_ohlc()

                if raw_df is None or raw_df.empty:
                    print("⚠️ No data received. Retrying in 1s...")
                    time.sleep(1)
                    continue

                # ==========================================================
                # Last completed candle timestamp
                # ==========================================================
                last_ts = raw_df["date"].iloc[-1]
                now = datetime.datetime.now()

                # Compute official next candle time
                next_ts = self.get_next_candle_time(last_ts)

                # ==========================================================
                # Case 1: Candle just closed → Predict it
                # ==========================================================
                if now < next_ts:
                    print(f"📊 Predicting candle: {last_ts}")
                    self.run_once(raw_df)

                    print(f"⏳ Next candle at {next_ts}")
                    self.sleep_until(next_ts)
                    continue

                # ==========================================================
                # Case 2: Clock is ahead of data (delayed API candle)
                # Just refetch until aligned
                # ==========================================================
                else:
                    print("⚠️ API delay detected. Waiting for fresh candle...")
                    time.sleep(1)
                    continue

            except KeyboardInterrupt:
                print("\n🛑 Live Loop Stopped Manually.")
                break

            except Exception as e:
                print("❌ FATAL ERROR in main loop:", e)
                print(traceback.format_exc())
                time.sleep(1)
