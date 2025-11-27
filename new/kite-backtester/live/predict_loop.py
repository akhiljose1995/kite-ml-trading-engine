# live/predict_loop.py

import datetime
import time
import traceback
import pandas as pd
from typing import Optional

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
    Continuously fetches OHLC data, builds features, preprocesses inputs,
    generates model predictions, and optionally alerts via Telegram.
    """

    def __init__(
        self,
        instrument_token: int,
        interval: str,
        model_path: str,
        encoder_dir: str = "models/encoders",
        scaler_dir: str = "models/scalers",
        sleep_seconds: int = 10,
        telegram_enabled: bool = False,
        telegram_chat_id: Optional[str] = None
    ):
        """
        Args:
            instrument_token (int): Kite token for stock/option.
            interval (str): One of ["5minute", "15minute", "60minute"].
            model_path (str): Path to saved trained model .pkl.
            encoder_dir (str): Directory of saved encoders.
            scaler_dir (str): Directory of saved scalers.
            sleep_seconds (int): Delay between polling cycles.
            telegram_enabled (bool): Enable Telegram alerts.
            telegram_chat_id (str): Chat ID if Telegram enabled.
        """
        self.instrument_token = instrument_token
        self.interval = interval
        self.sleep_seconds = sleep_seconds

        # Core components
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

    # --------------------------------------------------------------------

    def sleep_until_next_candle(interval: str):
        now = datetime.datetime.now()

        minute = now.minute
        second = now.second
        micro = now.microsecond

        if interval == "5minute":
            step = 5
        elif interval == "15minute":
            step = 15
        elif interval == "60minute":
            step = 60
        else:
            step = 5

        # Compute next candle minute mark
        next_minute = (minute // step + 1) * step

        # If next_minute hits 60, roll over hour
        delta_minutes = next_minute - minute
        if next_minute == 60:
            next_time = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        else:
            next_time = now.replace(minute=next_minute, second=0, microsecond=0)

        sleep_seconds = (next_time - now).total_seconds()

        print(f"⏳ Sleeping for {sleep_seconds:.1f} seconds until next {interval} candle closes at {next_time}.")
        time.sleep(max(1, sleep_seconds))

    # --------------------------------------------------------------------

    def run_once(self) -> Optional[dict]:
        """
        Runs a single prediction cycle:
        Fetch → Build features → Preprocess → Predict last candle.

        Returns:
            dict with prediction results, or None if failed.
        """
        try:
            # Step 1: Fetch last candles
            raw_df = self.fetcher.get_recent_ohlc()
            if raw_df is None or raw_df.empty:
                print("⚠️ No data fetched. Retrying...")
                return None

            # Step 2: Build feature-engineered frame
            feat_df = self.builder.build(raw_df)

            # Step 3: Only last row is used for prediction
            last_row_df = feat_df.tail(1).copy()

            # Step 4: Preprocess for inference
            X = self.preprocessor.transform_for_prediction(last_row_df)

            # Step 5: Predict label + probability
            pred_label, pred_prob = self.predict_engine.predict(X)

            result = {
                "timestamp": last_row_df["date"].iloc[-1] if "date" in last_row_df else None,
                "prediction": pred_label,
                "probability": pred_prob,
                "interval": self.interval
            }

            # Print result
            print("\n============================")
            print("🔮 LIVE PREDICTION RESULT")
            print("============================")
            print(result)
            print("============================\n")

            # Optional: Telegram alert
            if self.telegram_enabled and self.bot and self.chat_id:
                self.bot.send_message(self.chat_id, f"Prediction: {result}")

            return result

        except Exception as e:
            print("❌ ERROR in run_once:", e)
            print(traceback.format_exc())
            return None

    # --------------------------------------------------------------------

    def run_forever(self):
        """
        Infinite loop: runs prediction every sleep_seconds.
        Ideal for local script OR background FastAPI task.
        """
        print("\n🚀 Live Prediction Loop Started...\n")

        while True:
            try:
                self.run_once()
                # Sleep exactly until next candle close
                self.sleep_until_next_candle(self.interval)
            except KeyboardInterrupt:
                print("\n🛑 Live loop stopped manually.")
                break
            except Exception as e:
                print("❌ Fatal error:", e)
                print(traceback.format_exc())
                time.sleep(self.sleep_seconds)
