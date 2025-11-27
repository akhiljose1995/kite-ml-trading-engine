# live/fetcher.py
import datetime
import pandas as pd
from data.kite_loader import KiteDataLoader
import config


class LiveFetcher:
    """
    Fetches recent OHLC data for live prediction.
    Handles dynamic lookback depending on interval.
    """

    INTERVAL_MINUTES = {
        "5minute": 5,
        "15minute": 15,
        "60minute": 60
    }

    def __init__(self, instrument_token: int, interval: str, lookback: int = 80):
        """
        Args:
            instrument_token (int): Kite token for the instrument.
            interval (str): One of ["5minute", "15minute", "60minute"].
            lookback (int): Number of candles to fetch for feature building.
        """
        self.instrument_token = instrument_token
        self.interval = interval
        self.lookback = lookback

        self.loader = KiteDataLoader(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            access_token=config.ACCESS_TOKEN,
            request_token=getattr(config, "REQUEST_TOKEN", None)
        )

    # -------------------------------------------------------------

    def get_recent_ohlc(self) -> pd.DataFrame:
        """
        Fetch last N candles needed for prediction.
        Ensures date index, sorting, clean frame.
        """
        minutes = self.INTERVAL_MINUTES.get(self.interval, 15)
        to_dt = datetime.datetime.now()
        from_dt = to_dt - datetime.timedelta(minutes=minutes * (self.lookback + 5))

        df = self.loader.get_data(
            instrument_token=self.instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=self.interval
        )

        if df.empty:
            return df

        df = df.reset_index().rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        return df
