from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd


class CandleSnapshot:
    """
    Fetches and prepares candle data for multiple timeframes.
    """

    def __init__(self, data_loader):
        """
        data_loader: KiteDataLoader or compatible loader
        """
        self.loader = data_loader

    def fetch(
        self,
        instrument_token: int,
        intervals: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:

        data = {}

        for interval in intervals:
            df = self.loader.get_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval,
            )

            if df is None or df.empty:
                continue

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            data[interval] = df

        return data