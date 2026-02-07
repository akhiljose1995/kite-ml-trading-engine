import pandas as pd


class TimeframeConverter:
    """
    Convert OHLC data from any lower timeframe (1m, 5m, 15m, 1H)
    into higher timeframe candles (1H, 4H, 1D, 1W, etc.)
    """

    def __init__(self, df, date_col="date"):
        self.df = df.copy()
        self.date_col = date_col
        self._prepare()

    def _prepare(self):
        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df = self.df.set_index(self.date_col)

        # Ensure sorted
        self.df = self.df.sort_index()

        # Validate required columns
        required = {"open", "high", "low", "close"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {missing}")

    def convert(self, timeframe):
        """
        Convert to any higher timeframe using OHLC rules.
        Example: '1D', '4H', '1W', '1M'
        """
        result = self.df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        # Add Date column back (first column) 
        #result = result.reset_index().rename(columns={"index": self.date_col})
        result["interval"] = timeframe

        return result.dropna()

    def to_daily(self):
        """Shortcut for 1D candles."""
        return self.convert("1D")

    def to_weekly(self):
        """Shortcut for 1W candles."""
        return self.convert("1W")

    def to_monthly(self):
        """Shortcut for 1M candles."""
        return self.convert("1M")
