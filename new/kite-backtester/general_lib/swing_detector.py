import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict


# -------------------------------------------------
# Configuration
# -------------------------------------------------

@dataclass
class TimeframeConfig:
    lookback: int
    atr_period: int
    atr_multiplier: float


@dataclass
class SwingConfig:
    interval: str
    timeframe_params: Dict[str, TimeframeConfig]
    price_col_map: Dict[str, str] = None


# -------------------------------------------------
# Swing Detector
# -------------------------------------------------

class SwingDetector:
    """
    Detects swing highs and swing lows using window-based logic
    with ATR-based significance filtering.

    Outputs both flags and actual swing price values.
    """

    def __init__(self, config: SwingConfig):
        self.config = config
        self.tf_config = self._load_timeframe_config(config.interval)

        self.price_col_map = config.price_col_map or {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close"
        }

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_dataframe(df)
        df = self._calculate_atr(df)

        # Detect raw swings
        df["swing_high"] = self._find_swing_highs(df)
        df["swing_low"] = self._find_swing_lows(df)

        # Filter weak swings
        df = self._filter_weak_swings(df)

        # Attach swing prices
        df = self._attach_swing_prices(df)

        return df

    # -------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------

    def _load_timeframe_config(self, interval: str) -> TimeframeConfig:
        if interval not in self.config.timeframe_params:
            raise ValueError(f"No swing configuration found for interval: {interval}")
        return self.config.timeframe_params[interval]

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        h = self.price_col_map["high"]
        l = self.price_col_map["low"]
        c = self.price_col_map["close"]

        high_low = df[h] - df[l]
        high_close = (df[h] - df[c].shift()).abs()
        low_close = (df[l] - df[c].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(self.tf_config.atr_period).mean()

        return df

    def _find_swing_highs(self, df: pd.DataFrame) -> pd.Series:
        h = self.price_col_map["high"]
        n = self.tf_config.lookback

        return df[h] == df[h].rolling(window=2 * n + 1, center=True).max()

    def _find_swing_lows(self, df: pd.DataFrame) -> pd.Series:
        l = self.price_col_map["low"]
        n = self.tf_config.lookback

        return df[l] == df[l].rolling(window=2 * n + 1, center=True).min()

    def _filter_weak_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        h = self.price_col_map["high"]
        l = self.price_col_map["low"]

        min_move = df["atr"] * self.tf_config.atr_multiplier

        # Swing High → require meaningful drop after
        df.loc[
            (df["swing_high"]) &
            ((df[h] - df[l].shift(-1)) < min_move),
            "swing_high"
        ] = False

        # Swing Low → require meaningful rise after
        df.loc[
            (df["swing_low"]) &
            ((df[h].shift(-1) - df[l]) < min_move),
            "swing_low"
        ] = False

        return df

    def _attach_swing_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        h = self.price_col_map["high"]
        l = self.price_col_map["low"]

        df["swing_high_price"] = np.where(df["swing_high"], df[h], np.nan)
        df["swing_low_price"] = np.where(df["swing_low"], df[l], np.nan)

        return df