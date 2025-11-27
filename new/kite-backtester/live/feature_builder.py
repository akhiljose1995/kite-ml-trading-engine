# live/feature_builder.py

import pandas as pd

from indicators.ema import EMAIndicator
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.adx import ADXIndicator
from indicators.atr import ATRIndicator
from indicators.bbw import BBWIndicator
from indicators.divergence_detector import DivergenceDetector
from indicators.candle_patterns import CandlePatternDetector

from strategy.candle_pattern import CandlePatternStrategy
from strategy.ema import EMA


class FeatureBuilder:
    """
    Builds all technical indicators, candle patterns, divergences, and engineered features.
    This class is used in both training and live prediction.
    """

    def __init__(self):
        self.rsi_periods = [14]
        self.adx_periods = [14]
        self.atr_periods = [14]
        self.bbw_periods = [20]

    # ------------------------------------------------------------
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features and return fully enriched DataFrame.
        """
        df = df.copy()

        # EMA 20/50/200
        for p in [20, 50, 200]:
            ema_ind = EMAIndicator(period=p)
            df = ema_ind.compute(df)

            ema_feat = EMA(period=p)
            df = ema_feat.ema_to_price_distance(df)

        # Distance between EMAs
        ema_feat = EMA()
        df = ema_feat.distance_between_emas(df, "EMA_20", "EMA_50")
        df = ema_feat.distance_between_emas(df, "EMA_50", "EMA_200")
        df = ema_feat.distance_between_emas(df, "EMA_20", "EMA_200")

        # RSI + Divergence
        for p in self.rsi_periods:
            rsi = RSIIndicator(period=p)
            df = rsi.compute(df)

            det = DivergenceDetector(df)
            df = det.detect_rsi_divergence(period=p, order=1)

        det = DivergenceDetector(df)
        df = det.div_length_before_price_dir_change("RSI")

        # MACD + Divergence
        macd = MACDIndicator()
        df = macd.compute(df)

        det = DivergenceDetector(df)
        df = det.detect_macd_divergence(order=1)
        df = det.div_length_before_price_dir_change("MACD")

        # ADX
        for p in self.adx_periods:
            adx = ADXIndicator(period=p)
            df = adx.compute(df)

        # ATR
        for p in self.atr_periods:
            atr = ATRIndicator(period=p)
            df = atr.compute(df)

        # BBW
        for p in self.bbw_periods:
            bbw = BBWIndicator(period=p)
            df = bbw.compute(df)

        # Candle patterns
        cdl = CandlePatternDetector(df)
        df = cdl.detect_patterns()

        # Candle strategy → give price_dir, price_chg, etc.
        strat = CandlePatternStrategy(df)
        df = strat.run(trading_type="swing")

        # Cleanup
        df.fillna(method="ffill", inplace=True)
        df.fillna(0, inplace=True)

        return df
