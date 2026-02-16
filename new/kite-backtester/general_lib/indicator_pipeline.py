from typing import List, Callable
import pandas as pd

# Indicators
from indicators.ema import EMAIndicator
from strategy.ema import EMA
from indicators.rsi import RSIIndicator
from indicators.macd import MACDIndicator
from indicators.adx import ADXIndicator
from indicators.atr import ATRIndicator
from indicators.bbw import BBWIndicator
from indicators.divergence_detector import DivergenceDetector
from indicators.candle_patterns import CandlePatternDetector
from strategy.candle_pattern import CandlePatternStrategy


class IndicatorPipelineBuilder:
    """
    Builds reusable indicator pipelines (df -> df).
    """

    @staticmethod
    def build_default_pipeline(
        *,
        ema_periods=(20, 50, 200),
        rsi_periods=(14,),
        adx_periods=(14,),
        atr_periods=(14,),
        bbw_periods=(20,),
        enable_divergence=True,
        enable_candles=True,
    ) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:

        pipeline: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

        # -------------------------
        # EMA calculations
        # -------------------------
        for period in ema_periods:
            pipeline.append(
                lambda df, p=period: EMAIndicator().ema_calc(df, period=p)
            )
            pipeline.append(
                lambda df, p=period: EMA(period=p).ema_to_price_distance(df)
            )

        # -------------------------
        # EMA distance between EMAs
        # -------------------------
        def ema_inter_distances(df: pd.DataFrame) -> pd.DataFrame:
            ema = EMA()
            if {"EMA_20", "EMA_50"}.issubset(df.columns):
                df = ema.distance_between_emas(df, "EMA_20", "EMA_50")
            if {"EMA_50", "EMA_200"}.issubset(df.columns):
                df = ema.distance_between_emas(df, "EMA_50", "EMA_200")
            if {"EMA_20", "EMA_200"}.issubset(df.columns):
                df = ema.distance_between_emas(df, "EMA_20", "EMA_200")
            return df

        pipeline.append(ema_inter_distances)

        # -------------------------
        # RSI + divergence
        # -------------------------
        for period in rsi_periods:
            pipeline.append(
                lambda df, p=period: RSIIndicator(period=p).compute(df)
            )

            if enable_divergence:
                pipeline.append(
                    lambda df, p=period: DivergenceDetector(df)
                    .detect_rsi_divergence(period=p, order=1)
                )

        if enable_divergence:
            pipeline.append(
                lambda df: DivergenceDetector(df)
                .div_length_before_price_dir_change(column="RSI")
            )

        # -------------------------
        # MACD + divergence
        # -------------------------
        pipeline.append(lambda df: MACDIndicator().compute(df))

        if enable_divergence:
            pipeline.append(
                lambda df: DivergenceDetector(df)
                .detect_macd_divergence(order=1)
            )
            pipeline.append(
                lambda df: DivergenceDetector(df)
                .div_length_before_price_dir_change(column="MACD")
            )

        # -------------------------
        # ADX
        # -------------------------
        for period in adx_periods:
            pipeline.append(
                lambda df, p=period: ADXIndicator(period=p).compute(df)
            )

        # -------------------------
        # ATR
        # -------------------------
        for period in atr_periods:
            pipeline.append(
                lambda df, p=period: ATRIndicator(period=p).compute(df)
            )

        # -------------------------
        # BBW
        # -------------------------
        for period in bbw_periods:
            pipeline.append(
                lambda df, p=period: BBWIndicator(period=p).compute(df)
            )

        # -------------------------
        # Candle patterns + strategy
        # -------------------------
        if enable_candles:
            def candle_pipeline(df: pd.DataFrame) -> pd.DataFrame:
                cdl_detector = CandlePatternDetector(df)
                pattern_df = cdl_detector.detect_patterns()

                strategy = CandlePatternStrategy(pattern_df)
                return strategy.run(trading_type="swing")

            pipeline.append(candle_pipeline)

        return pipeline