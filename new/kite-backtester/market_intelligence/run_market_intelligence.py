"""
Entry point for Market Intelligence system
"""

import os
import time
from datetime import datetime, timedelta

# -----------------------------
# Config loaders
# -----------------------------
import config
from market_intelligence.config.config_loader import ConfigLoader
from market_intelligence.config.instrument_config import InstrumentConfig
from market_intelligence.config.market_session_config import MarketSessionConfig
from market_intelligence.config.llm_settings import LLMSettings
from market_intelligence.config.telegram_settings import TelegramSettings

# -----------------------------
# Context
# -----------------------------
from market_intelligence.context.instrument_context import InstrumentContext
from market_intelligence.context.session_context import SessionContext

# -----------------------------
# Data capture
# -----------------------------
from data.kite_loader import KiteDataLoader
from general_lib.indicator_pipeline import IndicatorPipelineBuilder
from general_lib.swing_detector import SwingDetector, SwingConfig, TimeframeConfig
from general_lib.zone_clustering import ZoneCluster, ZoneClusterConfig
from market_intelligence.data_capture.candle_snapshot import CandleSnapshot
from market_intelligence.data_capture.indicator_snapshot import IndicatorSnapshot
from market_intelligence.data_capture.pdh_pdl import PDHPDLExtractor
from market_intelligence.data_capture.sr_snapshot import SupportResistanceSnapshot

# -----------------------------
# News
# -----------------------------
from market_intelligence.config.new_config import NewsSettings
from market_intelligence.news.news_fetcher import NewsDataIOFetcher

# -----------------------------
# LLM
# -----------------------------
from market_intelligence.llm.llm_client import OpenAIClient
from market_intelligence.llm.conversation_manager import ConversationManager
from market_intelligence.llm.cost_guard import CostGuard

# -----------------------------
# Telegram
# -----------------------------
from market_intelligence.telegram.notifier import TelegramNotifier

# -----------------------------
# Runners
# -----------------------------
from market_intelligence.runners.pre_market_runner import PreMarketRunner
from market_intelligence.runners.live_update_runner import LiveUpdateRunner


def main():
    print("Starting Market Intelligence System")

    base_path = "market_intelligence/config"

    # -----------------------------
    # Load configs
    # -----------------------------
    instrument_cfg = InstrumentConfig(
        ConfigLoader.load_yaml(f"{base_path}/instruments.yaml")
    )
    session_cfg = MarketSessionConfig(
        ConfigLoader.load_yaml(f"{base_path}/market_sessions.yaml")
    )
    news_cfg = NewsSettings(
        ConfigLoader.load_yaml(f"{base_path}/news_config.yaml")
    )
    llm_cfg = LLMSettings(
        ConfigLoader.load_yaml(f"{base_path}/llm_config.yaml")
    )
    telegram_cfg = TelegramSettings(
        ConfigLoader.load_yaml(f"{base_path}/telegram_config.yaml")
    )

    # -----------------------------
    # Choose instrument
    # -----------------------------
    instrument_key = "banknifty"
    inst = instrument_cfg.get(instrument_key)

    instrument_ctx = InstrumentContext(
        instrument_key=instrument_key,
        token=inst["token"],
        symbol=inst["symbol"],
        exchange=inst["exchange"],
        instrument_type=inst["type"],
        timezone=inst["timezone"],
        tick_size=inst["tick_size"],
    )

    session_data = session_cfg.get_session(inst["exchange"])
    session_ctx = SessionContext(
        exchange=inst["exchange"],
        timezone=session_data["timezone"],
        market_open=session_data["market_open"],
        market_close=session_data["market_close"],
        pre_open_start=session_data.get("pre_open_start"),
        holidays=session_data.get("holidays", []),
    )

    # -----------------------------
    # Kite loader
    # -----------------------------
    kite_loader = KiteDataLoader(
        api_key=config.API_KEY,
        access_token=config.ACCESS_TOKEN,
        request_token=config.REQUEST_TOKEN,
        api_secret=config.API_SECRET
    )

    candle_snapshot = CandleSnapshot(kite_loader)

    # -----------------------------
    # Indicator pipeline (example)
    # -----------------------------
    indicator_pipeline = IndicatorPipelineBuilder.build_default_pipeline(
    ema_periods=(20, 50, 200),
    enable_divergence=True,
    enable_candles=True,  # IMPORTANT for LLM snapshot
    )
    #indicator_pipeline = [
    #    lambda df: df,  # placeholder, plug your real pipeline
    #]
    indicator_snapshot = IndicatorSnapshot(indicator_pipeline)

    # -----------------------------
    # SR snapshot (reuse your configs)
    # -----------------------------
    swing_config=SwingConfig(
                interval="1D",
                timeframe_params={
                    "1D": TimeframeConfig(lookback=7, atr_period=14, atr_multiplier=0.7),
                    "60minute": TimeframeConfig(lookback=6, atr_period=14, atr_multiplier=0.6),
                    "15minute": TimeframeConfig(lookback=5, atr_period=14, atr_multiplier=0.5),
                    "5minute": TimeframeConfig(lookback=4, atr_period=14, atr_multiplier=0.4),
                    "minute": TimeframeConfig(lookback=3, atr_period=14, atr_multiplier=0.3),
                },
                price_col_map={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                }
            )
    detector = SwingDetector(swing_config)
    sr_snapshot = SupportResistanceSnapshot(
        detector=detector,
        zone_cluster=ZoneCluster()
    )

    # -----------------------------
    # News
    # -----------------------------
    if news_cfg.enabled:
        news_fetcher = NewsDataIOFetcher(
            api_key=os.getenv("NEWSDATA_API_KEY"),
            max_results=10,
        )
    else:
        news_fetcher = None
    # -----------------------------
    # LLM
    # -----------------------------
    llm_client = None
    conversation_manager = ConversationManager(max_turns=10)
    cost_guard = CostGuard(
        max_daily_tokens=int(llm_cfg.cost_guard["max_daily_tokens"]),
        max_prompt_tokens=int(llm_cfg.cost_guard["max_prompt_tokens"]),
        enabled=llm_cfg.cost_guard["enabled"],
    )

    if llm_cfg.enabled:
        llm_client = OpenAIClient(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )

    # -----------------------------
    # Telegram
    # -----------------------------
    telegram = TelegramNotifier(
        bot_token=telegram_cfg.bot_token,
        chat_id=telegram_cfg.chat_id,
        enabled=telegram_cfg.enabled,
        parse_mode=telegram_cfg.parse_mode,
        rate_limit_sec=telegram_cfg.rate_limit_sec,
    )

    # -----------------------------
    # Pre-market run (ONCE)
    # -----------------------------
    pre_runner = PreMarketRunner(
        instrument_ctx=instrument_ctx,
        session_ctx=session_ctx,
        candle_snapshot=candle_snapshot,
        indicator_snapshot=indicator_snapshot,
        pdh_pdl_extractor=PDHPDLExtractor(),
        sr_snapshot=sr_snapshot,
        news_fetcher=news_fetcher,
        llm_client=llm_client,
        conversation_manager=conversation_manager,
        cost_guard=cost_guard,
        telegram_notifier=telegram,
        timezone=instrument_ctx.timezone,
    )

    pre_runner.run()

    # -----------------------------
    # Live updates (15m)
    # -----------------------------
    live_runner = LiveUpdateRunner(
        instrument_ctx=instrument_ctx,
        session_ctx=session_ctx,
        candle_snapshot=candle_snapshot,
        indicator_snapshot=indicator_snapshot,
        sr_snapshot=sr_snapshot,
        llm_client=llm_client,
        conversation_manager=conversation_manager,
        cost_guard=cost_guard,
        telegram_notifier=telegram,
        timeframe="15minute",
        timezone=instrument_ctx.timezone,
    )

    live_runner.run()


if __name__ == "__main__":
    main()