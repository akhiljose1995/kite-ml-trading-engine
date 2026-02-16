from market_intelligence.context.time_context import TimeContext
from market_intelligence.context.session_context import SessionContext
from market_intelligence.context.instrument_context import InstrumentContext

from market_intelligence.prompts.prompt_start import StartPromptBuilder
from market_intelligence.news.news_filter import NewsFilter
import datetime

class PreMarketRunner:
    """
    Executes one-time pre-market intelligence generation.
    """

    def __init__(
        self,
        *,
        instrument_ctx: InstrumentContext,
        session_ctx: SessionContext,
        candle_snapshot,
        indicator_snapshot,
        pdh_pdl_extractor,
        sr_snapshot,
        news_fetcher,
        llm_client,
        conversation_manager,
        cost_guard,
        telegram_notifier,
        timezone: str,
    ):
        self.instrument_ctx = instrument_ctx
        self.session_ctx = session_ctx
        self.candle_snapshot = candle_snapshot
        self.indicator_snapshot = indicator_snapshot
        self.pdh_pdl_extractor = pdh_pdl_extractor
        self.sr_snapshot = sr_snapshot
        self.news_fetcher = news_fetcher
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.cost_guard = cost_guard
        self.telegram = telegram_notifier
        self.timezone = timezone

    def run(self) -> None:
        time_ctx = TimeContext(self.timezone)
        session_state = self.session_ctx.session_state(time_ctx.get_now())

        # -----------------------
        # 1. Market data capture
        # -----------------------
        # start_date is previous day, market open time. Fetch it from market_session
        days=30
        prev_date = time_ctx.get_now().date() - datetime.timedelta(days=days)
        market_open_time = self.session_ctx.market_open
        start_date = datetime.datetime.combine(prev_date, market_open_time)
        #end_date=time_ctx.now + datetime.timedelta(minutes=1)
        end_date = datetime.datetime.now() + datetime.timedelta(minutes=1)
        #print(f"Fetching market data from {start_date} to {end_date}...")
        candles = self.candle_snapshot.fetch(
            instrument_token=self.instrument_ctx.token,
            intervals=["15minute", "60minute"],
            start_date=start_date,
            end_date=end_date,  # make end date as current date + 1 minute to ensure we get the latest data
        )
        
        indicators_15m = self.indicator_snapshot.capture(candles["15minute"])
        indicators_1h = self.indicator_snapshot.capture(candles["60minute"])

        # Only consider the below columns for the prompt to save tokens
        # date, open, high, low, close, volume, interval, EMA_20, EMA_20_dist_to_price, 
        # EMA_50, EMA_50_dist_to_price, EMA_200, EMA_200_dist_to_price, RSI_14, RSI_14_Div_Type, 
        # RSI_14_Div_Length, RSI_14_Div_Len_Nearest_Past, MACD, MACD_Signal, MACD_Hist, 
        # MACD_Div_Type, MACD_Div_Length, MACD_Div_Len_Nearest_Past, ADX_14, PDI_14, NDI_14, 
        # ADX_14_Strength_Label, ATR_14,BBW_20, candle
        cols_to_keep = [
            "date", "open", "high", "low", "close", "interval",
            "EMA_20", "EMA_20_dist_to_price",
            "EMA_50", "EMA_50_dist_to_price",
            "EMA_200", "EMA_200_dist_to_price",
            "RSI_14", "RSI_14_Div_Type", "RSI_14_Div_Length", "RSI_14_Div_Len_Nearest_Past",
            "MACD", "MACD_Signal", "MACD_Hist", "MACD_Div_Type", "MACD_Div_Length", "MACD_Div_Len_Nearest_Past",
            "ADX_14", "PDI_14", "NDI_14", "ADX_14_Strength_Label",
            "ATR_14","BBW_20", "candle"
        ]
        indicators_15m = indicators_15m[cols_to_keep]
        indicators_1h = indicators_1h[cols_to_keep]

        # For 60minute, consider only last 7 rows
        indicators_1h = indicators_1h.iloc[-7:]
        # For 15minute, consider only last 25 rows
        indicators_15m = indicators_15m.iloc[-25:]

        # Fetch 1D data to extract PDH/PDL and SR zones
        current_price = candles["15minute"].iloc[-1]["close"]
        candles = self.candle_snapshot.fetch(
            instrument_token=self.instrument_ctx.token,
            intervals=["1D"],
            start_date=datetime.datetime(2024, 1, 1),
            # make end date as current date + 1 minute to ensure we get the latest data
            end_date=datetime.datetime.now() + datetime.timedelta(minutes=1),
        )
        indicators_1d = self.indicator_snapshot.capture(candles["1D"])
        indicators_1d = indicators_1d[cols_to_keep]
        # For 1D, consider only last 10 rows
        indicators_1d = indicators_1d.iloc[-6:]

        prev_day, pdh, pdl = self.pdh_pdl_extractor.extract(candles["1D"])

        sr_zones = self.sr_snapshot.capture(
            df=candles["1D"],
            current_price=current_price,
        )

        # -----------------------
        # 2. News (ONCE)
        # -----------------------
        relevant_news = None
        if self.news_fetcher:
            raw_news = self.news_fetcher.fetch(
                query=self.instrument_ctx.symbol,
                country="in" if self.instrument_ctx.exchange == "NSE" else None,
            )

            news_filter = NewsFilter(max_age_hours=24)
            relevant_news = news_filter.filter(
                news=raw_news,
                instrument_ctx=self.instrument_ctx,
                now=time_ctx.get_now(),
            )

        # -----------------------
        # 3. Build prompt
        # -----------------------
        prompt = StartPromptBuilder().build(
            instrument_name=self.instrument_ctx.symbol,
            time_context={
                "now": time_ctx.get_now(),
                "session_state": session_state,
            },
            indicators_15m=indicators_15m,
            indicators_1h=indicators_1h,
            indicators_1d=indicators_1d,
            prev_day=prev_day,
            pdh=pdh,
            pdl=pdl,
            current_price=current_price,
            sr_zones=sr_zones,
            news=relevant_news,
        )

        # Save prompt to file for debugging
        with open("debug_premarket_prompt.txt", "w") as f:
            f.write(prompt)
        # -----------------------
        # 4. Telegram (prompt)
        # -----------------------
        self.telegram.send("[Pre-market Analysis]\n", notify=True)
        self.telegram.send(prompt)

        # -----------------------
        # 5. Optional LLM
        # -----------------------
        if self.llm_client and self.cost_guard.is_allowed(prompt):
            response, usage = self.llm_client.generate(
                prompt=prompt,
                history=self.conversation_manager.get_history(),
            )

            self.conversation_manager.add_user(prompt)
            self.conversation_manager.add_assistant(response)
            self.cost_guard.record_usage(usage)

            self.telegram.send("[Pre-market Analysis]\n", notify=True)
            self.telegram.send(response)