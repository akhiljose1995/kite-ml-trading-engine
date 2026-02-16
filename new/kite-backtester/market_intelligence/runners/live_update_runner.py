import time
import datetime
from datetime import datetime as dt

from market_intelligence.prompts.prompt_update import UpdatePromptBuilder
from market_intelligence.context.time_context import TimeContext
from market_intelligence.context.session_context import SessionContext
from market_intelligence.context.instrument_context import InstrumentContext

from rich.console import Console, Group
console = Console(record=True)


class LiveUpdateRunner:
    """
    Executes timeframe-based live market updates.
    """

    def __init__(
        self,
        *,
        instrument_ctx: InstrumentContext,
        session_ctx: SessionContext,
        candle_snapshot,
        indicator_snapshot,
        sr_snapshot,
        llm_client,
        conversation_manager,
        cost_guard,
        telegram_notifier,
        timeframe: str,
        timezone: str,
        sleep_sec: int = 10,
    ):
        self.instrument_ctx = instrument_ctx
        self.session_ctx = session_ctx
        self.candle_snapshot = candle_snapshot
        self.indicator_snapshot = indicator_snapshot
        self.sr_snapshot = sr_snapshot
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.cost_guard = cost_guard
        self.telegram = telegram_notifier
        self.timeframe = timeframe
        self.timezone = timezone
        self.sleep_sec = sleep_sec

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds.

        Supported formats:
        - "minute"
        - "5minute"
        - "15minute"
        - "60minute"

        Returns
        -------
        int
            Timeframe duration in seconds.

        Raises
        ------
        ValueError
            If timeframe is unsupported.
        """
        timeframe_map = {
            "minute": 60,
            "1minute": 60,
            "5minute": 5 * 60,
            "15minute": 15 * 60,
            "60minute": 60 * 60,
            "1hour": 60 * 60,
            "hour": 60 * 60,
        }

        tf = timeframe.lower().strip()

        if tf not in timeframe_map:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported values: {list(timeframe_map.keys())}"
            )

        return timeframe_map[tf]

    def run(self) -> None:
        time_ctx = TimeContext(self.timezone)
        session_state = self.session_ctx.session_state(time_ctx.get_now())
        last_processed_time = None
        market_open_time = self.session_ctx.market_open
        market_close_time = self.session_ctx.market_close
        break_loop = False

        # start_date is previous day, market open time. Fetch it from market_session
        days=30
        prev_date = time_ctx.get_now().date() - datetime.timedelta(days=days)
        start_date = datetime.datetime.combine(prev_date, market_open_time)
        #end_date=time_ctx.now + datetime.timedelta(minutes=1)
        end_date = datetime.datetime.now() + datetime.timedelta(minutes=1)

        # Convert self.timeframe to seconds for comparison
        timeframe_seconds = self._timeframe_to_seconds(self.timeframe)

        # Only consider the below columns for the prompt to save tokens
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
        #print(f"Fetching market data from {start_date} to {end_date}...")
        candles = self.candle_snapshot.fetch(
            instrument_token=self.instrument_ctx.token,
            intervals=[self.timeframe],
            start_date=start_date,
            end_date=end_date,
        )
        df = candles.get(self.timeframe)
        
        while True:
            latest_candle = df.iloc[-1]
            candle_time = latest_candle["date"]

            # Ensure candle close only once
            print("-" * 50)
            print("Last processed time:", last_processed_time)
            print("Latest candle time:", candle_time)
            print("Current time:", time_ctx.get_now())
            print("-" * 50)

            if last_processed_time:
                # Wait till current time > last candle time + timeframe duration to ensure we are processing only after the candle is closed
                wait_time = time_ctx.get_now() - candle_time
                if last_processed_time + datetime.timedelta(seconds=timeframe_seconds) < time_ctx.get_now():
                    print("Data not yet processed, for latest candle")
                elif wait_time.total_seconds() < timeframe_seconds:
                    wait_time = timeframe_seconds - wait_time.total_seconds()
                    next_ops_time = candle_time + datetime.timedelta(seconds=timeframe_seconds)
                    console.print(f"[bold yellow]Waiting until next operations time: {next_ops_time.strftime('%H:%M')}...[/bold yellow]")
                    while True:
                        now = dt.now().time()
                        if now >= next_ops_time.time():
                            console.print(f"[bold green]Proceeding with operations.[/bold green]")
                            break
                        else:
                            time_to_ops = dt.combine(dt.today(), next_ops_time.time()) - dt.combine(dt.today(), now)
                            minutes, seconds = divmod(time_to_ops.seconds, 60)
                            console.print(f"[yellow]Time until next operations time: {minutes} minutes and {seconds} seconds.[/yellow]", end='\r')
                            time.sleep(2)  # Sleep for 30 seconds before checking again

            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    #print(f"Fetching market data from {start_date} to {end_date}...")
                    prev_date = time_ctx.get_now().date() - datetime.timedelta(days=days)
                    market_open_time = self.session_ctx.market_open
                    start_date = datetime.datetime.combine(prev_date, market_open_time)
                    #end_date=time_ctx.now + datetime.timedelta(minutes=1)
                    end_date = datetime.datetime.now() + datetime.timedelta(minutes=1)
                    candles = self.candle_snapshot.fetch(
                        instrument_token=self.instrument_ctx.token,
                        intervals=[self.timeframe],
                        start_date=start_date,
                        end_date=end_date,
                    )
                    break  # Break out of retry loop if fetch is successful
                except Exception as e:
                    retry_count += 1
                    print(f"Error fetching candles: {e}")
                    time.sleep(10)
                    continue

            df = candles.get(self.timeframe)
            if df is None or df.empty:
                time.sleep(self.sleep_sec)
                continue
            
            df = self.indicator_snapshot.capture(candles[self.timeframe])
            
            df = df[cols_to_keep]

            # If market is closed, break the loop
            if time_ctx.get_now().time() > market_close_time:
                latest_candle = df.iloc[-1]
                break_loop = True
            else:
                latest_candle = df.iloc[-2]

            candle_time = latest_candle["date"]

            # last_processed_time is current date and time
            last_processed_time = time_ctx.get_now()

            #indicators = self.indicator_snapshot.capture(df)
            current_price = latest_candle["close"]

            # -----------------------
            # 1. SR zones capture
            # -----------------------
            sr_zones = self.sr_snapshot.get_above_and_below_zones(current_price)
            remark = f"Market is currently {session_state}."
            if break_loop:
                next_trading_day = candle_time.date() + datetime.timedelta(days=1)
                remark += f"\nMarket will be open on {next_trading_day.strftime('%Y-%m-%d')} at {market_open_time.strftime('%H:%M')}."

            prompt = UpdatePromptBuilder().build(
                instrument_name=self.instrument_ctx.symbol,
                time_context={
                "now": time_ctx.get_now(),
                "session_state": session_state,
                "remark": remark,
                },
                timeframe=self.timeframe,
                candle_info={
                    "open": latest_candle["open"],
                    "high": latest_candle["high"],
                    "low": latest_candle["low"],
                    "close": latest_candle["close"],
                },
                indicator_snapshot=latest_candle,
                current_price=current_price,
                sr_zones=sr_zones,
            )

            # Save prompt to file for debugging
            with open("debug_market_prompt.txt", "w") as f:
                f.write(prompt)

            # Telegram update
            self.telegram.send(f"[Live Update - {self.timeframe}]\n", notify=True)
            self.telegram.send(prompt)

            # Optional LLM
            if self.llm_client and self.cost_guard.is_allowed(prompt):
                response = self.llm_client.generate(
                    prompt=prompt,
                    history=self.conversation_manager.get_history(),
                )

                self.conversation_manager.add_user(prompt)
                self.conversation_manager.add_assistant(response)
                self.cost_guard.record_usage(prompt, response)

                self.telegram.send(response)

            # Break the loop if market is closed
            if break_loop:
                console.print(f"[bold red]Market is closed. Ending live updates.[/bold red]")
                break
            
