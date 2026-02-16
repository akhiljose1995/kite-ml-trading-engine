from typing import Dict, List
import re
from market_intelligence.prompts.prompt_formatter import PromptFormatter


class StartPromptBuilder:
    """
    Builds the startup (pre-market) prompt.
    """

    def build(
        self,
        *,
        instrument_name: str,
        time_context: Dict,
        indicators_15m: Dict,
        indicators_1h: Dict,
        indicators_1d: Dict,
        prev_day,
        pdh: float,
        pdl: float,
        current_price: float,
        sr_zones: Dict,
        news: List[Dict],
    ) -> str:

        pf = PromptFormatter
        sections = []

        sections.append(pf.format_header("Market Context"))
        sections.append(
            f"Instrument: {instrument_name}\n"
            f"Current Date & Time: {time_context['now']}\n"
            f"Current Price: {current_price}\n"
            f"Session: {time_context['session_state']}\n"
        )

        sections.append(pf.format_header("Previous Day Levels"))
        sections.append(f"Previous Day: {prev_day} - PDH: {pdh}\n- PDL: {pdl}\n")

        sections.append(pf.format_header("Daily Support / Resistance Zones"))
        sections.append(pf.format_zones(sr_zones))

        sections.append(pf.format_header("Relevant News"))
        sections.append(pf.format_news(news))

        sections.append(pf.format_header("Reasoning Task"))
        sections.append("You are a rule-based trading analyst.\n"
                        "Consider all the above data and indicators carefully.\n"
                        "Check previous day, indicator date, session state and current date and time.\n"
                        "Your response MUST include:\n"
                        "1) Market state & setup\n"
                        "   - Trend / Range / Breakout / Failed breakout / Transition\n"
                        "   - Current bias: Bullish / Bearish / Neutral\n"
                        "2) Scenarios (2 to 3 only)\n"
                        "   - What must happen for each scenario to be valid and it's probability\n"
                        "3) Trades (if any, else mention 'no trade, wait')\n"
                        "   - BUY / SELL\n"
                        "   - Intraday or Swing\n"
                        "   - Entry condition\n"
                        "   - Stop-loss logic\n"
                        "   - Exit / target logic\n"
                        "4) WAIT conditions\n"
                        "   - Clearly state what to wait for if no trade exists\n"
                        "Strict Rules to follow:\n"
                        "- Give the response in shortest possible version, use bullet points, avoid long sentences\n"
                        "- No prediction\n"
                        "- No forced trades\n"
                        "- Respect higher timeframe first\n"
                        "- If no trade, say 'No trade, wait'\n"
        )

        sections.append(pf.format_header("Indicators 15min"))
        sections.append(pf.format_df(indicators_15m))

        sections.append(pf.format_header("Indicators 1H"))
        sections.append(pf.format_df(indicators_1h))

        sections.append(pf.format_header("Indicators 1D"))
        sections.append(pf.format_df(indicators_1d))

        final_prompt = "".join(sections)
        
        return final_prompt