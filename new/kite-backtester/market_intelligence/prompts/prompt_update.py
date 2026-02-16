from typing import Dict
from market_intelligence.prompts.prompt_formatter import PromptFormatter


class UpdatePromptBuilder:
    """
    Builds incremental prompt updates on each timeframe close.
    """

    def build(
        self,
        *,
        instrument_name: str,
        time_context: Dict,
        timeframe: str,
        candle_info: Dict,
        indicator_snapshot: Dict,
        current_price: float,
        sr_zones: Dict,
    ) -> str:

        pf = PromptFormatter
        sections = []

        sections.append(pf.format_header("Market Update"))
        sections.append(
            f"Instrument: {instrument_name}\n"
            f"Current Date & Time: {time_context['now']}\n"
            f"Timeframe Closed: {timeframe}\n"
            f"Current Price: {current_price}\n"            
            f"Market State: {time_context['session_state']}\n"
            f"Remark: {time_context['remark']}\n"
        )

        sections.append(pf.format_header("Daily Support / Resistance Zones"))
        sections.append(pf.format_zones(sr_zones))

        sections.append(pf.format_header("Reasoning Task"))
        sections.append("Consider the new candle and updated indicators carefully.\n"
                    "Context:\n"
                    "- Previous market state and bias\n"
                    "- Key zones to respect: support resistance zones\n"
                    "Below are the candles that formed AFTER the previous analysis.\n"
                    "Your task:\n"
                    "1) Explain what actually happened vs expectation\n"
                    "2) State whether the market confirmed or rejected prior bias\n"
                    "3) Update the current market state\n"
                    "4) Mention if any BUY or SELL is now valid\n"
                    "   - Intraday or Swing\n"
                    "   - Entry condition\n"
                    "   - Stop-loss\n"
                    "   - Exit / target\n"
                    "5) If no trade exists, clearly say what to WAIT for next\n"
                    "Rules:\n"
                    "- No hindsight judgment\n"
                    "- No prediction\n"
                    "- Update bias only if structure changes\n"
                    "- Respect key zones over indicators\n"
                    "Analyze the following new candles\n"
        )

        sections.append(pf.format_header("New Candle and Indicator Values"))
        sections.append(pf.format_kv(indicator_snapshot))

        return "\n".join(sections)