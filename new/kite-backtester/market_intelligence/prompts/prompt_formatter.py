from typing import Dict, List, Optional
from datetime import datetime


class PromptFormatter:
    """
    Helper utilities to format market data into prompt-ready text.
    """

    @staticmethod
    def format_header(title: str) -> str:
        return f"=== {title.upper()} ===\n"

    @staticmethod
    def format_kv(data: Dict, precision: int = 2) -> str:
        lines = []
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, float):
                v = round(v, precision)
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def format_list(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)
    
    # A function to format df. Use "\n".join for each row
    # First line is column names, then each line is values of the row
    @staticmethod
    def format_df(df) -> str:
        if df is None or df.empty:
            return "Empty DataFrame"

        lines = []
        # First line is column names
        lines.append(", ".join(df.columns))
        # Then each line is values of the row
        for _, row in df.iterrows():
            lines.append(", ".join([str(v) for v in row.values]))
        return "\n".join(lines)+"\n"

    @staticmethod
    def format_news(news: List[Dict], max_items: int = 5) -> str:
        if not news:
            return "No major relevant news.\n"

        lines = []
        for item in news[:max_items]:
            ts = item.get("published_at")
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "N/A"

            lines.append(
                f"- [{ts_str}] {item.get('title')} ({item.get('source')})"
            )

        return "\n".join(lines)+"\n"

    @staticmethod
    def format_zones(zones: Dict[str, List[Dict]], max_levels: int = 3) -> str:
        lines = []

        for side in ("above_price", "below_price"):
            side_zones = zones.get(side, [])[:max_levels]
            if not side_zones:
                continue

            lines.append(f"{side.replace('_', ' ').title()}:")
            for z in side_zones:
                price = z.get("zone_center")
                strength = z.get("strength_label")
                lines.append(f"  - {price} (strength={strength})")

        return "\n".join(lines)+"\n"