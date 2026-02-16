import pandas as pd
from typing import Dict, List


class IndicatorSnapshot:
    """
    Applies a sequence of indicator computations and captures latest values.
    """

    def __init__(self, indicator_pipeline: List[callable]):
        """
        indicator_pipeline:
            List of callables: fn(df) -> df
        """
        self.pipeline = indicator_pipeline

    def capture(self, df: pd.DataFrame, only_last_candle=False) -> Dict[str, float]:
        if df is None or df.empty:
            return {}

        df = df.copy()

        for step in self.pipeline:
            df = step(df)

        if only_last_candle:
            last_row = df.iloc[-1]

            snapshot = {}
            for col, val in last_row.items():
                if isinstance(val, (int, float)):
                    snapshot[col] = float(val)
        
        else:
            return df
        
        return snapshot