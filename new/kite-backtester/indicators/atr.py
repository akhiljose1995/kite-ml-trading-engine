from .base import Indicator
import pandas as pd
from ta.volatility import AverageTrueRange
import matplotlib.pyplot as plt

class ATRIndicator(Indicator):
    """
    Computes the Average True Range (ATR) as a volatility indicator.
    """

    def __init__(self, period=14):
        """
        :param period: Lookback period for ATR calculation.
        """
        super().__init__()
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ATR using high, low, and close prices.

        :param df: DataFrame containing 'high', 'low', and 'close'.
        :return: DataFrame with ATR_<period> column.
        """
        atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=self.period, fillna=False)
        df[f"ATR_{self.period}"] = atr.average_true_range()
        df[f"ATR_{self.period}_Percent"] = df[f"ATR_{self.period}"] / df['close']  # normalize

        # Optional plotting
        #self.plot_atr(df, self.period)

        return df

    def plot_atr(self, df: pd.DataFrame, period: int):
        """
        Plot ATR for visual inspection.

        :param df: DataFrame with computed ATR.
        :param period: Lookback period for ATR.
        """
        atr_col = f"ATR_{period}"

        plt.figure(figsize=(14, 5))
        x_vals = range(len(df))  # index-based x-axis
        plt.plot(x_vals, df[atr_col], label=f"ATR ({period})", color="orange")

        plt.title(f"Average True Range (Period: {period})")
        plt.xlabel("Index")
        plt.ylabel("ATR Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
