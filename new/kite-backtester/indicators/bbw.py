from .base import Indicator
import pandas as pd
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt


class BBWIndicator(Indicator):
    """
    Computes Bollinger Bandwidth (BBW) as a volatility indicator.
    """

    def __init__(self, period=20, std_dev=2):
        """
        :param period: Window size for Bollinger Bands.
        :param std_dev: Number of standard deviations for the band width.
        """
        super().__init__()
        self.period = period
        self.std_dev = std_dev

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bandwidth.

        :param df: DataFrame containing 'close' price.
        :return: DataFrame with BBW_<period> column.
        """
        bb = BollingerBands(close=df["close"], window=self.period, window_dev=self.std_dev, fillna=False)
        df[f"BBW_{self.period}"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

        # Optional plotting
        #self.plot_bbw(df, self.period)

        return df

    def plot_bbw(self, df: pd.DataFrame, period: int):
        """
        Plot Bollinger Bandwidth.

        :param df: DataFrame with computed BBW.
        :param period: Lookback period.
        """
        bbw_col = f"BBW_{period}"

        plt.figure(figsize=(14, 5))
        x_vals = range(len(df))  # index-based x-axis
        plt.plot(x_vals, df[bbw_col], label=f"BBW ({period})", color="purple")

        plt.title(f"Bollinger Bandwidth (Period: {period})")
        plt.xlabel("Index")
        plt.ylabel("Bandwidth")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
