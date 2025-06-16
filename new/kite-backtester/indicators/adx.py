from .base import Indicator
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator as adxi
import matplotlib.pyplot as plt

class ADXIndicator(Indicator):
    """
    Computes the Average Directional Index (ADX).
    """
    def __init__(self, period=14):
        """
        :param data: DataFrame with 'high', 'low', and 'close' prices.
        :param period: Period for ADX calculation.
        """
        super().__init__()
        self.period = period

    def compute(self, df):
        """
        Compute ADX, along with +DI and -DI lines.
        :return: DataFrame with ADX, +DI, and -DI columns.
        """
        """ df['TR'] = df[['high', 'low', 'close']].apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close']),
                abs(x['low'] - x['close'])), axis=1)
        df['+DM'] = df['high'].diff()
        df['-DM'] = df['low'].diff()

        df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0.0)
        df['-DM'] = df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0.0)

        tr_smooth = df['TR'].rolling(window=self.period).mean()
        plus_di = 100 * (df['+DM'].rolling(window=self.period).mean() / tr_smooth)
        minus_di = 100 * (df['-DM'].rolling(window=self.period).mean() / tr_smooth)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df[f'ADX_{self.period}'] = dx.rolling(window=self.period).mean() """

        adx = adxi(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False)
        df[f"ADX_{self.period}"] = adx.adx()
        df[f"+DI_{self.period}"] = adx.adx_pos()
        df[f"-DI_{self.period}"] = adx.adx_neg()
        print("\nPlease make sure additional 30% of data is available for ADX calculation!!!")

        #self.plot_adx(df, self.period)
        return df
    
    def plot_adx(self, df, period):
        adx_col = f"ADX_{period}"
        plus_di_col = f"+DI_{period}"
        minus_di_col = f"-DI_{period}"

        plt.figure(figsize=(14, 6))
        
        #x_vals = df.index  # Use datetime index for x-axis if available
        x_vals = range(len(df))  # simple index-based x-axis
        # Plot ADX, +DI, -DI
        plt.plot(x_vals, df[adx_col], label='ADX', color='blue', linewidth=1.5)
        plt.plot(x_vals, df[plus_di_col], label='+DI', color='green', linewidth=1.5)
        plt.plot(x_vals, df[minus_di_col], label='-DI', color='red', linewidth=1.5)

        # Optional horizontal reference line
        plt.axhline(25, color='gray', linestyle='--', label='Threshold (25)')

        plt.title(f"ADX and Directional Indicators (Period: {period})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()