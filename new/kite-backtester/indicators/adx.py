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
        adx = adxi(high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False)
        df[f"ADX_{self.period}"] = round(adx.adx(), 2)
        df[f"PDI_{self.period}"] = round(adx.adx_pos(), 2)
        df[f"NDI_{self.period}"] = round(adx.adx_neg(), 2)

        # Add ADX Strength Label
        def adx_strength(val):
            if pd.isna(val):
                return None
            elif val >= 40:
                return "very_strong"
            elif val >= 25:
                return "strong"
            elif val >= 20:
                return "moderate"
            else:
                return "weak"

        df[f"ADX_{self.period}_Strength_Label"] = df[f"ADX_{self.period}"].apply(adx_strength)

        # Print a warning if there are not enough data points for ADX calculation
        #print("\nPlease make sure additional 30% of data is available for ADX calculation!!!")

        #self.plot_adx(df, self.period)
        return df
    
    def plot_adx(self, df, period):
        adx_col = f"ADX_{period}"
        plus_di_col = f"PDI_{period}"
        minus_di_col = f"NDI_{period}"

        plt.figure(figsize=(14, 6))
        
        #x_vals = df.index  # Use datetime index for x-axis if available
        x_vals = range(len(df))  # simple index-based x-axis
        # Plot ADX, +DI, -DI
        plt.plot(x_vals, df[adx_col], label='ADX', color='blue', linewidth=1.5)
        plt.plot(x_vals, df[plus_di_col], label='PDI', color='green', linewidth=1.5)
        plt.plot(x_vals, df[minus_di_col], label='NDI', color='red', linewidth=1.5)

        # Optional horizontal reference line
        plt.axhline(25, color='gray', linestyle='--', label='Threshold (25)')

        plt.title(f"ADX and Directional Indicators (Period: {period})")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()