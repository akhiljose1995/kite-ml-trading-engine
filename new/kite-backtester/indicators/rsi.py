import pandas as pd
from .base import Indicator
import numpy as np
import matplotlib.pyplot as plt

class RSIIndicator(Indicator):
    """
    Implements the Relative Strength Index (RSI) indicator.
    The RSI is a momentum oscillator that measures the speed and change of price movements.
    It ranges from 0 to 100 and is typically used to identify overbought or oversold conditions.
    """
    def __init__(self, period=14, column='close'):
        """
        :param period: Period for the RSI calculation.
        """
        super().__init__(column)
        self.period = period

    def compute(self, df):
        """
        Calculate the RSI for the given DataFrame.
        :param df: DataFrame with price data.
        :return: DataFrame with RSI column added.
        """        
        delta = df[self.column].diff()

        
        """ gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        df['gain'] = gain
        df['loss'] = loss
        print(df[['delta','gain', 'loss']].head(20))
        
        
        # Wilder's smoothing (equivalent to RMA)
        avg_gain = gain.ewm(alpha=1/self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1/self.period, min_periods=self.period).mean()
        df['avg_gain'] = avg_gain
        df['avg_loss'] = avg_loss
        rs = avg_gain / avg_loss
        df['rs'] = rs
        df[f'RSI_{self.period}'] = 100 - (100 / (1 + rs))
        print(df[['gain', 'loss', 'avg_gain', 'avg_loss', 'rs', f'RSI_{self.period}']].head(20)) """
       
        diff_price_values = delta.values
        p_diff = 0
        n_diff = 0
        curr_avg_positive = 0
        curr_avg_negative = 0
        price_index = 0
        rsi = []

        for diff in diff_price_values:
            if diff > 0:
                p_diff = diff
                n_diff = 0
            elif diff < 0:
                n_diff = -diff
                p_diff = 0
            else:
                p_diff = 0
                n_diff = 0

            if price_index < self.period:
                curr_avg_positive += ((1 / self.period) * p_diff)
                curr_avg_negative += ((1 / self.period) * n_diff)
                rsi.append(None)  # Not enough data to calculate RSI yet
                if price_index == self.period - 1:
                    if curr_avg_negative != 0:
                        rsi[-1] = 100 - (100 / (1 + (curr_avg_positive / curr_avg_negative)))
                    else:
                        rsi[-1] = 100
            else:
                curr_avg_positive = ((curr_avg_positive * (self.period - 1)) + p_diff) / self.period
                curr_avg_negative = ((curr_avg_negative * (self.period - 1)) + n_diff) / self.period
                if curr_avg_negative != 0:
                    rsi.append(100 - (100 / (1 + (curr_avg_positive / curr_avg_negative))))
                else:
                    rsi.append(100)

            price_index += 1

        df[f'RSI_{self.period}'] = round(pd.Series(rsi), 2)
        
        """ # RSI plot
        plt.subplot(2, 1, 2)
        x_vals = range(len(df))  # simple index-based x-axis
        plt.plot(x_vals, df[f'RSI_{self.period}'], label=f'RSI {self.period}', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()

        plt.tight_layout()
        plt.show() """

        return df
