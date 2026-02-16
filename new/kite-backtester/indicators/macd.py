from .base import Indicator
import matplotlib.pyplot as plt

class MACDIndicator(Indicator):
    """
    Computes the Moving Average Convergence Divergence (MACD).
    """
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, column='close'):
        """
        :param data: DataFrame with price data.
        :param fast: Fast EMA period.
        :param slow: Slow EMA period.
        :param signal: Signal line EMA period.
        """
        super().__init__(column)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute(self, df):
        """
        Compute MACD, Signal Line and MACD Histogram.
        :return: DataFrame with MACD-related columns added.
        """
        ema_fast = df[self.column].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df[self.column].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = round(ema_fast - ema_slow, 2)
        df['MACD_Signal'] = round(df['MACD'].ewm(span=self.signal_period, adjust=False).mean(), 2)
        df['MACD_Hist'] = round(df['MACD'] - df['MACD_Signal'], 2)
        #self.plot_macd(df)
        return df

    def plot_macd(self, df):
        """
        Plot MACD, Signal Line, and Histogram from a DataFrame.
        Assumes df contains 'MACD' and 'MACD_Signal' columns.
        """
        # Optional Histogram
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        plt.figure(figsize=(14, 6))
        x_vals = range(len(df))  # simple index-based x-axis

        # Plot MACD and Signal
        plt.plot(x_vals, df['MACD'], label='MACD', color='blue', linewidth=1.5)
        plt.plot(x_vals, df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.5)

        # Histogram (bars)
        plt.bar(x_vals, df['MACD_Hist'], label='MACD Histogram', color='grey', alpha=0.5)

        plt.title('MACD Indicator')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()