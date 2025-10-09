import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base import Indicator

class Volume:
    """
    Volume-based indicator class for analyzing volume behavior.
    This class is specifically designed to operate on the 'volume' column
    and generate relevant volume indicators such as z-score normalization.
    """

    def __init__(self, df):
        """
        Initializes the Volume indicator with a DataFrame.

        :param df: Input DataFrame that must contain a 'volume' column.
        """
        self.df = df

    def VolumeZscore(self, period=14):
        """
        Computes the z-score of the volume column using a rolling window.

        :param period: Rolling window size to calculate mean and standard deviation.
        :return: DataFrame with an added 'vol_zscore' column.
        """
        rolling_mean = self.df['volume'].rolling(window=period).mean()
        rolling_std = self.df['volume'].rolling(window=period).std()
        self.df['vol_zscore'] = (self.df['volume'] - rolling_mean) / rolling_std
        return self.df