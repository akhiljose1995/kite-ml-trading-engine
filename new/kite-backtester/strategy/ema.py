import pandas as pd
import numpy as np

class EMA:
    """
    Exponential Moving Average (EMA) indicator.
    """
    def __init__(self, period=14):
        """
        :param period: Period for EMA calculation.
        """
        self.period = period

    def ema_to_price_distance(self, df):
        """
        Compute the EMA and add it as a new column.
        :return: DataFrame with EMA column added.
        """
        df[f'EMA_{self.period}_dist_to_price'] = round(df['close'] - df[f'EMA_{self.period}'], 2)
        return df
    
    def distance_between_emas(self, df, column1, column2):
        """
        Compute the distance between two EMA columns.
        :param df: DataFrame with EMA columns.
        :param column1: First EMA column name.
        :param column2: Second EMA column name.
        :return: DataFrame with distance column added.
        """
        df[f'Distance_{column1}_{column2}'] = round(df[column1] - df[column2], 2)
        return df