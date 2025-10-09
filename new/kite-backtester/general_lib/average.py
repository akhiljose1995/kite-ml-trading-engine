import numpy as np
import pandas as pd

class AverageCalculator:
    """
    A general-purpose average calculation utility.
    Supports various average methods like stepped moving average.
    """

    def __init__(self, data):
        """
        :param data: list, numpy array, or pandas Series
        """
        self.data = pd.Series(data) if not isinstance(data, pd.Series) else data

    def stepped_moving_average(self, n=0, step=1, m=3):
        """
        Calculate stepped moving average for a specific index n.
        Example: step=2, m=3 => values at n+1, n+3, n+5
        :param n: starting index
        :param step: gap between values
        :param m: number of values to average
        :return: float or None
        """
        indices = [n + 1 + step * i for i in range(m)]
        values = [self.data.iloc[i] for i in indices if i < len(self.data)]
        return np.mean(values) if values else None

    def stepped_moving_average_all(self, step=2, m=3):
        """
        Calculate stepped moving average for all indices in the series.
        :param step: gap between values
        :param m: number of values to average
        :return: Pandas Series
        """
        return pd.Series([self.stepped_moving_average(n, step, m) for n in range(len(self.data))],
                         index=self.data.index)