from .base import Indicator

class EMAIndicator(Indicator):
    """
    Computes the Exponential Moving Average (EMA).
    """
    def __init__(self, period=20, column='close'):
        """
        :param data: DataFrame with price data.
        :param period: Period for EMA.
        """
        super().__init__(column)
        self.period = period

    def compute(self, df):
        """
        Compute the EMA and add it as a new column.
        :return: DataFrame with EMA column added.
        """
        df[f'EMA_{self.period}'] = df[self.column].ewm(span=self.period, adjust=False).mean()
        return df
