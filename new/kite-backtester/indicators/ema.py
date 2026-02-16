

class EMAIndicator:
    """
    Computes the Exponential Moving Average (EMA).
    """
    def __init__(self, period=20, column='close'):
        """
        :param data: DataFrame with price data.
        :param period: Period for EMA.
        """
        self.column = column
        self.period = period

    def ema_calc(self, df, period=None):
        """
        Compute the EMA and add it as a new column.
        :return: DataFrame with EMA column added.
        """
        if period is not None:
            self.period = period
        df[f'EMA_{self.period}'] = round(df[self.column].ewm(span=self.period, adjust=False).mean(), 2)
        return df
