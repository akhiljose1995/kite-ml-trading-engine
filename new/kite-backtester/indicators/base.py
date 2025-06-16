# indicators/base.py
from abc import ABC, abstractmethod
import pandas as pd

class Indicator(ABC):
    """
    Base class for all technical indicators.
    Each indicator must implement the compute() method.
    """
    def __init__(self, column='close'):
        """
        Initialize with price data.
        
        :param data: A pandas DataFrame with at least a 'close' column.
        """
        self.column = column

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the indicator. This method should be overridden in derived classes.
        """
        pass