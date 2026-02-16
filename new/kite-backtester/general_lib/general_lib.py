import datetime
import pandas as pd

class GeneralLib:
    """
    General library for common functions used across the project.
    """

    @staticmethod
    def get_current_date() -> str:
        """
        Returns the current date in YYYY-MM-DD format.
        """
        return datetime.datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def get_current_time() -> str:
        """
        Returns the current time in HH:MM:SS format.
        """
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    def calculate_quantile(data, q):
        """
        Calculate quantile for a given list using the basic formula.
        :param data: list of numbers
        :param q: quantile between 0 and 1 (e.g., 0.25 for 25th percentile)
        :return: quantile value
        """
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        if len(data) == 0:
            raise ValueError("Data list cannot be empty")

        # Step 1: Sort the data
        sorted_data = sorted(data)

        # Step 2: Find the position
        n = len(sorted_data)
        pos = q * (n - 1)

        # Step 3: Interpolation
        lower_index = int(pos)
        upper_index = min(lower_index + 1, n - 1)
        fraction = pos - lower_index

        # If position is an integer, no interpolation needed
        if fraction == 0:
            return sorted_data[lower_index]
        else:
            return sorted_data[lower_index] + fraction * (sorted_data[upper_index] - sorted_data[lower_index])


    # Example usage
    #data = [8, 3, 7, 5, 1]
    #print("25th percentile:", calculate_quantile(data, 0.25))
    #print("60th percentile:", calculate_quantile(data, 0.6))
