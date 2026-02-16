import datetime

class PDHPDLExtractor:
    """
    Extracts Previous Day High and Low.
    """

    @staticmethod
    def extract(daily_df, market_close_time=datetime.time(15, 30)):
        if daily_df is None or len(daily_df) < 2:
            return None, None
        
        # Capture last row's date and time to ensure we are looking at the previous day's data
        last_row = daily_df.iloc[-1]
        last_date = last_row["date"].date()
        last_time = last_row["date"].time()

        current_datetime = datetime.datetime.now()
        # If the last row's date is same as current date and current time is before market close, 
        # we should use the second last row for PDH/PDL as the last row might be incomplete
        if last_date == current_datetime.date() and last_time < market_close_time:
            prev = daily_df.iloc[-2]
        else:
            prev = daily_df.iloc[-1]

        return last_date, float(prev["high"]), float(prev["low"])