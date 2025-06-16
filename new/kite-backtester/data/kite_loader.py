from kiteconnect import KiteConnect
import pandas as pd

class KiteDataLoader:
    """
    Wrapper for Kite Connect historical data API.
    Fetches and returns price data as pandas DataFrame.
    """

    def __init__(self, api_key, access_token, request_token, api_secret):
        """
        :param api_key: Zerodha Kite API key.
        :param api_secret: Zerodha Kite API secret.
        :param request_token: Request token from the browser login redirect.
        """
        self.kite = KiteConnect(api_key=api_key)
        print(self.kite.login_url())

        try:
            self.kite.set_access_token(access_token)
            self.kite.profile()

        except Exception:
            print("failed")
            # Exchange request token for access token
            session_data = self.kite.generate_session(request_token, api_secret=api_secret)
            print("[INFO] Session data:", session_data)
            access_token = session_data["access_token"]
            print("[INFO] Access Token:", access_token)
            print("Please update the config.py with new Access Token")
            #self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            self.kite.profile()

        print("Access token accespted!!!")

        # Set access token
        #self.kite.set_access_token(access_token)

    def get_data(self, instrument_token, from_date, to_date, interval="15minute") -> pd.DataFrame:
        """
        Fetch historical OHLC data from Kite API.
        :param instrument_token: Instrument token of the security.
        :param from_date: Start date (datetime).
        :param to_date: End date (datetime).
        :param interval: Candle interval (e.g., 5minute, 15minute).
        :return: DataFrame with historical OHLC data.
        """
        data = self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False
        )
        df = pd.DataFrame(data)
        df["interval"] = interval
        #df.set_index("date", inplace=True)
        return df

