from kiteconnect import KiteConnect
import pandas as pd
import warnings
import datetime
from indicators.volume import Volume
import os
from general_lib.convert_timeframe import TimeframeConverter
# Rich library imports for enhanced terminal output
from rich.console import Console, Group
from html2image import Html2Image
import requests
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
import sys
warnings.filterwarnings("ignore", category=FutureWarning)
console = Console(record=True) # For Rich text and table display

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
            login_url = self.kite.login_url()
            console.print(f"Kite Login URL: [link={login_url}]{login_url}[/link]")
            self.kite.set_access_token(access_token)

            console.print("[bold green]Kite API session generated and access token set successfully![/bold green]")
            profile = self.kite.profile() # Verify connection by fetching profile
            console.print(f"[green]Successfully connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")

        except Exception as e:
            print("failed")

            console.print(f"[bold red]Error during Kite API login: {e}[/bold red]")
            login_url = self.kite.login_url()
            console.print(f"Kite Login URL: [link={login_url}]{login_url}[/link]")
            request_token = console.input("[bold cyan]Enter Request Token from the above URL: [/bold cyan]").strip()
            if not request_token:
                console.print("[bold red]No request token entered. Exiting.[/bold red]")
                sys.exit(1) # Critical error, cannot proceed

            data = self.kite.generate_session(request_token, api_secret=api_secret)
            self.kite.set_access_token(data["access_token"])
            #print("[INFO] Session data:", data)
            access_token = data["access_token"]
            print("[INFO] Access Token:", access_token)
            console.print("[bold green]Kite API session generated and access token set successfully![/bold green]")
        
            profile = self.kite.profile() # Verify connection by fetching profile
            console.print(f"[green]Successfully connected for user: {profile.get('user_id')} ({profile.get('user_name')})[/green]")
    
        print("Access token accespted!!!")

        # Set access token
        #self.kite.set_access_token(access_token)
    
    def kite_object(self):
        """
        Returns the KiteConnect object.
        """
        return self.kite
    
    def get_data(self, instrument_token, from_date, to_date, interval="15minute") -> pd.DataFrame:
        """
        Fetch historical OHLC data from Kite API.
        :param instrument_token: Instrument token of the security.
        :param from_date: Start date (datetime).
        :param to_date: End date (datetime).
        :param interval: Candle interval (e.g., 5minute, 15minute).
        :return: DataFrame with historical OHLC data.
        """
        available_interval=["60minute", "30minute", "15minute", "10minute","5minute", "3minute", "minute"]
        req_intv = interval
        if interval not in available_interval:
            req_intv, interval = interval, "15minute"
        data_dict = {}
        data_limit_dict = {"1D": 200, "60minute": 400, "30minute": 200, "15minute": 200, "5minute": 100, "3minute": 50, "minute": 25}
        current_start = from_date
        all_df = []
        while current_start < to_date:
            current_end = min(current_start + datetime.timedelta(days=data_limit_dict[interval]), to_date)
            
            #print(f"Fetching {interval} data for {instrument_token} from {current_start.date()} to {current_end.date()}")
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=current_start,
                to_date=current_end,
                interval=interval,
                continuous=False
            )
            df = pd.DataFrame(data)
            df["interval"] = req_intv
            #print(df.head())
            #print("*" * 50)
            #print(df.tail())
            all_df.append(df)
            current_start = current_end + datetime.timedelta(days=0)  # Move to next day
            
        combined_df = pd.concat(all_df, ignore_index=True)
        if req_intv not in available_interval:
            #print(f"Converting from {interval} to {req_intv}...")
            converter = TimeframeConverter(combined_df)
            combined_df = converter.convert(req_intv)
            combined_df.reset_index(inplace=True)  # Ensure 'date' column is available after conversion
        
        df = combined_df.copy() # To avoid SettingWithCopyWarning when adding indicators
        #vol_indicator = Volume(df)
        #df = vol_indicator.VolumeZscore(period=14)

        return df
        
    def get_instrument_names(self):
        """
        Load instrument names from instruments.csv
        """
        # Get project root (one level above tests/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "instruments.csv")
        df = pd.read_csv(csv_path)
        token_name_map = dict(zip(df["instrument_token"], df["name"]))
        return token_name_map
    