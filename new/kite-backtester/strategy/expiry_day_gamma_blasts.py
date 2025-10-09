import pandas as pd
import datetime as dt
from kiteconnect import KiteConnect
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from rich.console import Console
from rich.table import Table

class ExpiryDayGammaBlast:
    def __init__(self, kite: KiteConnect, start_date='2019-02-01', interval='5minute', index="nifty50"):
        self.kite = kite
        self.start_date = pd.to_datetime(start_date).date()
        self.end_date = dt.date.today()
        self.interval = interval
        self.index = index.lower()
        if index.lower() == "nifty50":
            self.token = 256265  # NIFTY 50 index token in Kite
        elif index.lower() == "banknifty":
            self.token = 260105  # BANKNIFTY index token in Kite
        elif index.lower() == "sensex":
            self.token = 265  # SENSEX index token
        self.len_expiry_days = 0
        self.len_blasts = 0
        self.percentage_blast = 0 # len(expiry_days)/len(blasts) * 100 if blasts else 0
        self.start_time = "13:30"  # Default start time for gamma blast detection
        self.price_diff_threshold = 100  # Default price difference threshold for gamma blasts
        self.avg_start_time = None  # To store average start time of blasts
        self.avg_end_time = None
        self.avg_time_diff = None  # To store average time difference of blasts

    def monthly_expiry_dates(self, weekly_expiry_dates=None):
        """Returns all monthly expiry dates (last Thursday of each month)."""
        if weekly_expiry_dates is None or weekly_expiry_dates.empty:
            return pd.DatetimeIndex([])

        # Ensure sorted and unique
        weekly_expiry_dates = pd.DatetimeIndex(sorted(set(weekly_expiry_dates)))

        # Group by month/year and pick last date
        monthly_expiries = weekly_expiry_dates.to_series().groupby(
            [weekly_expiry_dates.year, weekly_expiry_dates.month]
        ).max()

        return pd.DatetimeIndex(monthly_expiries)
    
    def get_weekly_expiry_dates(self):
        """Returns all weekly expiry Thursdays since start date."""
        if self.index == "nifty50":
            freq = 'W-THU'
        elif self.index == "banknifty":
            freq = 'W-THU'
        elif self.index == "sensex":
            freq = 'W-TUE'
        expiry_dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq)
        if self.index == "banknifty":
            expiry_dates = self.monthly_expiry_dates(weekly_expiry_dates=expiry_dates)
        print(f"Found {len(expiry_dates)} expiry dates from {self.start_date} to {self.end_date}")
        print("Expiry Dates:", expiry_dates)
        self.len_expiry_days = len(expiry_dates)
        return [d.date() for d in expiry_dates]

    def fetch_intraday_data(self, date):
        """Fetch intraday OHLC data for Nifty on a given date."""
        from_dt = dt.datetime.combine(date, dt.time(9, 15))
        to_dt = dt.datetime.combine(date, dt.time(15, 30))
        #print(f"Fetching data for {date} from {from_dt} to {to_dt}")

        try:
            data = self.kite.historical_data(
                instrument_token=self.token,
                from_date=from_dt,
                to_date=to_dt,
                interval=self.interval
            )
            #print(f"Fetched {len(data)} records for {date}")
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()
            df['datetime'] = pd.to_datetime(df['date'])
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {date}: {e}")
            return pd.DataFrame()

    def detect_gamma_blasts(self, df, start_time="13:30", end_time="15:30", price_diff_threshold=100):
        """
        Detect expiry-day gamma blasts based on post-start_time moves until end_time.
        """
        # Filter for the time range
        df = df.between_time(start_time, end_time)

        if df.empty:
            return pd.DataFrame()

        # Identify extremes
        min_low_idx = df['low'].idxmin()
        max_high_idx = df['high'].idxmax()

        min_low_row = df.loc[min_low_idx]
        max_high_row = df.loc[max_high_idx]

        # Determine sequence of extremes
        if min_low_idx < max_high_idx:
            first_time = min_low_idx.time()
            later_time = max_high_idx.time()
            first_price = min_low_row['low']
            later_price = max_high_row['high']
        else:
            first_time = max_high_idx.time()
            later_time = min_low_idx.time()
            first_price = max_high_row['high']
            later_price = min_low_row['low']

        # Calculate differences
        diff = later_price - first_price
        abs_diff = abs(diff)
        pct_change = abs_diff / first_price * 100

        # Convert single datetime.time objects to seconds
        start_time_seconds = first_time.hour * 3600 + first_time.minute * 60 + first_time.second
        end_time_seconds   = later_time.hour * 3600 + later_time.minute * 60 + later_time.second

        # Difference in time in minutes
        time_diff_minutes = (end_time_seconds - start_time_seconds) / 60

        # Prepare result
        result = pd.DataFrame([{
            "blast_date": min_low_idx.date(),  # date of the move
            "start_time": first_time,
            "end_time": later_time,
            "first_price": first_price,
            "later_price": later_price,
            "Time_diff_minutes": time_diff_minutes,
            "diff": diff,
            "abs_diff": abs_diff,
            "pct_change": pct_change
        }])

        # Apply threshold
        return result[result['abs_diff'] > price_diff_threshold]

    # Plotting and displaying results
    def plot_blast_summary(self, csv_path="nifty_gamma_blasts.csv", figsize=(12, 6)):
        """Plot max gamma blast abs_diff per expiry with average line and summary text."""
        try:
            df = pd.read_csv(csv_path, parse_dates=['actual_date', 'expiry'])
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        # Group by expiry and get max abs_diff
        summary_df = df.groupby('expiry')['abs_diff'].max().reset_index()
        summary_df['expiry'] = pd.to_datetime(summary_df['expiry'])

        # Calculate average
        avg_abs_diff = summary_df['abs_diff'].mean()

        plt.figure(figsize=figsize)
        sns.lineplot(x='expiry', y='abs_diff', data=summary_df, marker='o', label="Index Chg Per Expiry")

        # Draw average line
        plt.axhline(avg_abs_diff, color='red', linestyle='--', label=f"Average ({avg_abs_diff:.2f})")

        # Add summary box inside the plot
        text_str = (
            f"Total expiry days: {self.len_expiry_days}\n"
            f"Total blast days: {self.len_blasts}\n"
            f"Blast Percentage: {round(self.percentage_blast,2)}\n"
            f"Start time taken: {self.start_time}\n"
            f"Threshold price chg: {self.price_diff_threshold}\n"
            f"Average start time: {self.avg_start_time if self.avg_start_time else 'N/A'}\n"
            f"Average end time: {self.avg_end_time if self.avg_end_time else 'N/A'}\n"
            f"Average time diff: {self.avg_time_diff if self.avg_time_diff else 'N/A'}"
        )
        plt.gca().text(
            0.02, 0.95, text_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.5)
        )

        plt.title(f"Blasts per Expiry Day for {self.index.upper()}")
        plt.xlabel("Expiry Date")
        plt.ylabel("Index point blasts")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def display_blast_table(self, csv_path="nifty_gamma_blasts.csv", max_rows=20):
        """Display gamma blast details as a rich table in terminal."""
        console = Console()

        try:
            df = pd.read_csv(csv_path, parse_dates=['actual_date', 'expiry'])
        except Exception as e:
            console.print(f"[red]Error reading CSV: {e}[/red]")
            return

        df = df[['expiry', 'actual_date', 'first_price', 'later_price', 'abs_diff']].copy()
        df.sort_values(by='abs_diff', ascending=False, inplace=True)

        if max_rows:
            df = df.head(max_rows)

        # Create rich table
        table = Table(show_header=True, header_style="bold magenta")
        for col in df.columns:
            table.add_column(col)

        for _, row in df.iterrows():
            table.add_row(
                str(row['expiry'].date()),
                str(row['actual_date'].date()),
                f"{row['first_price']:.2f}",
                f"{row['later_price']:.2f}",
                f"{row['abs_diff']:.2f}"
            )

        console.print(table)

    def average_time(self, df, col_name='start_time'):
        """Calculate the average start time from a DataFrame with start_time column."""

        # Ensure start_time is datetime.time
        if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
            df[col_name] = pd.to_datetime(df[col_name], format="%H:%M:%S").dt.time

        # Convert time to seconds
        seconds = df[col_name].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
        
        # Mean in seconds
        avg_seconds = int(seconds.mean())

        # Convert back to time
        avg_time = (dt.datetime.min + dt.timedelta(seconds=avg_seconds)).time()
        return avg_time

    def process_expiry_days(self,
                        save_path="nifty_gamma_blasts.csv",
                        limit=None,
                        start_time="13:30",
                        end_time="15:30",
                        price_diff_threshold=100):
        """Main logic to iterate through expiry days and collect gamma blasts.

        Parameters
        ----------
        save_path : str
            CSV file path to save results.
        limit : int or None
            If provided, only process the most recent `limit` expiry dates.
        start_time : str
            Time (HH:MM) to start the detection window (default "13:30").
        end_time : str
            Time (HH:MM) to end the detection window (default "15:30").
        price_diff_threshold : float
            Minimum absolute price move (points) to consider a gamma blast (default 100).
        """
        try:
            self.start_time = start_time
            self.price_diff_threshold = price_diff_threshold
            expiry_dates = self.get_weekly_expiry_dates()
            if limit:
                expiry_dates = expiry_dates[-limit:]
            all_blasts = []

            for expiry in expiry_dates:
                data_found = False
                # Try Thursday, fallback to Wednesday, then Tuesday
                for offset in [0, -1, -2]:
                    check_date = expiry + dt.timedelta(days=offset)
                    #print(f"Checking data for: {check_date}")
                    df = self.fetch_intraday_data(check_date)

                    if df is None or df.empty:
                        # no intraday data for this date — try previous fallback
                        continue

                    # We have intraday data; run the updated detector
                    blasts = self.detect_gamma_blasts(
                        df,
                        start_time=start_time,
                        end_time=end_time,
                        price_diff_threshold=price_diff_threshold
                    )

                    # mark that we had data for this expiry week (even if no blast detected)
                    data_found = True

                    if blasts is None or blasts.empty:
                        # No blast for the data found — stop fallback and move next expiry
                        break

                    # Prepare blasts DataFrame for concatenation/saving
                    b = blasts.copy()

                    # Add expiry and actual_date as date-only fields
                    # expiry might be a datetime.date already; ensure it's date
                    b['expiry'] = expiry if isinstance(expiry, dt.date) else pd.to_datetime(expiry).date()
                    b['actual_date'] = check_date if isinstance(check_date, dt.date) else pd.to_datetime(check_date).date()

                    # Ensure blast_date is date-only (detect_gamma_blasts should already do this)
                    if 'blast_date' in b.columns:
                        # if it's Timestamp, convert to date
                        b['blast_date'] = b['blast_date'].apply(lambda x: x.date() if hasattr(x, 'date') else x)

                    # Format times as HH:MM:SS strings for CSV friendliness
                    if 'start_time' in b.columns:
                        b['start_time'] = b['start_time'].apply(
                            lambda x: x.strftime('%H:%M:%S') if hasattr(x, 'strftime') else str(x)
                        )
                    if 'end_time' in b.columns:
                        b['end_time'] = b['end_time'].apply(
                            lambda x: x.strftime('%H:%M:%S') if hasattr(x, 'strftime') else str(x)
                        )

                    # Append and stop trying earlier fallback days for this expiry
                    all_blasts.append(b)
                    break

                if not data_found:
                    print(f"No data found for expiry week of {expiry}")

            if all_blasts:
                final_df = pd.concat(all_blasts, ignore_index=True)

                # Re-order to a sensible column order if available
                preferred_cols = [
                    'expiry', 'actual_date', 'blast_date', 'start_time', 'end_time', 'Time_diff_minutes',
                    'first_price', 'later_price', 'diff', 'abs_diff', 'pct_change'
                ]
                cols = [c for c in preferred_cols if c in final_df.columns] + \
                    [c for c in final_df.columns if c not in preferred_cols]
                final_df = final_df[cols]

                final_df.to_csv(save_path, index=False)
                print(f"Saved {len(final_df)} blast record(s) to {save_path}")

                # Print average of abs_diff
                avg_abs_diff = final_df['abs_diff'].mean()
                print(f"Average absolute price move across all blasts: {avg_abs_diff:.2f} points")

                # Print minimum and maximum abs_diff
                min_abs_diff = final_df['abs_diff'].min()
                max_abs_diff = final_df['abs_diff'].max()
                print(f"Minimum absolute price move: {min_abs_diff:.2f} points")
                print(f"Maximum absolute price move: {max_abs_diff:.2f} points")

                # Calculate percentage of blasts detected
                self.len_blasts = len(final_df)
                self.percentage_blast = len(final_df) / len(expiry_dates) * 100 if expiry_dates else 0
                print(f"Percentage of expiry days with gamma blasts: {self.percentage_blast:.2f}%")

                # Calculate average start time
                self.avg_start_time = self.average_time(final_df, col_name='start_time')
                print(f"Average start time for blasts: {self.avg_start_time}")

                # Calculate average end time
                self.avg_end_time = self.average_time(final_df, col_name='end_time')
                print(f"Average end time for blasts: {self.avg_end_time}")

                # Calculate average time difference in minutes
                self.avg_time_diff = round(final_df['Time_diff_minutes'].mean(), 2)
                print(f"Average time difference in minutes: {self.avg_time_diff:.2f}")             

                return final_df
            else:
                print("No gamma blast data detected.")
                return None

        except Exception as e:
            print(f"Error processing expiry days: {e}")
            return None


