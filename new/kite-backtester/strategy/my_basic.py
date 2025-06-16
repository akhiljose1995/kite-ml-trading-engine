"""
This strategy module evaluates a stock's potential entry signal using RSI Divergence,
RSI trend direction, MACD Divergence, and MACD Cross checks.

It expects a DataFrame with the following columns:
- 'low', 'high', 'open', 'close', 'volume'
- 'RSI_{period}', 'RSI_Div_Type'
- 'MACD', 'MACD_Signal', 'MACD_Hist', 'MACD_Div_Type'
- 'ADX_{period}', '+DI_{period}', '-DI_{period}'

The output is a dictionary of stock trading signals with suggested entries.
"""

import pandas as pd
import re
from .adx_signal import ADXSignalDetector

class BasicTradingStrategy:
    def __init__(self, df: pd.DataFrame, stock_name: str, period: int = 14):
        self.df = df.copy()
        self.stock = stock_name
        self.period = period
        self.rsi_periods = self._extract_periods('RSI')
        self.adx_periods = self._extract_periods('ADX')
        self.overbought = 70
        self.oversold = 30
        self.signals = {
            "Stock": self.stock
        }
        self.last_macd_div_date = None

    def _extract_periods(self, indicator_prefix):
        pattern = re.compile(f"^{indicator_prefix}_(\\d+)$")
        periods = []

        for col in self.df.columns:
            match = pattern.match(col)
            if match:
                periods.append(int(match.group(1)))

        return sorted(periods)
    
    def evaluate_rsi_divergence(self):
        for period in self.rsi_periods:
            col = f"RSI_{period}_Div_Type"
            div_col = f"RSI_{period}_DIV"
            self.signals[div_col] = "No Entry"
            recent_df = self.df.tail(30)
            last_div = recent_df[col].dropna().iloc[-1] if not recent_df[col].dropna().empty else None

            if last_div == "Bullish":
                self.signals[div_col] = "Buy"
            elif last_div == "Bearish":
                self.signals[div_col] = "Sell"

    def evaluate_rsi_direction(self):
        for period in self.rsi_periods:
            col = f"RSI_{period}"
            self.signals[col] = "No Entry"

            rsi_data = self.df[col].dropna().tail(5)
            if len(rsi_data) < 2:
                return

            rsi_values = rsi_data.values
            rsi_dir = "upward" if rsi_values[-1] > rsi_values[0] else "downward"
            last_rsi = rsi_values[-1]

            if last_rsi > self.overbought:
                self.signals[col] = "Sell" if rsi_dir == "downward" else "No Entry"
            elif last_rsi < self.oversold:
                self.signals[col] = "Buy" if rsi_dir == "upward" else "No Entry"
            else:
                self.signals[col] = "Buy" if rsi_dir == "upward" else "Sell"

    def evaluate_macd_divergence(self):
        col = "MACD_Div_Type"
        div_col = "MACD_DIV"
        self.signals[div_col] = "No Entry"

        recent_df = self.df.tail(50)
        # Drop rows where the column is NaN
        non_null_divs = recent_df.dropna(subset=[col])

        # Get last divergence type and its date
        if not non_null_divs.empty:
            last_row = non_null_divs.iloc[-1]
            last_div = last_row[col]
            self.last_macd_div_date = last_row["date"]
        else:
            last_div = None
            self.last_macd_div_date = None



        if last_div == "Bullish":
            self.signals[div_col] = "Buy"
        elif last_div == "Bearish":
            self.signals[div_col] = "Sell"

    def evaluate_macd_crossover(self):
        macd_col, signal_col = "MACD", "MACD_Signal"
        macd_cross_col = "MACD_Cross"
        self.signals[macd_cross_col] = "No Entry"

        period = 5
        macd = self.df[macd_col].dropna().tail(period)
        signal = self.df[signal_col].dropna().tail(period)
        # Lets print columns of self.df for debugging
        #print("Available columns in DataFrame:", self.df.columns.tolist())
        start_date = self.df["date"].dropna().tail(period).iloc[0] if not self.df["date"].dropna().empty else None
        end_date = self.df["date"].dropna().tail(period).iloc[-1] if not self.df["date"].dropna().empty else None
        
        if len(macd) < 2 or len(signal) < 2:
            return

        macd_mean = macd.mean()
        signal_mean = signal.mean()

        macd_div = self.signals.get("MACD_DIV", "No Entry")

        # Converging or diverging?
        diff_now = abs(macd.iloc[-1] - signal.iloc[-1])
        diff_prev = abs(macd.iloc[-2] - signal.iloc[-2])
        converging = diff_now < diff_prev

        print("MACD Divergence:", macd_div)
        print("MACD Converging:", converging)
        print("MACD Mean:", macd_mean, "Signal Mean:", signal_mean)
        print("MACD Last Value:", macd.iloc[-1], "Signal Last Value:", signal.iloc[-1])
        print("MACD Start Date:", start_date, "End Date:", end_date)
        print("MACD Divergence Date:", self.last_macd_div_date)

        if self.last_macd_div_date and start_date <= self.last_macd_div_date <= end_date:
            print("Date of last MACD divergence is within the current period.")
            if macd.iloc[-1] < 0 and signal.iloc[-1] < 0 and macd_mean < signal_mean and macd_div == "Buy":
                print("MACD Cross Buy Signal within divergence period")
                self.signals[macd_cross_col] = "Buy" if converging else "No Entry"
            elif macd.iloc[-1] > 0 and signal.iloc[-1] > 0 and macd_mean > signal_mean and macd_div == "Sell":
                print("MACD Cross Sell Signal within divergence period")
                self.signals[macd_cross_col] = "Sell" if converging else "No Entry"
        else:
            if macd.iloc[-1] > 0 and signal.iloc[-1] > 0 and macd_mean > signal_mean:
                print("MACD Cross Buy Signal")
                self.signals[macd_cross_col] = "Buy" if not converging else "No Entry"
            elif macd.iloc[-1] < 0 and signal.iloc[-1] < 0 and macd_mean < signal_mean:
                print("MACD Cross Sell Signal")
                self.signals[macd_cross_col] = "Sell" if not converging else "No Entry"
            else:
                print("No MACD Cross Signal")
                self.signals[macd_cross_col] = "No Entry"

    def evaluate_adx(self):
        detector = ADXSignalDetector(self.df, adx_periods=self.adx_periods)
        signals = detector.detect_signals()
        for period in signals:
            adx_col = f"ADX_{period}"
            self.signals[adx_col] = signals[period]

    def run(self):
        self.evaluate_rsi_divergence()
        self.evaluate_rsi_direction()
        self.evaluate_macd_divergence()
        self.evaluate_macd_crossover()
        self.evaluate_adx()
        return self.signals
