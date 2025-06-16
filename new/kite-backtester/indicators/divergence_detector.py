import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

class DivergenceDetector:
    """
    Detects bullish and bearish divergences between price and technical indicators
    such as RSI and MACD.
    """

    def __init__(self, df):
        """
        Initialize with a DataFrame containing price and indicators.

        :param df: DataFrame with columns: 'close', 'RSI_<period>', 'MACD'
        """
        self.df = df.copy()

    def _find_extrema(self, series, order=5):
        """
        Find local minima and maxima.

        :param series: Series of values (price or indicator)
        :param order: Number of points to consider for peak detection
        :return: Indices of local minima and maxima
        """
        local_max = argrelextrema(series.values, np.greater_equal, order=order)[0]
        local_min = argrelextrema(series.values, np.less_equal, order=order)[0]
        return local_min, local_max

    def print_divergence_table(self, result):
        """
        Prints the divergence result in a tabulated format, replacing start/end indices with dates.

        Parameters:
        - self: object with self.df containing a datetime index
        - result: list of dictionaries with divergence info
        """
        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)

        # Reset index so date becomes a column
        self.df.reset_index(inplace=True)  # Now 'date' is a column

        rsi_keys = set()
        macd_keys = set()

        for entry in result:
            for key in entry:
                key_lower = key.lower()
                if key_lower.startswith("rsi"):
                    rsi_keys.add(key)
                elif key_lower.startswith("macd"):
                    macd_keys.add(key)

        # Sort by 'end index' in descending order
        sorted_result = sorted(result, key=lambda x: x.get('end index', x.get('index', 0)), reverse=True)

        # Detect available indicator keys (dynamically)
        indicator_keys = set()
        for r in result:
            for key in r:
                if key.lower().startswith(('rsi', 'macd')) and isinstance(r[key], (int, float, np.float64)):
                    indicator_keys.add(key)

        indicator_keys = sorted(indicator_keys)  # Optional: consistent column order

        # Prepare table rows
        table = []
        for r in sorted_result:
            start_idx = int(r.get('start index', r.get('index', 0)))
            end_idx = int(r.get('end index', r.get('index', 0)))

            start_date = self.df.loc[start_idx, 'date']
            end_date = self.df.loc[end_idx, 'date']

            row = [
                r.get('type', 'N/A'),
                start_date.strftime('%Y-%m-%d %H:%M'),
                end_date.strftime('%Y-%m-%d %H:%M'),
                round(float(r.get('price', 0.0)), 2)
            ]

            for key in indicator_keys:
                row.append(round(float(r.get(key, 0.0)), 2))

            table.append(row)

        # Header for the table
        headers = ["Type", "Start Date", "End Date", "Price"] + indicator_keys
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def plot_rsi_with_divergence(self, df, divergences, period):
        plt.figure(figsize=(14, 6))
        
        plt.subplot(2, 1, 2)
        x_vals = range(len(df))
        rsi_col = f'RSI_{period}'
        
        # Plot RSI
        plt.plot(x_vals, df[rsi_col], label=f'RSI {period}', color='orange')
        plt.axhline(70, color='grey', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='grey', linestyle='--', label='Oversold (30)')
        
        # Plot divergence lines
        for div in divergences:
            x_start = div['start index']
            x_end = div['end index']
            y_start = df[rsi_col].iloc[x_start]
            y_end = df[rsi_col].iloc[x_end]

            color = 'green' if div['type'] == 'Bullish' else 'red'
            plt.plot([x_start, x_end], [y_start, y_end], color=color, linestyle='-', linewidth=1.5)

        plt.title('Relative Strength Index (RSI) with Divergences')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def detect_rsi_divergence(self, period=14, order=5):
        """
        Detect bullish/bearish divergence between price and RSI.

        :param period: Period used in RSI calculation (used to access correct column like 'RSI_14').
        :param order: Sensitivity for extrema detection
        :return: List of dicts with divergence info
        """
        rsi_col = f'RSI_{period}'
        self.df[f'RSI_{period}_Div_Type'] = np.nan
        result = []
        used_rsi_indices_bull = set()
        used_rsi_indices_bear = set()
        low_local_min, low_local_max = self._find_extrema(self.df['low'], order)
        high_local_min, high_local_max = self._find_extrema(self.df['high'], order)
        rsi_min, rsi_max = self._find_extrema(self.df[rsi_col], order)
        #rsi_min = np.append(rsi_min, [56, 57, 58]) # Ensure we always have a starting point
        """ print("RSI values at rsi_min indices:")
        for idx in rsi_min:
            print(f"Index: {idx}, RSI: {self.df[rsi_col].iloc[idx]}")
        print(f"low local_min: {low_local_min}, \nhigh local_max: {high_local_max}, \
               \nrsi_min: {rsi_min}, \nrsi_max: {rsi_max}") """
        
        # Bullish divergence on local minima
        for i in range(len(low_local_min)):
            curr = low_local_min[i]

            # Match price local min with RSI local min (allow ±1 index tolerance)
            rsi_curr = next((rsi_idx for rsi_idx in rsi_min if abs(curr - rsi_idx) <= 1), None)
            if rsi_curr is None or rsi_curr in used_rsi_indices_bull:
                continue

            # Look ahead (from higher to lower indices) to find matching divergence
            for j in range(min(len(low_local_min) - 1, i + period), i, -1):
                if low_local_min[j] > low_local_min[i]+15:
                    continue # Ignore large gaps in local minima
                other = low_local_min[j]
                rsi_other = next((rsi_idx for rsi_idx in rsi_min if abs(other - rsi_idx) <= 1), None)
                if rsi_other is None:
                    continue

                if self.df['low'].iloc[curr] > self.df['low'].iloc[other] and \
                self.df[rsi_col].iloc[rsi_curr] < self.df[rsi_col].iloc[rsi_other]:
                    
                    if abs(self.df[rsi_col].iloc[rsi_curr] - self.df[rsi_col].iloc[rsi_other]) > 20:
                        continue  # Ignore very large RSI divergence

                    # Check that no intermediate RSI value is above the connecting line
                    rsi_start = self.df[rsi_col].iloc[rsi_curr]
                    rsi_end = self.df[rsi_col].iloc[rsi_other]
                    index_diff = rsi_other - rsi_curr
                    valid = True

                    for k in [idx for idx in rsi_min if rsi_curr < idx < rsi_other]:                        
                        interpolated_rsi = rsi_start + (rsi_end - rsi_start) * (k - rsi_curr) / index_diff
                        if self.df[rsi_col].iloc[k] < interpolated_rsi:
                            valid = False
                            break

                    if not valid:
                        continue

                    """ print(f"--------------\ncurr: {curr}, rsi_curr: {rsi_curr}")
                    print(f"other: {other}, rsi_other: {rsi_other}")
                    print(f"price_curr: {self.df['low'].iloc[curr]}, price_other: {self.df['low'].iloc[other]}")
                    print(f"rsi_curr: {self.df[rsi_col].iloc[rsi_curr]}, rsi_other: {self.df[rsi_col].iloc[rsi_other]}")
                    print("-----------------------------------") """

                    result.append({
                        'type': 'Bullish',
                        'start index': rsi_curr,
                        'end index': rsi_other,
                        'price': self.df['low'].iloc[other],
                        'rsi': self.df[rsi_col].iloc[rsi_other]
                    })

                    # Mark these indices as used
                    used_rsi_indices_bull.update(range(min(rsi_curr, rsi_other), max(rsi_curr, rsi_other) + 1))
                    break

        # Bearish divergence on local maxima
        for i in range(len(high_local_max)):
            curr = high_local_max[i]

            # Match price local max with RSI local max (allow ±1 index tolerance)
            rsi_curr = next((rsi_idx for rsi_idx in rsi_max if abs(curr - rsi_idx) <= 1), None)
            if rsi_curr is None or rsi_curr in used_rsi_indices_bear:
                continue

            # Look ahead (from higher to lower indices) to find matching divergence
            for j in range(min(len(high_local_max) - 1, i + period), i, -1):
                if high_local_max[j] > high_local_max[i] + 15:
                    continue  # Ignore large gaps in local maxima
                other = high_local_max[j]
                rsi_other = next((rsi_idx for rsi_idx in rsi_max if abs(other - rsi_idx) <= 1), None)
                if rsi_other is None:
                    continue

                if self.df['high'].iloc[curr] < self.df['high'].iloc[other] and \
                self.df[rsi_col].iloc[rsi_curr] > self.df[rsi_col].iloc[rsi_other]:

                    if abs(self.df[rsi_col].iloc[rsi_curr] - self.df[rsi_col].iloc[rsi_other]) > 20:
                        continue  # Ignore very large RSI divergence

                    # Check that no intermediate RSI value is below the connecting line
                    rsi_start = self.df[rsi_col].iloc[rsi_curr]
                    rsi_end = self.df[rsi_col].iloc[rsi_other]
                    index_diff = rsi_other - rsi_curr
                    valid = True

                    for k in [idx for idx in rsi_max if rsi_curr < idx < rsi_other]:
                        interpolated_rsi = rsi_start + (rsi_end - rsi_start) * (k - rsi_curr) / index_diff
                        if self.df[rsi_col].iloc[k] > interpolated_rsi:
                            valid = False
                            break

                    if not valid:
                        continue

                    """ print(f"--------------\ncurr: {curr}, rsi_curr: {rsi_curr}")
                    print(f"other: {other}, rsi_other: {rsi_other}")
                    print(f"price_curr: {self.df['high'].iloc[curr]}, price_other: {self.df['high'].iloc[other]}")
                    print(f"rsi_curr: {self.df[rsi_col].iloc[rsi_curr]}, rsi_other: {self.df[rsi_col].iloc[rsi_other]}")
                    print("-----------------------------------") """

                    result.append({
                        'type': 'Bearish',
                        'start index': rsi_curr,
                        'end index': rsi_other,
                        'price': self.df['high'].iloc[other],
                        'rsi': self.df[rsi_col].iloc[rsi_other]
                    })

                    # Mark these indices as used
                    used_rsi_indices_bear.update(range(min(rsi_curr, rsi_other), max(rsi_curr, rsi_other) + 1))
                    break

        for entry in result:
            start = int(entry['start index'])
            end = int(entry['end index'])
            div_type = entry['type']
            self.df.iloc[start:end+1, self.df.columns.get_loc(f'RSI_{period}_Div_Type')] = div_type

        # Print divergence table
        self.print_divergence_table(result)


        #self.plot_rsi_with_divergence(self.df, divergences=result, period=period)
        return self.df

    def plot_macd_with_divergence(self, df, divergences=None):
        """
        Plot MACD, Signal Line, and Histogram from a DataFrame.
        Optionally highlight MACD divergences using start/end indices.
        """
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        plt.figure(figsize=(14, 6))
        x_vals = range(len(df))

        # Plot MACD and Signal Line
        plt.plot(x_vals, df['MACD'], label='MACD', color='blue', linewidth=1.5)
        plt.plot(x_vals, df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1.5)

        # Plot Histogram
        plt.bar(x_vals, df['MACD_Hist'], label='MACD Histogram', color='grey', alpha=0.5)

        # Highlight MACD divergences
        if divergences:
            for div in divergences:
                x_start = int(div['start index'])
                x_end = int(div['end index'])
                y_start = df['MACD'].iloc[x_start]
                y_end = df['MACD'].iloc[x_end]
                color = 'green' if div['type'].lower() == 'bullish' else 'red'

                # Draw divergence line
                plt.plot([x_start, x_end], [y_start, y_end], linestyle='--', color=color, linewidth=2)
                plt.scatter([x_start, x_end], [y_start, y_end], color=color)

        plt.title('MACD Indicator with Divergences')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def detect_macd_divergence(self, order=5):
        """
        Detect bullish/bearish divergence between price and MACD line.

        :param order: Sensitivity for extrema detection
        :return: List of dicts with divergence info
        """
        result = []
        self.df['MACD_Div_Type'] = np.nan
        used_rsi_indices_bull = set()
        used_rsi_indices_bear = set()
        macd_col = 'MACD'  # or dynamically assigned if needed
        local_min, local_max = self._find_extrema(self.df['close'], order)
        macd_min, macd_max = self._find_extrema(self.df['MACD'], order)
        """ print(f"local_min: {local_min}, \nlocal_max: {local_max}, \nmacd_min: {macd_min}, \nmacd_max: {macd_max}")
        print("MACD values at macd_min indices:")
        for idx in macd_min:
            print(f"Index: {idx}, MACD: {self.df[macd_col].iloc[idx]}") """

        # Bullish divergence on local minima using MACD
        for i in range(len(local_min)):
            curr = local_min[i]

            # Match price local min with MACD local min (±1 index tolerance)
            macd_curr = next((macd_idx for macd_idx in macd_min if abs(curr - macd_idx) <= 1), None)
            if macd_curr is None or macd_curr in used_rsi_indices_bull:
                continue

            # Look ahead (from higher to lower indices)
            for j in range(min(len(local_min) - 1, i + 14), i, -1):
                if local_min[j] > local_min[i] + 30:
                    continue  # Ignore large gaps in local minima

                other = local_min[j]

                # Match other local min with MACD index (±1)
                macd_other = next((macd_idx for macd_idx in macd_min if abs(other - macd_idx) <= 1), None)
                if macd_other is None:
                    continue

                price_curr = self.df['low'].iloc[curr]
                price_other = self.df['low'].iloc[other]
                macd_val_curr = self.df[macd_col].iloc[macd_curr]
                macd_val_other = self.df[macd_col].iloc[macd_other]

                """ # Debugging output
                print(f"--------------\nChecking divergence between indices {curr} and {other}")
                print(f"price_curr: {price_curr}, price_other: {price_other}")
                print(f"curr: {curr}, macd_curr: {macd_curr}")
                print(f"other: {other}, macd_other: {macd_other}") """

                if price_curr > price_other and macd_val_curr < macd_val_other:

                    if abs(macd_val_curr - macd_val_other) > 4:
                        continue  # Ignore large MACD divergence

                    # Validate MACD line: check intermediate MACD points above the connecting line
                    macd_start = macd_val_curr
                    macd_end = macd_val_other
                    index_diff = macd_other - macd_curr
                    valid = True

                    for k in [idx for idx in macd_min if macd_curr < idx < macd_other]:
                        interpolated_macd = macd_start + (macd_end - macd_start) * (k - macd_curr) / index_diff
                        if self.df[macd_col].iloc[k] < interpolated_macd:
                            valid = False
                            break

                    if not valid:
                        continue

                    """ print(f"--------------\ncurr: {curr}, macd_curr: {macd_curr}")
                    print(f"other: {other}, macd_other: {macd_other}")
                    print(f"price_curr: {price_curr}, price_other: {price_other}")
                    print(f"macd_curr: {macd_val_curr}, macd_other: {macd_val_other}")
                    print("-----------------------------------") """

                    result.append({
                        'type': 'Bullish',
                        'start index': macd_curr,
                        'end index': macd_other,
                        'price': price_other,
                        'MACD': macd_val_other
                    })

                    # Mark as used
                    used_rsi_indices_bull.update(range(min(macd_curr, macd_other), max(macd_curr, macd_other) + 1))
                    break

        # Bearish divergence on local maxima using MACD
        for i in range(len(local_max)):
            curr = local_max[i]

            # Match price local max with MACD local max (±1 index tolerance)
            macd_curr = next((macd_idx for macd_idx in macd_max if abs(curr - macd_idx) <= 1), None)
            if macd_curr is None or macd_curr in used_rsi_indices_bear:
                continue

            # Look ahead (from higher to lower indices)
            for j in range(min(len(local_max) - 1, i + 14), i, -1):
                if local_max[j] > local_max[i] + 30:
                    continue  # Ignore large gaps in local maxima

                other = local_max[j]

                # Match other local max with MACD index (±1)
                macd_other = next((macd_idx for macd_idx in macd_max if abs(other - macd_idx) <= 1), None)
                if macd_other is None:
                    continue

                price_curr = self.df['high'].iloc[curr]
                price_other = self.df['high'].iloc[other]
                macd_val_curr = self.df[macd_col].iloc[macd_curr]
                macd_val_other = self.df[macd_col].iloc[macd_other]

                if price_curr < price_other and macd_val_curr > macd_val_other:

                    if abs(macd_val_curr - macd_val_other) > 4:
                        continue  # Ignore very large MACD divergence

                    # Validate MACD line: no intermediate MACD value below the connecting line
                    macd_start = macd_val_curr
                    macd_end = macd_val_other
                    index_diff = macd_other - macd_curr
                    valid = True

                    for k in [idx for idx in macd_max if macd_curr < idx < macd_other]:
                        interpolated_macd = macd_start + (macd_end - macd_start) * (k - macd_curr) / index_diff
                        if self.df[macd_col].iloc[k] > interpolated_macd:
                            valid = False
                            break

                    if not valid:
                        continue

                    """ print(f"--------------\ncurr: {curr}, macd_curr: {macd_curr}")
                    print(f"other: {other}, macd_other: {macd_other}")
                    print(f"price_curr: {price_curr}, price_other: {price_other}")
                    print(f"macd_curr: {macd_val_curr}, macd_other: {macd_val_other}")
                    print("-----------------------------------") """

                    result.append({
                        'type': 'Bearish',
                        'start index': macd_curr,
                        'end index': macd_other,
                        'price': price_other,
                        'MACD': macd_val_other
                    })

                    # Mark as used
                    used_rsi_indices_bear.update(range(min(macd_curr, macd_other), max(macd_curr, macd_other) + 1))
                    break
        
        for entry in result:
            start = int(entry['start index'])
            end = int(entry['end index'])
            div_type = entry['type']
            self.df.iloc[start:end+1, self.df.columns.get_loc(f'MACD_Div_Type')] = div_type

        #self.plot_macd_with_divergence(self.df, divergences=result)
        self.print_divergence_table(result)
        return self.df
