import pandas as pd
import numpy as np
import json
import os
import config
from general_lib.average import AverageCalculator
from general_lib.general_lib import GeneralLib

class CandlePatternStrategy:
    """
    Strategy to enhance detected candlestick pattern data with classification and price action.
    
    This class uses definitions from a candle_json file to map pattern names to their expected
    type and behavior, then analyzes future price movements to derive trend direction and
    relative strength.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the strategy with dataframe and interval
        
        :param df: DataFrame with detected candlestick patterns
        """
        self.df = df.copy()
        self.candle_json_path = os.path.join("candle.json")
        self.pattern_map = self._load_candle_json()
        self.risk = config.RISK

    def _load_candle_json(self):
        """Load candlestick pattern definitions from candle_json file"""
        with open(self.candle_json_path, 'r') as f:
            return {item['name']: item for item in json.load(f)}

    def map_pattern_metadata(self):
        """
        Map each detected pattern with its metadata (type, exp1, exp2, etc.)
        by referencing the pattern definitions JSON.

        Adds columns dynamically to the dataframe.
        """
        def extract_metadata(row):
            info = self.pattern_map.get(row['candle'], {})
            return pd.Series({k: v for k, v in info.items() if k.startswith("cdl") or k == "type"})

        meta_df = self.df.apply(extract_metadata, axis=1)
        self.df = pd.concat([self.df, meta_df], axis=1)
        return self.df

    def derive_price_direction(self, trading_type = "intraday", max_lookahead: int = 30):
        """
        Detect price direction after pattern using high/low breakout and expected direction (exp4).

        Uses:
        - 'cdl_direction' expected direction ("up", "down", or "sideways")
        - Current candle's high/low as breakout thresholds
        """
        price_dirs = []
        price_chgs = []
        reversal_window = 2
        self.df['last_close'] = np.nan  # Initialize last close column

        date_col = pd.to_datetime(self.df['date']).dt.date.values
        time_col = pd.to_datetime(self.df['date']).dt.time.values

        # Extract columns for candle data
        open_col = self.df['CDL_open'].values
        close_col = self.df['CDL_close'].values
        high_col = self.df['CDL_high'].values
        low_col = self.df['CDL_low'].values

        # Extract columns for candle metadata
        cdl_type_col = self.df['type'].values
        cdl_emotion_col = self.df['cdl_emotion'].values
        cdl_strength_col = self.df['cdl_strength'].values
        cdl_implication_col = self.df['cdl_implication'].values
        cdl_dir_col = self.df['cdl_direction'].values
        
        # Actual price columns for lookahead
        act_open_col = self.df['open'].values
        act_close_col = self.df['close'].values
        act_high_col = self.df['high'].values
        act_low_col = self.df['low'].values

        for i in range(len(self.df)):
            curr_open = open_col[i]
            curr_close = close_col[i]
            curr_high = high_col[i]
            curr_low = low_col[i]
            cdl_dir = cdl_dir_col[i]
            reversal_count = 0
            #sideways_window_h = max(curr_open, curr_close) * (1 + self.risk)
            sideways_window_h = curr_close * (1 + self.risk)
            #sideways_window_l = min(curr_open, curr_close) * (1 - self.risk)
            sideways_window_l = curr_close * (1 - self.risk)
            """ print("-" * 40)
            print(f"Processing row {i}:")
            print(f"date: {time_col[i]}")
            print(f"curr_open: {curr_open}, curr_close: {curr_close}")
            print(f"sideways_window_h: {sideways_window_h}, sideways_window_l: {sideways_window_l}") """
        
            #print("-" * 40)
            #print(f"Processing row {i}: close={curr_close}, high={curr_high}, low={curr_low}, cdl_dir={cdl_dir}")

            breakout_dir = "sideways"
            breakout_price = curr_close
            found = False
            intermediate_closes = []

            # Find the rolling average for next 0.1*max_lookahead values
            avg_calculator = AverageCalculator(act_close_col[i:i + max_lookahead])
            moving_avg = avg_calculator.stepped_moving_average(m=10)
            #print(f"Moving average for row {i}: {moving_avg}")
            if moving_avg is not None:
                if moving_avg > curr_close:
                    breakout_dir = "up"
                elif moving_avg < curr_close:
                    breakout_dir = "down"

            for j in range(1, max_lookahead + 1):
                if i + j >= len(self.df):
                    break

                # Skip if next row is not the same date for intraday trading
                if trading_type == "intraday" and date_col[i + j] != date_col[i]:
                    break
                
                next_open = act_open_col[i + j]
                next_close = act_close_col[i + j]
                next_high = act_high_col[i + j]
                next_low = act_low_col[i + j]
                next_cdl_strength = cdl_strength_col[i + j]
                next_cdl_implication = cdl_implication_col[i + j]
                next_cdl_dir = cdl_dir_col[i + j]
                intermediate_closes.append(next_close)

                
                if breakout_dir == "up":
                    if next_close < curr_open:
                        reversal_count += 1
                        if reversal_count >= reversal_window:
                            break # Reversal detected
                    breakout_price = max(next_close, breakout_price)

                elif breakout_dir == "down":
                    if next_close > curr_open:
                        reversal_count += 1
                        if reversal_count >= reversal_window:
                            break # Reversal detected
                    breakout_price = min(next_close, breakout_price)

            if cdl_dir != breakout_dir and cdl_dir != "trend":
                breakout_dir = "fake"

            # Get base price for % move calc (lowest/highest before breakout)
            #prior_segment = intermediate_closes[:-1] if len(intermediate_closes) > 1 else []
            prior_segment = [curr_close] + intermediate_closes[:-1]
            if breakout_dir == "up":
                base_price = min(prior_segment) if prior_segment else curr_close
            elif breakout_dir == "down":
                base_price = max(prior_segment) if prior_segment else curr_close
            else:
                base_price = curr_close

            price_change = (breakout_price - base_price)*100 / curr_close if curr_close != 0 else 0
            price_change = round(price_change, 2)
            self.df.at[i, 'last_close'] = breakout_price

            price_dirs.append(breakout_dir)
            price_chgs.append(price_change)

        self.df['price_dir'] = price_dirs
        self.df['price_chg'] = price_chgs
        self.df['price_chg'] = self.df['price_chg'].fillna(0)
        return self.df

    def updating_sideways_price_dir(self, keep_fake=True):
        """
        Update up/down moves by checking the price change we have obtained from derive_price_direction()
        We will calculate the absolute price change and if it is less than 0.25% then we will mark it as sideways.
        This needs to be done only for the rows where price_dir is not already sideways.
        """
        # Take only rows where price_dir is not sideways
        non_sideways_df = self.df.copy()
        if non_sideways_df.empty:
            return self.df
        # Calculate absolute price change
        non_sideways_df['abs_price_chg'] = non_sideways_df['price_chg'].abs()

        # Get the list of price_chg, where it is neither 0 nor NaN
        temp_df = non_sideways_df[non_sideways_df['abs_price_chg'].notna() & (non_sideways_df['abs_price_chg'] != 0)].copy()

        # Find the 25th percentile of absolute price change
        quantile_call = GeneralLib
        q25 = quantile_call.calculate_quantile(
            temp_df['abs_price_chg'].tolist(), 0.25)
        q75 = quantile_call.calculate_quantile(
            temp_df['abs_price_chg'].tolist(), 0.75)
        #threshold = non_sideways_df['abs_price_chg'].quantile(0.20)

        print(f"Price Change 25th percentile for sideways detection: {q25:.2f}%")
        print(f"Price Change 75th percentile: {q75:.2f}%")

        # Update price_dir to sideways if abs_price_chg is less than threshold
        #self.df.loc[non_sideways_df['abs_price_chg'] < threshold, 'price_dir'] = 'sideways'
        updated_dirs = []
        for i, row in self.df.iterrows():
            dir_now = row['price_dir']
            chg = row['price_chg']

            # Preserve fake if required
            if keep_fake and dir_now == "fake":
                updated_dirs.append("fake")
                continue

            # Assign based on quantiles
            if abs(chg) < q25:
                updated_dirs.append("sideways")
            #elif q25 <= abs(chg) < q75:
            #    updated_dirs.append("small up" if chg > 0 else "small down")
            #else:
            #    updated_dirs.append("big up" if chg > 0 else "big down")
            else:
                updated_dirs.append(dir_now)

        self.df['price_dir'] = updated_dirs

        # print tail of the DataFrame for debugging
        print("Rows with price_dir updated to sideways:")
        print(self.df.tail(10)[['date', 'candle', 'cdl_implication', 'cdl_strength', 'cdl_direction', 'price_chg', 'price_dir']])


    def run(self, trading_type="intraday", max_lookahead=30) -> pd.DataFrame:
        """
        Run the full candle pattern strategy: metadata mapping + direction derivation.

        :return: Final DataFrame with strategy analysis
        """
        self.map_pattern_metadata()
        self.derive_price_direction(trading_type = trading_type, max_lookahead = max_lookahead)
        self.updating_sideways_price_dir()
        return self.df