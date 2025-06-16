import pandas as pd
import numpy as np
import json
import os

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
        self.df['last_close'] = np.nan  # Initialize last close column

        date_col = pd.to_datetime(self.df['date']).dt.date.values

        # Extract columns for candle data
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
        act_close_col = self.df['close'].values
        act_high_col = self.df['high'].values
        act_low_col = self.df['low'].values

        for i in range(len(self.df)):
            curr_close = close_col[i]
            curr_high = high_col[i]
            curr_low = low_col[i]
            cdl_dir = cdl_dir_col[i]
            
            print("-" * 40)
            print(f"Processing row {i}: close={curr_close}, high={curr_high}, low={curr_low}, cdl_dir={cdl_dir}")

            breakout_dir = "sideways"
            breakout_price = curr_close
            found = False
            intermediate_closes = []

            for j in range(1, max_lookahead + 1):
                if i + j >= len(self.df):
                    break

                # Skip if next row is not the same date for intraday trading
                if trading_type == "intraday" and date_col[i + j] != date_col[i]:
                    break

                next_close = act_close_col[i + j]
                next_high = act_high_col[i + j]
                next_low = act_low_col[i + j]
                next_cdl_strength = cdl_strength_col[i + j]
                next_cdl_implication = cdl_implication_col[i + j]
                next_cdl_dir = cdl_dir_col[i + j]
                print(f"  Lookahead {j}: next_close={next_close}, next_high={next_high}, next_low={next_low}")
                if next_close == curr_close and next_high == curr_high and next_low == curr_low:
                    # Skip if no change in price
                    continue
                intermediate_closes.append(next_close)

                if cdl_dir == "up":
                    # Expecting up, but watch for a break of the pattern low
                    if next_close > curr_close:
                        breakout_dir = "up"
                        if next_cdl_strength == "strong" and next_cdl_implication == "reversal" and \
                            next_cdl_dir != cdl_dir:
                            break
                        breakout_price = max(next_close, breakout_price)
                        found = True
                        continue
                    elif next_close < curr_low:
                        if j-i==1:
                            breakout_dir = "down"
                        #breakout_price = next_close  #This is a break in opposite direction, so don't update breakout price
                        found = True
                        break
                    
                elif cdl_dir == "down":
                    # Expecting down, but watch for a break of the pattern high
                    if next_close < curr_close:
                        breakout_dir = "down"
                        if next_cdl_strength == "strong" and next_cdl_implication == "reversal" and \
                            next_cdl_dir != cdl_dir:
                            break
                        breakout_price = min(next_close, breakout_price)
                        found = True
                        continue
                    elif next_close > curr_high:
                        if j-i==1:
                            breakout_dir = "up"
                        #breakout_price = next_close #This is a break in opposite direction, so don't update breakout price
                        found = True
                        break

                elif cdl_dir == "sideways":
                    # Sideways until a break either way
                    if next_high > curr_high:
                        breakout_dir = "up"
                        breakout_price = next_close
                        found = True
                        break
                    elif next_low < curr_low:
                        breakout_dir = "down"
                        breakout_price = next_close
                        found = True
                        break

            # Get base price for % move calc (lowest/highest before breakout)
            prior_segment = intermediate_closes[:-1] if len(intermediate_closes) > 1 else []
            if breakout_dir == "up":
                base_price = min(prior_segment) if prior_segment else curr_close
            elif breakout_dir == "down":
                base_price = max(prior_segment) if prior_segment else curr_close
            else:
                base_price = curr_close

            price_change = abs(breakout_price - base_price)*100 / curr_close if curr_close != 0 else 0
            price_change = round(price_change, 2)
            self.df.at[i, 'last_close'] = breakout_price

            price_dirs.append(breakout_dir)
            price_chgs.append(price_change)

        self.df['price_dir'] = price_dirs
        self.df['price_chg'] = price_chgs
        return self.df

    def run(self) -> pd.DataFrame:
        """
        Run the full candle pattern strategy: metadata mapping + direction derivation.

        :return: Final DataFrame with strategy analysis
        """
        self.map_pattern_metadata()
        self.derive_price_direction()
        return self.df