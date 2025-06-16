import pandas as pd
import numpy as np

class CandlePatternDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_patterns(self) -> pd.DataFrame:
        self.df['single_candle'] = np.nan
        self.df['multi_candle'] = np.nan
        self.df['candle'] = np.nan
        self.df['CDL_open'] = np.nan
        self.df['CDL_high'] = np.nan
        self.df['CDL_low'] = np.nan
        self.df['CDL_close'] = np.nan

        patterns = [
            self.doji,
            self.hammer,
            self.inverted_hammer,
            self.hanging_man,
            self.shooting_star,
            self.marubozu_bullish,
            self.marubozu_bearish,
            self.bullish_engulfing,
            self.bearish_engulfing,
            self.piercing_pattern,
            self.dark_cloud_cover,
            self.morning_star,
            self.evening_star,
            self.morning_doji_star,
            self.evening_doji_star,
            self.three_white_soldiers,
            self.three_black_crows,
            self.bullish_harami,
            self.bearish_harami,
            self.bullish_kicker,
            self.bearish_kicker,
            self.bullish_belt_hold,
            self.bearish_belt_hold,
            self.tweezer_bottom,
            self.tweezer_top,
            self.upside_gap_two_crows,
            self.two_black_gapping,
            self.abandoned_baby_bullish,
            self.abandoned_baby_bearish,
            self.separating_lines_bullish,
            self.separating_lines_bearish,
            self.thrusting,
            self.counterattack,
            self.deliberation,
            self.tristar_bullish,
            self.tristar_bearish,
            self.mat_hold_bullish,
            self.mat_hold_bearish,
            self.rising_three_methods,
            self.falling_three_methods,
            self.advance_block,
            self.stalled_pattern
        ]

        pattern_classification = {
            "doji": "single",
            "hammer": "single",
            "inverted_hammer": "single",
            "hanging_man": "single",
            "shooting_star": "single",
            "marubozu": "single",
            "bullish_belt_hold": "single",
            "bearish_belt_hold": "single",

            "bullish_engulfing": "multi",
            "bearish_engulfing": "multi",
            "piercing_pattern": "multi",
            "dark_cloud_cover": "multi",
            "morning_star": "multi",
            "evening_star": "multi",
            "morning_doji_star": "multi",
            "evening_doji_star": "multi",
            "three_white_soldiers": "multi",
            "three_black_crows": "multi",
            "bullish_harami": "multi",
            "bearish_harami": "multi",
            "bullish_kicker": "multi",
            "bearish_kicker": "multi",
            "tweezer_bottom": "multi",
            "tweezer_top": "multi",
            "upside_gap_two_crows": "multi",
            "two_black_gapping": "multi",
            "abandoned_baby_bullish": "multi",
            "abandoned_baby_bearish": "multi",
            "separating_lines_bullish": "multi",
            "separating_lines_bearish": "multi",
            "thrusting": "multi",
            "counterattack": "multi",
            "deliberation": "multi",
            "tristar_bullish": "multi",
            "tristar_bearish": "multi",
            
            "mat_hold_bullish": "multi",
            "mat_hold_bearish": "multi",
            "rising_three_methods": "multi",
            "falling_three_methods": "multi",
            "advance_block": "multi",
            "stalled_pattern": "multi"
        }

        for pattern_fn in patterns:
            pattern_name, mask, o, h, l, c = pattern_fn()
            candle_type = pattern_classification.get(pattern_name, "unknown")

            self.df.loc[mask, 'CDL_open'] = o
            self.df.loc[mask, 'CDL_high'] = h
            self.df.loc[mask, 'CDL_low'] = l
            self.df.loc[mask, 'CDL_close'] = c

            if candle_type == "single":
                self.df.loc[mask, 'single_candle'] = pattern_name
                # Only write to Candle column if it's still empty (i.e., no multi written yet)
                self.df.loc[mask & self.df['candle'].isna(), 'candle'] = pattern_name

            elif candle_type == "multi":
                self.df.loc[mask, 'multi_candle'] = pattern_name
                # Always overwrite with multi pattern
                self.df.loc[mask, 'candle'] = pattern_name

        return self.df

    def body(self, o, c):
        return abs(c - o)

    def shadow_upper(self, h, o, c):
        return h - max(o, c)

    def shadow_lower(self, l, o, c):
        return min(o, c) - l

    def doji(self):
        body = abs(self.df['open'] - self.df['close'])
        range_ = self.df['high'] - self.df['low']
        return "doji", (body <= 0.1 * range_), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def hammer(self):
        body = abs(self.df['open'] - self.df['close'])
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        return "hammer", ((lower_shadow > 2 * body) & (upper_shadow < body)), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def inverted_hammer(self):
        body = abs(self.df['open'] - self.df['close'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        return "inverted_hammer", ((upper_shadow > 2 * body) & (lower_shadow < body)), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def hanging_man(self):
        _, hammer_mask, *_ = self.hammer()
        hanging_man_mask = hammer_mask & (self.df['close'] < self.df['open'])  # Bearish body
        return "hanging_man", hanging_man_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']
    
    def shooting_star(self):
        body = abs(self.df['open'] - self.df['close'])
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        
        # Shooting star has a long upper shadow, small body near low
        shooting_star_mask = (upper_shadow > 2 * body) & (lower_shadow < body) & (self.df['close'] < self.df['open'])
        
        return "shooting_star", shooting_star_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']
    
    def marubozu_bullish(self, threshold=0.01):
        body = abs(self.df['open'] - self.df['close'])
        range_ = self.df['high'] - self.df['low']
        
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        
        # Shadows should be very small compared to range
        marubozu_mask = (lower_shadow <= threshold * range_) & (upper_shadow <= threshold * range_)
        marubozu_mask &= (self.df['close'] > self.df['open'])
        return "marubozu_bullish", marubozu_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']
    
    def marubozu_bearish(self, threshold=0.01):
        body = abs(self.df['open'] - self.df['close'])
        range_ = self.df['high'] - self.df['low']
        
        lower_shadow = self.df[['open', 'close']].min(axis=1) - self.df['low']
        upper_shadow = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        
        # Shadows should be very small compared to range
        marubozu_mask = (lower_shadow <= threshold * range_) & (upper_shadow <= threshold * range_)
        marubozu_mask &= (self.df['close'] < self.df['open'])
        return "marubozu_bearish", marubozu_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']
    
    def bullish_engulfing(self):
        prev = self.df.shift(1)
        return "bullish_engulfing", ((prev['close'] < prev['open']) &
                                      (self.df['close'] > self.df['open']) &
                                      (self.df['open'] < prev['close']*0.99) &
                                      (self.df['close'] > prev['open'])), \
                                        self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def bearish_engulfing(self):
        prev = self.df.shift(1)
        return "bearish_engulfing", ((prev['close'] > prev['open']) &
                                      (self.df['close'] < self.df['open']) &
                                      (self.df['open'] > prev['close']*0.99) &
                                      (self.df['close'] < prev['open'])), \
                                        self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def piercing_pattern(self):
        prev = self.df.shift(1)
        mid = (prev['open'] + prev['close']) / 2

        piercing_mask = (
            (prev['close'] < prev['open']) &               # Previous candle is bearish
            (self.df['open'] < prev['low']) &              # Gap down open
            (self.df['close'] > mid) &                     # Close above midpoint of prev body
            (self.df['close'] < prev['open']) &            # Close still below prev open
            (self.df['close'] > self.df['open'])           # Current candle is bullish
        )

        return "piercing_pattern", piercing_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def dark_cloud_cover(self):
        prev = self.df.shift(1)
        mid = (prev['open'] + prev['close']) / 2

        dark_cloud_mask = (
            (prev['close'] > prev['open']) &                # Previous bullish candle
            (self.df['open'] > prev['high']) &              # Gap up
            (self.df['close'] < mid) &                      # Close below midpoint of prev body
            (self.df['close'] > prev['open']) &             # Not a full bearish engulfing
            (self.df['close'] < self.df['open'])            # Current candle is bearish
        )

        return "dark_cloud_cover", dark_cloud_mask, self.df['open'], self.df['high'], self.df['low'], self.df['close']
    
    def morning_star(self):
        prev = self.df.shift(1)
        next_ = self.df.shift(-1)

        # Midpoint of the first (bearish) candle
        prev_mid = (prev['open'] + prev['close']) / 2
        body = abs(self.df['close'] - self.df['open'])

        # Optional: small-body check for the middle candle
        range_ = self.df['high'] - self.df['low']
        small_body = body < 0.3 * range_

        morning_star_mask = (
            (prev['close'] < prev['open']) &                # First candle is bearish
            (small_body) &                                  # Second candle has small body (indecision)
            (next_['close'] > next_['open']) &              # Third candle is bullish
            (next_['close'] > prev_mid)                     # Third candle closes above midpoint of first
        )

        return "morning_star", morning_star_mask, next_['open'], next_['high'], next_['low'], next_['close']

    def evening_star(self):
        prev = self.df.shift(1)
        next_ = self.df.shift(-1)

        # Midpoint of the first bullish candle
        prev_mid = (prev['open'] + prev['close']) / 2

        # Middle candle's body size
        body = abs(self.df['close'] - self.df['open'])
        range_ = self.df['high'] - self.df['low']
        small_body = body < 0.3 * range_

        evening_star_mask = (
            (prev['close'] > prev['open']) &            # First candle is bullish
            (small_body) &                              # Second candle has small body (indecision)
            (next_['close'] < next_['open']) &          # Third candle is bearish
            (next_['close'] < prev_mid)                 # Third candle closes below midpoint of first
        )

        return "evening_star", evening_star_mask, next_['open'], next_['high'], next_['low'], next_['close']

    def morning_doji_star(self):
        return "morning_doji_star", self.morning_star()[1] & self.doji()[1], \
                self.morning_star()[2], self.morning_star()[3], self.morning_star()[4], self.morning_star()[5]

    def evening_doji_star(self):
        return "evening_doji_star", self.evening_star()[1] & self.doji()[1], \
                self.evening_star()[2], self.evening_star()[3], self.evening_star()[4], self.evening_star()[5]

    def three_white_soldiers(self):
        a = self.df.shift(2)  # First candle
        b = self.df.shift(1)  # Second candle
        c = self.df           # Third candle

        pattern_mask = (
            (a['close'] > a['open']) &
            (b['close'] > b['open']) &
            (c['close'] > c['open']) &
            (b['open'] >= a['open']) & (b['open'] <= a['close']) &  # b opens within a's body
            (c['open'] >= b['open']) & (c['open'] <= b['close']) &  # c opens within b's body
            (b['close'] > a['close']) &
            (c['close'] > b['close'])
        )

        return "three_white_soldiers", pattern_mask, c['open'], c['high'], c['low'], c['close']

    def three_black_crows(self):
        a = self.df.shift(2)  # First candle
        b = self.df.shift(1)  # Second candle
        c = self.df           # Third candle

        pattern_mask = (
            (a['close'] < a['open']) &
            (b['close'] < b['open']) &
            (c['close'] < c['open']) &

            (b['open'] >= a['close']) & (b['open'] <= a['open']) &  # b opens within a's body
            (c['open'] >= b['close']) & (c['open'] <= b['open']) &  # c opens within b's body

            (b['close'] < a['close']) &
            (c['close'] < b['close'])
        )

        return "three_black_crows", pattern_mask, c['open'], c['high'], c['low'], c['close']

    def bullish_harami(self):
        prev = self.df.shift(1)
        return "bullish_harami", (
            (prev['open'] > prev['close']) &                    # Previous candle is bearish
            (self.df['close'] > self.df['open']) &              # Current candle is bullish
            (self.df['open'] > prev['close']) &                 # Current open above prev close
            (self.df['close'] < prev['open'])                   # Current close below prev open
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def bearish_harami(self):
        prev = self.df.shift(1)
        return "bearish_harami", (
            (prev['close'] > prev['open']) &                    # Previous candle is bullish
            (self.df['close'] < self.df['open']) &              # Current candle is bearish
            (self.df['open'] < prev['close']) &                 # Current open below prev close
            (self.df['close'] > prev['open'])                   # Current close above prev open
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def bullish_kicker(self):
        prev = self.df.shift(1)
        gap = self.df['open'] - prev['close']
        body_prev = abs(prev['close'] - prev['open'])

        abs_gap = gap > 0.1 * body_prev

        return "bullish_kicker", (
            (prev['close'] < prev['open']) &              # Previous candle bearish
            (self.df['open'] > prev['close']) &           # Gap up
            (self.df['close'] > self.df['open']) &        # Current candle bullish
            abs_gap                                       # Significant gap
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']


    def bearish_kicker(self):
        prev = self.df.shift(1)
        gap = prev['close'] - self.df['open']
        body_prev = abs(prev['close'] - prev['open'])

        abs_gap = gap > 0.1 * body_prev

        return "bearish_kicker", (
            (prev['close'] > prev['open']) &              # Previous candle bullish
            (self.df['open'] < prev['close']) &           # Gap down
            (self.df['close'] < self.df['open']) &        # Current candle bearish
            abs_gap                                       # Significant gap
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def bullish_belt_hold(self):
        body = self.df['close'] - self.df['open']
        lower_shadow = self.df['open'] - self.df['low']
        return "bullish_belt_hold", (
            (body > 0) &                                   # Bullish candle
            (lower_shadow <= 0.1 * body)                   # Open near low
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']


    def bearish_belt_hold(self):
        body = self.df['open'] - self.df['close']
        upper_shadow = self.df['high'] - self.df['open']
        return "bearish_belt_hold", (
            (body > 0) &                                   # Bearish candle
            (upper_shadow <= 0.1 * body)                   # Open near high
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def tweezer_bottom(self):
        prev = self.df.shift(1)
        low_diff = abs(prev['low'] - self.df['low'])
        tolerance = 0.01 * (self.df['high'] - self.df['low'])  # 1% of current candle's range

        return "tweezer_bottom", (
            (low_diff <= tolerance) &
            (prev['close'] < prev['open']) &                   # Previous candle bearish
            (self.df['close'] > self.df['open'])               # Current candle bullish
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']


    def tweezer_top(self):
        prev = self.df.shift(1)
        high_diff = abs(prev['high'] - self.df['high'])
        tolerance = 0.01 * (self.df['high'] - self.df['low'])  # 1% of current candle's range

        return "tweezer_top", (
            (high_diff <= tolerance) &
            (prev['close'] > prev['open']) &                   # Previous candle bullish
            (self.df['close'] < self.df['open'])               # Current candle bearish
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def upside_gap_two_crows(self):
        a = self.df.shift(2)  # First candle (bullish)
        b = self.df.shift(1)  # Second candle (bearish with gap up)
        c = self.df           # Third candle (bearish)

        gap1 = b['open'] - a['close']
        gap2 = c['open'] - b['open']
        min_gap = 0.001 * a['close']  # Small gap tolerance (can tune this)

        return "upside_gap_two_crows", (
            (a['close'] > a['open']) &                      # 1st candle bullish
            (b['close'] < b['open']) &                      # 2nd candle bearish
            (gap1 > min_gap) &                              # 2nd candle opens above 1st candle close
            (c['close'] < c['open']) &                      # 3rd candle bearish
            (gap2 > 0) &                                     # 3rd candle opens above 2nd open
            (c['close'] < b['close'])                       # Closes lower than 2nd candle
        ), c['open'], c['high'], c['low'], c['close']

    def two_black_gapping(self):
        a = self.df.shift(2)  # Candle -2 (prior trend)
        b = self.df.shift(1)  # Candle -1 (first black candle)
        c = self.df           # Candle 0 (second black candle)

        pattern = (
            (b['close'] < b['open']) &                    # Candle -1 is bearish
            (c['close'] < c['open']) &                    # Candle 0 is bearish
            (b['open'] > a['high']) &                     # Gap up from previous high
            (c['open'] < b['close'])                      # Gap down from previous close
        )

        return "two_black_gapping", pattern, c['open'], c['high'], c['low'], c['close']

    def abandoned_baby_bullish(self):
        prev = self.df.shift(1)
        curr = self.df
        next_ = self.df.shift(-1)

        # Doji condition
        body = abs(curr['open'] - curr['close'])
        range_ = curr['high'] - curr['low']
        doji_mask = body <= 0.1 * range_

        # Bullish Abandoned Baby logic
        first_bearish = prev['close'] < prev['open']
        gap_down = curr['high'] < prev['low']
        third_bullish = next_['close'] > next_['open']
        gap_up = next_['low'] > curr['high']

        bullish_mask = (
            first_bearish &
            doji_mask &
            gap_down &
            third_bullish &
            gap_up
        )

        return (
            "abandoned_baby_bullish",
            bullish_mask,
            next_['open'],
            next_['high'],
            next_['low'],
            next_['close']
        )
    
    def abandoned_baby_bearish(self):
        prev = self.df.shift(1)
        curr = self.df
        next_ = self.df.shift(-1)

        # Doji condition
        body = abs(curr['open'] - curr['close'])
        range_ = curr['high'] - curr['low']
        doji_mask = body <= 0.1 * range_

        # Bearish Abandoned Baby logic
        first_bullish = prev['close'] > prev['open']
        gap_up = curr['low'] > prev['high']
        third_bearish = next_['close'] < next_['open']
        gap_down = next_['high'] < curr['low']

        bearish_mask = (
            first_bullish &
            doji_mask &
            gap_up &
            third_bearish &
            gap_down
        )

        return (
            "abandoned_baby_bearish",
            bearish_mask,
            next_['open'],
            next_['high'],
            next_['low'],
            next_['close']
        )

    def separating_lines_bullish(self):
        prev = self.df.shift(1)
        return "separating_lines_bullish", (
            (prev['close'] < prev['open']) &
            (abs(self.df['open'] - prev['open']) <= 0.01) &
            (self.df['close'] > self.df['open'])
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def separating_lines_bearish(self):
        prev = self.df.shift(1)
        return "separating_lines_bearish", (
            (prev['close'] > prev['open']) &
            (abs(self.df['open'] - prev['open']) <= 0.01) &
            (self.df['close'] < self.df['open'])
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def thrusting(self):
        prev = self.df.shift(1)
        mid = (prev['open'] + prev['close']) / 2
        return "thrusting", (
            (prev['close'] > prev['open']) &  # Previous bullish
            (self.df['open'] < prev['low']) &  # Gap down
            (self.df['close'] > prev['open']) &  # Closes inside previous body
            (self.df['close'] < mid)  # But below the midpoint
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def counterattack(self):
        prev = self.df.shift(1)
        return "counterattack", (
            (prev['close'] > prev['open']) &  # First bullish
            (self.df['open'] > prev['close']) &  # Gap up
            (abs(self.df['close'] - prev['open']) <= 0.01)  # Closes at open of prev (use tolerance)
        ), self.df['open'], self.df['high'], self.df['low'], self.df['close']

    def deliberation(self):
        a = self.df.shift(2)
        b = self.df.shift(1)
        c = self.df

        body_c = c['close'] - c['open']
        body_b = b['close'] - b['open']
        close_near_prev = abs(c['close'] - b['close']) <= 0.25 * body_b
        upper_shadow = c['high'] - c['close']
        
        return "deliberation", (
            (a['close'] > a['open']) &
            (b['close'] > b['open']) &
            (c['close'] > c['open']) &
            (body_c < body_b) &
            close_near_prev &
            (upper_shadow < body_c)  # Optional: small upper shadow
        ), c['open'], c['high'], c['low'], c['close']

    def tristar_bullish(self, doji_ratio_threshold=0.05):
        prev = self.df.shift(1)
        curr = self.df
        next_ = self.df.shift(-1)

        # Stricter Doji condition
        body = abs(curr['open'] - curr['close'])
        range_ = curr['high'] - curr['low']
        doji_mask = (range_ > 0) & (body <= doji_ratio_threshold * range_)

        # Doji at t-1, t, and t+1
        prev_doji = doji_mask.shift(1).fillna(False)
        curr_doji = doji_mask
        next_doji = doji_mask.shift(-1).fillna(False)

        # Middle doji is lowest
        lower_middle = (
            (curr['low'] < prev['low']) &
            (curr['low'] < next_['low'])
        )

        bullish_mask = prev_doji & curr_doji & next_doji & lower_middle

        return "tristar_bullish", bullish_mask, next_['open'], next_['high'], next_['low'], next_['close']


    def tristar_bearish(self, doji_ratio_threshold=0.05):
        prev = self.df.shift(1)
        curr = self.df
        next_ = self.df.shift(-1)

        # Stricter Doji condition
        body = abs(curr['open'] - curr['close'])
        range_ = curr['high'] - curr['low']
        doji_mask = (range_ > 0) & (body <= doji_ratio_threshold * range_)

        # Doji at t-1, t, and t+1
        prev_doji = doji_mask.shift(1).fillna(False)
        curr_doji = doji_mask
        next_doji = doji_mask.shift(-1).fillna(False)

        # Middle doji is highest
        higher_middle = (
            (curr['high'] > prev['high']) &
            (curr['high'] > next_['high'])
        )

        bearish_mask = prev_doji & curr_doji & next_doji & higher_middle

        return "tristar_bearish", bearish_mask, next_['open'], next_['high'], next_['low'], next_['close']

    def mat_hold_bullish(self):
        a = self.df.shift(4)
        b = self.df.shift(3)
        c = self.df.shift(2)
        d = self.df.shift(1)
        e = self.df

        pattern = (
            (a['close'] > a['open']) &
            (b['close'] < b['open']) &
            (c['close'] < c['open']) &
            (d['close'] < d['open']) &
            (e['close'] > e['open']) &
            (e['close'] > a['close'])  # Breaks previous high
        )

        return "mat_hold_bullish", pattern, e['open'], e['high'], e['low'], e['close']

    def mat_hold_bearish(self):
        a = self.df.shift(4)
        b = self.df.shift(3)
        c = self.df.shift(2)
        d = self.df.shift(1)
        e = self.df

        pattern = (
            (a['close'] < a['open']) &
            (b['close'] > b['open']) &
            (c['close'] > c['open']) &
            (d['close'] > d['open']) &
            (e['close'] < e['open']) &
            (e['close'] < a['close'])  # Breaks previous low
        )

        return "mat_hold_bearish", pattern, e['open'], e['high'], e['low'], e['close']

    def rising_three_methods(self):
        a = self.df.shift(4)
        b = self.df.shift(3)
        c = self.df.shift(2)
        d = self.df.shift(1)
        e = self.df

        pattern = (
            (a['close'] > a['open']) &
            (b['close'] < b['open']) & (b['close'] > a['open']) &
            (c['close'] < c['open']) & (c['close'] > a['open']) &
            (d['close'] < d['open']) & (d['close'] > a['open']) &
            (e['close'] > e['open']) & (e['close'] > a['close'])
        )

        return "rising_three_methods", pattern, e['open'], e['high'], e['low'], e['close']

    def falling_three_methods(self):
        a = self.df.shift(4)
        b = self.df.shift(3)
        c = self.df.shift(2)
        d = self.df.shift(1)
        e = self.df

        pattern = (
            (a['close'] < a['open']) &
            (b['close'] > b['open']) & (b['close'] < a['open']) &
            (c['close'] > c['open']) & (c['close'] < a['open']) &
            (d['close'] > d['open']) & (d['close'] < a['open']) &
            (e['close'] < e['open']) & (e['close'] < a['close'])
        )

        return "falling_three_methods", pattern, e['open'], e['high'], e['low'], e['close']

    def advance_block(self):
        a = self.df.shift(2)
        b = self.df.shift(1)
        c = self.df

        pattern = (
            (a['close'] > a['open']) &
            (b['close'] > b['open']) & (b['open'] > a['open']) & (b['close'] < a['close']) &
            (c['close'] > c['open']) & (c['open'] > b['open']) & (c['close'] < b['close'])  # smaller body
        )

        return "advance_block", pattern, c['open'], c['high'], c['low'], c['close']

    def stalled_pattern(self):
        a = self.df.shift(2)
        b = self.df.shift(1)
        c = self.df

        pattern = (
            (a['close'] > a['open']) &
            (b['close'] > b['open']) & (b['open'] > a['open']) &
            (c['close'] > c['open']) & ((c['close'] - c['open']) < (b['close'] - b['open']) * 0.5)
        )

        return "stalled_pattern", pattern, c['open'], c['high'], c['low'], c['close']

