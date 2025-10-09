import pandas as pd
import re
import os
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


class BasePreprocessor:
    def __init__(self, filepath: str, keywords: Optional[List[str]] = ["RSI", "ADX"]):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.selected_columns = [
            "interval", "vol_zscore", 'EMA_20', 'EMA_20_dist_to_price', 'EMA_50',
            'EMA_50_dist_to_price', 'EMA_200', 'EMA_200_dist_to_price',
            'Distance_EMA_20_EMA_50', 'Distance_EMA_50_EMA_200',
            'Distance_EMA_20_EMA_200', "MACD", "MACD_Signal", "MACD_Hist", "MACD_Div_Len_Nearest_Past",
            "single_candle", "multi_candle", "candle", "CDL_body_size", "type",
            "cdl_emotion", "cdl_strength", "cdl_implication", "cdl_direction",
            "CDL_shadow_ratio", "price_dir", "price_chg"
        ]
        self.selected_columns.extend(self.get_columns_by_keywords(keywords))
        self.label_encoders = {}
        self.scaler = None

    def get_columns_by_keywords(self, keywords: List[str]) -> List[str]:
        col_list = []
        rsi_columns = [col for col in self.df.columns if "rsi" in col.lower()]
        adx_columns = [col for col in self.df.columns if "adx" in col.lower()]

        for col in rsi_columns:
            match = re.match(r'^RSI_(\d+)$', col)
            if match:
                period = match.group(1)
                col_list += [f"RSI_{period}", f"RSI_{period}_Div_Len_Nearest_Past"]

        for col in adx_columns:
            match = re.match(r'^ADX_(\d+)$', col)
            if match:
                period = match.group(1)
                col_list += [
                    f"ADX_{period}", f"PDI_{period}", f"NDI_{period}",
                    f"ADX_{period}_Strength_Label"
                ]

        return col_list

    def filter_columns(self):
        self.df = self.df[self.selected_columns].copy()

    def drop_nulls(self):
        self.df.dropna(inplace=True)

    def encode_categoricals(self, cat_cols: List[str]):
        for col in cat_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

    def scale_columns(self, num_cols: List[str]):
        scaler = StandardScaler()
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.scaler = scaler

    def save_cleaned_data(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df
