# models/preprocessing/preprocessor_predict.py

import pandas as pd
import os
from typing import List

from models.preprocessing.data_cleaner import DataCleaner
from models.preprocessing.feature_encoder import FeatureEncoder
from models.preprocessing.scaler import FeatureScaler


class PreprocessorPredict:
    """
    Preprocessor used during live prediction.
    - Does NOT drop rows.
    - Does NOT filter direction.
    - Loads saved encoders/scalers.
    - Returns ONLY transformed X (no target).
    """

    def __init__(self, model_type: str = "without sideways"):
        self.model_type = model_type
        self.encoder = FeatureEncoder(save_dir="models/encoders", encoding_type="label")
        self.scaler = FeatureScaler(save_dir="models/scalers")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run preprocessing on a LIVE DATAFRAME.
        df is the last few rows of features after feature generation.
        Returns processed X of ONLY the last row.
        """

        # Make a copy
        df = df.copy()

        # ---- STEP 1: Basic Cleaning ----
        # Replace missing candle values
        for col in ["single_candle", "multi_candle", "candle"]:
            if col in df.columns:
                df[col].fillna("unknown", inplace=True)

        # Fill NA numerics forward/backward then with 0
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.fillna(0, inplace=True)

        # ---- STEP 2: Determine categorical and numerical columns ----
        categorical_cols = [
            "single_candle", "multi_candle", "candle"
        ]

        # Add RSI/ADX strength labels dynamically
        for col in df.columns:
            if "Strength_Label" in col:
                categorical_cols.append(col)

        numerical_cols = [
            "vol_zscore", "CDL_body_size", "CDL_shadow_ratio",
            "ATR_14", "BBW_20"
        ]

        # Add any EMA or distance columns if present
        for col in df.columns:
            if any(x in col for x in ["EMA", "Distance_", "dist_to_price"]):
                numerical_cols.append(col)

        numerical_cols = [col for col in numerical_cols if col in df.columns]

        # ---- STEP 3: Load encoders and transform ----
        self.encoder.load_encoders(categorical_cols)
        df = self.encoder.transform(df)

        # ---- STEP 4: Load scalers and transform ----
        self.scaler.load_scalers(numerical_cols)
        df = self.scaler.transform(df, numerical_cols)

        # ---- STEP 5: Drop target columns if present ----
        for col in ["price_dir", "price_chg"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ---- STEP 6: Return only last row ----
        X = df.tail(1).copy()
        return X

