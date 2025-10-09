import pandas as pd
import os
from models.preprocessing.data_cleaner import DataCleaner
from models.preprocessing.feature_encoder import FeatureEncoder
from models.preprocessing.scaler import FeatureScaler
from typing import List


class PreprocessorTest:
    """
    Handles full preprocessing for testing data: cleaning, encoding, and scaling.
    Loads saved fitted encoders and scalers for transforming test data.
    """

    def __init__(self, filepath: str, model_type: str = "without sideways"):
        self.filepath = filepath
        self.model_type = model_type
        self.data_cleaner = DataCleaner(filepath=filepath, model_type=model_type)
        self.encoder = FeatureEncoder(save_dir="models/encoders", encoding_type="label")
        self.scaler = FeatureScaler(save_dir="models/scalers")

    def run(self, target: str) -> tuple:
        """
        Run the full preprocessing pipeline.

        Args:
            target (str): Name of the target column (e.g., 'price_dir').

        Returns:
            Tuple of (X, y): Processed features and target.
        """
        # Step 1: Filter Columns and Price Direction
        self.data_cleaner.filter_columns()

        # Consider the rows only with cdl_strength == "strong"
        #self.data_cleaner.filter_rows(col="cdl_strength", value="strong")
        #self.data_cleaner.filter_rows(col="candle", value="unknown")
        self.data_cleaner.filter_rows(col="candle", value="unknown", condition="exclude")

        # Filter price direction based on model type
        if target == "price_dir":
            directions = ["up", "down"] if self.model_type == "without sideways" else self.data_cleaner.unique_values_in_col(col="price_dir")
            self.data_cleaner.filter_price_direction(directions)
        self.data_cleaner.drop_nulls()

        df = self.data_cleaner.get_cleaned_data()
        print("Tail 10 rows of DataFrame:\n", df.tail(10))
        print("Null values in DataFrame:", df.isnull().sum())
        print(df.shape, df.columns)
        
        # Step 2: Encode categorical features
        categorical_cols = [
            "single_candle", "multi_candle", "candle"
            #, "type", "cdl_emotion", "cdl_implication", "cdl_direction"
        ]
        for col in self.data_cleaner.get_columns_by_keywords(["RSI", "ADX"]):
            if "Strength_Label" in col:
                categorical_cols.append(col)

        # Load and apply encoders
        self.encoder.load_encoders(categorical_cols)
        df = self.encoder.transform(df)

        # Load and apply scalers
        # Scale numeric columns
        numerical_col_list = ["vol_zscore", "CDL_body_size", "CDL_shadow_ratio", "range_to_body_ratio", 
                            "EMA_20", "EMA_20_dist_to_price", "EMA_50",
                            "EMA_50_dist_to_price", "EMA_200", "EMA_200_dist_to_price",
                            "Distance_EMA_20_EMA_50", "Distance_EMA_50_EMA_200", "Distance_EMA_20_EMA_200"]
        for col in self.data_cleaner.get_columns_by_keywords(keywords=["ATR", "BBW"]):
            if "Percent" in col:
                numerical_col_list.append(col)

            if "bbw" in col.lower():
                numerical_col_list.append(col)
        self.scaler.load_scalers(numerical_col_list)
        df = self.scaler.transform(df, numerical_col_list)

        # Return X, y
        X, y = self.data_cleaner.get_features_targets(df, target)
        
        return X, y

if __name__ == "__main__":
    processor = PreprocessorTest(filepath="data/STATE_BANK_OF_INDIA_test.csv")
    X, y = processor.run(target="price_dir")
    print("Preprocessing completed. Shape:", X.shape, y.shape)

    # Save to CSV
    X.to_csv("data/x_test.csv", index=False)
    y.to_frame(name="price_dir").to_csv("data/y_test.csv", index=False)  # Ensures y is saved as a column
