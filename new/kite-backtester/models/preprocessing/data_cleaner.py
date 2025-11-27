import pandas as pd
import re
from typing import List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DataCleaner:
    """
    A class to clean and preprocess trading data for model training.

    Attributes:
        filepath (str): Path to the CSV file.
        df (pd.DataFrame): Loaded DataFrame.
        label_encoders (dict): Stores fitted LabelEncoders for categorical columns.
        scaler (StandardScaler): Scaler for numerical features (e.g., price change).
    """

    def __init__(self, filepath: str, keywords: Optional[List[str]] = ["RSI", "ADX", "ATR", "BBW"], model_type: str = "without sideways"):
        """
        Initializes the DataCleaner with a file path.

        Args:
            filepath (str): The path to the CSV file containing the raw data.
            keywords (List[str]): Keywords to dynamically extract feature columns.
        """
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        #Fill "single_candle" and "multi_candle" null values with "unkown"
        self.df['single_candle'].fillna('unknown', inplace=True)
        self.df['multi_candle'].fillna('unknown', inplace=True)
        self.selected_columns = [
            "vol_zscore",
            "EMA_20", "EMA_20_dist_to_price", "EMA_50",
            "EMA_50_dist_to_price", "EMA_200", "EMA_200_dist_to_price",
            "Distance_EMA_20_EMA_50", "Distance_EMA_50_EMA_200", "Distance_EMA_20_EMA_200",
            "MACD", "MACD_Signal", "MACD_Hist", "MACD_Div_Len_Nearest_Past",
            "single_candle", "multi_candle", "candle", "CDL_body_size",
            "CDL_shadow_ratio", "range_to_body_ratio", "price_dir", 
            "price_chg"
        ]
        self.label_encoders = {}
        self.scaler = None
        self.model_type = model_type
        self.selected_columns.extend(self.get_columns_by_keywords(keywords))

    def get_columns_by_keywords(self, keywords: List[str]) -> None:
        """
        Dynamically appends columns that match keyword patterns.

        Args:
            keywords (List[str]): Keywords like 'RSI', 'ADX' to search column names.
        """
        """
        Returns column names that contain any of the specified keywords.

        Args:
            keywords (List[str]): Keywords to match in column names.

        Returns:
            List[str]: List of matching column names.
        """
        rsi_columns = [col for col in self.df.columns if "rsi" in col.lower()]
        col_list = []
        periods = set()
        for col in rsi_columns:
            match = re.match(r'^RSI_(\d+)$', col)
            if not match:
                continue
            periods.add(int(match.group(1)))
        for period in periods:
            col_list.append(f"RSI_{period}_Div_Len_Nearest_Past")
            col_list.append(f"RSI_{period}")

        adx_columns = [col for col in self.df.columns if "adx" in col.lower()]
        periods = set()
        for col in adx_columns:
            match = re.match(r'^ADX_(\d+)$', col)
            if not match:
                continue
            periods.add(int(match.group(1)))
        for period in periods:
            col_list.append(f"ADX_{period}_Strength_Label")
            col_list.append(f"PDI_{period}")
            col_list.append(f"NDI_{period}")
            col_list.append(f"ADX_{period}")

        atr_columns = [col for col in self.df.columns if "atr" in col.lower()]
        periods = set()
        for col in atr_columns:
            match = re.match(r'^ATR_(\d+)$', col)
            if not match:
                continue
            periods.add(int(match.group(1)))
        for period in periods:
            col_list.append(f"ATR_{period}_Percent")

        bbw_columns = [col for col in self.df.columns if "bbw" in col.lower()]
        periods = set()
        for col in bbw_columns:
            match = re.match(r'^BBW_(\d+)$', col)
            if not match:
                continue
            periods.add(int(match.group(1)))
        for period in periods:
            col_list.append(f"BBW_{period}")

        return col_list

    def filter_columns(self) -> None:
        """
        Filters the DataFrame to keep only selected columns.
        """
        if "vol_zscore" not in self.df.columns:
            self.selected_columns.remove("vol_zscore")
        self.df = self.df[self.selected_columns].copy()

    def filter_rows(self, col="cdl_strength", value = "strong", condition="only") -> None:
        """
        Filters the DataFrame to keep only rows with a specific candle strength.

        Args:
            strength (str): The candle strength to filter by (e.g., "strong").
        """
        if condition == "only":
            self.df = self.df[self.df[col] == value].copy()
        else:
            self.df = self.df[self.df[col] != value].copy()
        print(f"Filtered DataFrame for '{col}' '{value}': {self.df.shape[0]} rows remaining.")
    
    def unique_values_in_col(self, col="") -> List[str]:
        """
        Returns the unique values in a specified column.

        Returns:
            List[str]: List of unique values in the column.
        """
        return self.df[col].unique().tolist()

    def filter_price_direction(self, directions: List[str] = ["up", "down"]) -> None:
        """
        Filters rows to retain only specified price directions.

        Args:
            directions (List[str]): Target values for price direction.
        """
        self.df = self.df[self.df["price_dir"].isin(directions)].copy()
        print("Unique price directions after filtering:", self.df["price_dir"].unique())

    def encode_categorical(self, categorical_cols: List[str]) -> None:
        """
        Encodes categorical features using Label Encoding.

        Args:
            categorical_cols (List[str]): List of columns to encode.
        """
        for col in categorical_cols:
            if col in self.df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.df[col] = self.label_encoders[col].fit_transform(self.df[col].astype(str))

    def scale_numerical(self, numerical_cols: List[str]) -> None:
        """
        Applies standard scaling to specified numeric columns.

        Args:
            numerical_cols (List[str]): List of numeric column names.
        """
        self.scaler = StandardScaler()
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])

    def drop_nulls(self) -> None:
        """
        Drops rows with any null values.
        """
        self.df.dropna(inplace=True)

    def get_features_targets(self, df, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns features (X) and target (y) for model training.

        Args:
            target (str): Column name of the target variable.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: X (features), y (target)
        """
        if target == "price_dir":
            df.drop(columns=["price_chg"], inplace=True)
        elif target == "price_chg":
            df.drop(columns=["price_dir"], inplace=True)
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Returns the cleaned and processed DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        return self.df

    def save_to_csv(self, output_path: str) -> None:
        """
        Saves cleaned data to a CSV file.

        Args:
            output_path (str): Output file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)


if __name__ == "__main__":
    cleaner = DataCleaner(filepath="data/STATE_BANK_OF_INDIA.csv")
    cleaner.filter_columns()
    if cleaner.model_type == "without sideways":
        directions = ["up", "down"]
    elif cleaner.model_type == "with sideways":
        directions = ["up", "down", "sideways"]
    cleaner.filter_price_direction(directions=directions)
    cleaner.drop_nulls()

    encode_categorical_list = [
        "interval", "single_candle", "multi_candle", "candle", "type",
        "cdl_emotion", "cdl_strength", "cdl_implication", "cdl_direction"
    ]
    for col in cleaner.get_columns_by_keywords(keywords=["RSI", "ADX"]):
        if "Strength_Label" in col:
            encode_categorical_list.append(col)
    
    # Encode categorical columns
    cleaner.encode_categorical(categorical_cols=encode_categorical_list)

    numerical_col_list = [
        "vol_zscore", "CDL_body_size", "CDL_shadow_ratio", "range_to_body_ratio", 
        "EMA_20", "EMA_20_dist_to_price", "EMA_50",
        "EMA_50_dist_to_price", "EMA_200", "EMA_200_dist_to_price",
        "Distance_EMA_20_EMA_50", "Distance_EMA_50_EMA_200", "Distance_EMA_20_EMA_200"
    ]
    for col in cleaner.get_columns_by_keywords(keywords=["ATR", "BBW"]):
        if "Percent" in col:
            numerical_col_list.append(col)

        if "bbw" in col.lower():
            numerical_col_list.append(col)

    # Scale numerical columns
    cleaner.scale_numerical(numerical_cols=numerical_col_list)

    cleaned_df = cleaner.get_cleaned_data()
    print("Cleaned DataFrame shape:", cleaned_df.shape)
    filename = cleaner.filepath.split("/")[-1]
    cleaner.save_to_csv(f"data/Cleaned_{filename}")
