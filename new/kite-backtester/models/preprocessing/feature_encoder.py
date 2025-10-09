import os
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, List, Union
import pandas as pd


class FeatureEncoder:
    """
    A utility class to fit, transform, and persist label or one-hot encoders.
    """

    def __init__(self, save_dir: str = "models/encoders", encoding_type: str = "label"):
        """
        Args:
            save_dir (str): Directory to save or load encoders.
            encoding_type (str): "label" or "onehot"
        """
        self.save_dir = save_dir
        self.encoding_type = encoding_type.lower()
        os.makedirs(save_dir, exist_ok=True)
        self.encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Fits encoders on specified columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to encode.
        """
        for col in columns:
            if self.encoding_type == "label":
                encoder = LabelEncoder()
                self.encoders[col] = encoder.fit(df[col].astype(str))
            elif self.encoding_type == "onehot":
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                self.encoders[col] = encoder.fit(df[[col]].astype(str))
            else:
                raise ValueError("Invalid encoding_type. Choose 'label' or 'onehot'.")

            # Save encoder
            joblib.dump(self.encoders[col], os.path.join(self.save_dir, f"{col}_encoder.pkl"))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns using fitted encoders.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        transformed_df = df.copy()
        for col, encoder in self.encoders.items():
            if self.encoding_type == "label":
                transformed_df[col] = encoder.transform(df[col].astype(str))
            elif self.encoding_type == "onehot":
                encoded = encoder.transform(df[[col]].astype(str))
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=df.index)
                transformed_df = pd.concat([transformed_df.drop(columns=[col]), encoded_df], axis=1)

        return transformed_df

    def load_encoders(self, columns: List[str]) -> None:
        """
        Loads saved encoders for inference or test processing.

        Args:
            columns (List[str]): Column names for which encoders were saved.
        """
        for col in columns:
            path = os.path.join(self.save_dir, f"{col}_encoder.pkl")
            if os.path.exists(path):
                self.encoders[col] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Encoder not found for column: {col}")
