import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureScaler:
    def __init__(self, method='standard', save_dir='models/scalers'):
        """
        Initializes the FeatureScaler.

        Args:
            method (str): Scaling method - 'standard' or 'minmax'.
            save_dir (str): Directory where scalers will be saved/loaded.
        """
        self.method = method.lower()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        if self.method == 'standard':
            self.scaler_cls = StandardScaler
        elif self.method == 'minmax':
            self.scaler_cls = MinMaxScaler
        else:
            raise ValueError("Unsupported method. Use 'standard' or 'minmax'.")

        self.scalers = {}  # {column_name: scaler_object}

    def fit_transform(self, df, columns):
        """
        Fit and transform columns using selected scaler.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to scale.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        for col in columns:
            scaler = self.scaler_cls()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            self._save_scaler(col, scaler)
        return df

    def transform(self, df, columns):
        """
        Apply saved scalers to new data.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to scale.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        for col in columns:
            scaler_path = os.path.join(self.save_dir, f"{col}_scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                df[col] = scaler.transform(df[[col]])
                self.scalers[col] = scaler
            else:
                raise FileNotFoundError(f"Scaler for '{col}' not found at {scaler_path}.")
        return df

    def load_scalers(self, columns):
        """
        Load saved scalers from disk.

        Args:
            columns (List[str]): List of column names for which to load scalers.
        """
        for col in columns:
            path = os.path.join(self.save_dir, f"{col}_scaler.pkl")
            if os.path.exists(path):
                self.scalers[col] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Scaler not found for column: {col}")

    def _save_scaler(self, column, scaler):
        path = os.path.join(self.save_dir, f"{column}_scaler.pkl")
        joblib.dump(scaler, path)
