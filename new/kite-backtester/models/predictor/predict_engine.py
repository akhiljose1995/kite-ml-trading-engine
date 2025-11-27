import pandas as pd
import numpy as np
import os
from typing import Dict, Any

from models.preprocessing.preprocessor_predict import PreprocessorPredict
from models.predictor.model_loader import ModelLoader


class PredictEngine:
    """
    A unified engine for single-step model prediction.

    Responsibilities:
    - Load trained ML model
    - Load saved encoders, scalers, label mappings
    - Preprocess new incoming OHLC data (last candle)
    - Return class prediction + probability scores
    """

    def __init__(
        self,
        model_path: str = "models/saved/best_price_dir_model.pkl",
        label_map_path: str = "models/saved/label_mapping.json",
        encoder_dir: str = "models/encoders",
        scaler_dir: str = "models/scalers",
    ):
        """
        Initialize PredictEngine.

        Args:
            model_path (str): Path to the saved model.
            label_map_path (str): Path to JSON label mapping.
            encoder_dir (str): Directory of saved encoders.
            scaler_dir (str): Directory of saved scalers.
        """
        self.model_loader = ModelLoader(
            model_path=model_path,
            label_map_path=label_map_path
        )

        # Load trained model + label map
        self.model = self.model_loader.load_model()
        self.label_map = self.model_loader.load_label_map()

        # Reverse label mapping for output (0→down etc.)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Preprocessor for prediction
        self.preprocessor = PreprocessorPredict(
            encoder_dir=encoder_dir,
            scaler_dir=scaler_dir
        )

    # ---------------------------------------------------------
    def prepare_input(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Perform full preprocessing on the input data.

        Args:
            df_raw (pd.DataFrame): A DF containing latest OHLC rows
                                   (must include at least latest candle)

        Returns:
            pd.DataFrame: Preprocessed feature vector for prediction.
        """
        X = self.preprocessor.run(df_raw)
        return X

    # ---------------------------------------------------------
    def predict(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform prediction on raw OHLC data (single/multiple rows).

        Args:
            df_raw (pd.DataFrame): Raw OHLC rows before feature engineering.

        Returns:
            Dict[str, Any]: Output containing:
                - predicted_label: string label
                - predicted_id: int class id
                - probabilities: dict(label → prob)
        """
        # Step 1: Prepare input feature vector
        X = self.prepare_input(df_raw)
        if X.empty:
            raise ValueError("❌ Preprocessor returned empty feature set.")

        # Step 2: Predict class
        pred_class_id = int(self.model.predict(X)[0])

        # Step 3: Predict class probabilities
        raw_probs = self.model.predict_proba(X)[0]

        # Step 4: Map probs to human-readable labels
        prob_dict = {
            self.reverse_label_map[i]: float(raw_probs[i])
            for i in range(len(raw_probs))
        }

        return {
            "predicted_id": pred_class_id,
            "predicted_label": self.reverse_label_map[pred_class_id],
            "probabilities": prob_dict,
        }

    # ---------------------------------------------------------
    def predict_last_candle(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """
        Convenience method:
        - extracts last row
        - predicts on only that candle

        Args:
            df_raw (pd.DataFrame): Raw DF with many rows.

        Returns:
            Prediction dictionary (same as predict()).
        """
        last_row_df = df_raw.tail(1).copy()
        return self.predict(last_row_df)


# ---------------------------------------------------------
# Manual Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    print("⚡ Running PredictEngine test...")

    # Example dummy input (real pipeline will build features before calling this)
    dummy_df = pd.DataFrame({
        "date": ["2025-06-20 15:15:00"],
        "open": [500],
        "high": [510],
        "low": [495],
        "close": [505],
        "volume": [100000],
        "interval": ["15minute"]
    })

    engine = PredictEngine()
    output = engine.predict(dummy_df)

    print("\nPrediction Output:")
    print(output)
