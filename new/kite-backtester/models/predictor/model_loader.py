import os
import joblib
import pandas as pd
from typing import Optional, Union


class ModelLoader:
    """
    Loads ML models and provides prediction utilities.
    Supports classification and regression models.
    """

    def __init__(self, model_dir: str = "models/saved_models"):
        """
        Args:
            model_dir (str): Directory where trained models are stored.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}  # Holds loaded models

    def load_model(self, model_name: str) -> None:
        """
        Load a model from disk.

        Args:
            model_name (str): File name of the model (e.g., 'price_dir_model.pkl')
        """
        path = os.path.join(self.model_dir, model_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found: {path}")

        model = joblib.load(path)
        self.models[model_name] = model
        print(f"✅ Loaded model: {model_name}")

    def load_all(self) -> None:
        """
        Automatically load all .pkl models from the directory.
        """
        for file in os.listdir(self.model_dir):
            if file.endswith(".pkl"):
                self.load_model(file)

    def predict(
        self,
        model_name: str,
        X: Union[pd.DataFrame, pd.Series],
        return_proba: bool = False
    ) -> Union[float, int, dict]:
        """
        Make prediction using a loaded model.

        Args:
            model_name (str): Name of the model file.
            X (DataFrame or Series): Feature row(s) for prediction.
            return_proba (bool): Whether to return class probabilities.

        Returns:
            Prediction output or probability dict.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Call load_model() first.")

        model = self.models[model_name]

        # Reshape if only one row provided
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        # Handle probabilities (classification only)
        if return_proba and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            return {cls: float(prob) for cls, prob in zip(classes, probs)}

        # Standard prediction
        prediction = model.predict(X)[0]
        return prediction

    def list_models(self) -> list:
        """List loaded model names."""
        return list(self.models.keys())
