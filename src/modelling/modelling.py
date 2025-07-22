import pandas as pd
import numpy as np
import pickle
import logging
from typing import Any, Tuple
from sklearn.ensemble import RandomForestClassifier
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("modelling.log"),
        logging.StreamHandler()
    ]
)

def read_params(params_path: str = "params.yaml") -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to read parameters file: {e}")
        raise

def load_train_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded training data from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data from {path}: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df.drop(columns=['label']).values
        y = df['label'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def train_model(X: np.ndarray, y: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X, y)
        logging.info("RandomForest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: Any, path: str) -> None:
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model to {path}: {e}")
        raise

def main() -> None:
    try:
        params = read_params()
        n_estimators = params["model_building"]["n_estimators"]
        max_depth = params["model_building"]["max_depth"]

        train_data = load_train_data("data/interim/train_bow.csv")
        X_train, y_train = extract_features_labels(train_data)

        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")

        logging.info("Model building pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model building pipeline failed: {e}")

if __name__ == "__main__":
    main()