import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("features.log"),
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

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path).dropna(subset=['content'])
        logging.info(f"Loaded data from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X = df['content'].values
        y = df['sentiment'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features and labels: {e}")
        raise

def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Text vectorization completed.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Text vectorization failed: {e}")
        raise

def save_features(X: np.ndarray, y: np.ndarray, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(X.toarray())
        df['label'] = y
        df.to_csv(path, index=False)
        logging.info(f"Saved features to {path}")
    except Exception as e:
        logging.error(f"Failed to save features to {path}: {e}")
        raise

def main() -> None:
    try:
        params = read_params()
        max_features = params["feature_engg"]["max_features"]

        train_data = load_data("data/processed/train.csv")
        test_data = load_data("data/processed/test.csv")

        X_train, y_train = extract_features_labels(train_data)
        X_test, y_test = extract_features_labels(test_data)

        X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test, max_features)

        save_features(X_train_bow, y_train, "data/interim/train_bow.csv")
        save_features(X_test_bow, y_test, "data/interim/test_bow.csv")

        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()