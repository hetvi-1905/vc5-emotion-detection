import logging
import pandas as pd
import pickle
import json
from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)

def load_model(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {path}: {e}")
        raise

def load_test_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Test data loaded from {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load test data from {path}: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics calculated: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save metrics to {path}: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()