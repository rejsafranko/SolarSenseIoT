import os
import numpy
import keras
import wandb
import sklearn.metrics

from typing import Dict
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def setup_tracking() -> None:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="solar", config="configs/config-defaults.yaml", job_type="evaluation")


def load_model(model_path: str) -> keras.models.Sequential:
    return keras.models.load_model(model_path)


def load_test_data(test_data_path: str) -> Dict[str, Dict[str, numpy.ndarray]]:
    data = numpy.load(test_data_path)
    return data["test_features"], data["test_labels"]


def evaluate_model(model: keras.models.Sequential, test_features, test_labels) -> Dict[str, float]:
    predictions = model.predict(test_features)
    predicted_classes = (predictions > 0.5).astype(int)

    accuracy = sklearn.metrics.accuracy_score(test_labels, predicted_classes)
    precision = sklearn.metrics.precision_score(test_labels, predicted_classes)
    recall = sklearn.metrics.recall_score(test_labels, predicted_classes)
    f1 = sklearn.metrics.f1_score(test_labels, predicted_classes)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


if __name__ == "__main__":
    setup_tracking()
    
    model = load_model("models/mobilenetv2.keras")
    test_features, test_labels = load_test_data("data/data_splits.npz")
    
    metrics = evaluate_model(model, test_features, test_labels)
    wandb.log(metrics)
    wandb.finish()
    
    print(f"Accuracy: {metrics["accuracy"]:.4f}")
    print(f"Precision: {metrics["precision"]:.4f}")
    print(f"Recall: {metrics["recall"]:.4f}")
    print(f"F1 Score: {metrics["f1"]:.4f}")
