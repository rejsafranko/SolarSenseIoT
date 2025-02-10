import argparse
import logging
import os
import sys

import dotenv
import keras
import numpy
import sklearn.metrics
import wandb
import wandb.sdk

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate a model, and deploy to IoT devices."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration yaml for training parameters.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name to save the model as."
    )
    return parser.parse_args()


def validate_env_vars() -> None:
    """Ensure all necessary environment variables are set."""
    required_vars = {
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_PROJECT_NAME": WANDB_PROJECT_NAME,
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

def setup_tracking(config_path: str) -> wandb.sdk.wandb_run.Run:
    """Set up WandB tracking."""
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT_NAME, config=config_path)
    logging.info("WandB run initialized.")
    return run


def load_model(model_path: str) -> keras.models.Sequential:
    return keras.models.load_model(model_path)


def load_test_data(test_data_path: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    try:
        data = numpy.load(test_data_path)
        logging.info(f"Test data loaded from {test_data_path}.")
        return data["test_features"], data["test_labels"]
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise


def evaluate_model(model: keras.models.Sequential, test_features: numpy.ndarray, test_labels:numpy.ndarray) -> dict[str, float]:
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


def main(args:argparse.Namespace):
    config_path: str = args.config_path
    data_path: str = args.data_path
    model_name: str = args.model_name
    
    try:
        validate_env_vars()
        run: wandb.sdk.wandb_run.Run = setup_tracking(config_path)
        model = load_model(f"models/{model_name}.keras")
        test_features, test_labels = load_test_data(data_path)
        metrics = evaluate_model(model, test_features, test_labels)
        run.log(metrics)
        run.finish()
        
        print(f"Accuracy: {metrics["accuracy"]:.4f}")
        print(f"Precision: {metrics["precision"]:.4f}")
        print(f"Recall: {metrics["recall"]:.4f}")
        print(f"F1 Score: {metrics["f1"]:.4f}")
    except Exception as e:
        logging.error(f"An error occurred during the execution: {e}", exc_info=True)
        sys.exit(1)

if __name__=="__main__":
    args = parse_args()
    main(args)