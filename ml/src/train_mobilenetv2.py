import argparse
import json
import logging
import os
import sys
import time
from typing import Union, Optional

import boto3
import dotenv
import numpy
import keras
import paho.mqtt.client as mqtt
import sklearn.metrics
import tensorflow
import tensorflow.lite
import wandb
import wandb.integration
import wandb.integration.keras
import wandb.sdk


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH")


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
        "MQTT_HOST": MQTT_HOST,
        "MQTT_TOPIC": MQTT_TOPIC,
        "BUCKET_NAME": BUCKET_NAME,
        "S3_MODEL_PATH": S3_MODEL_PATH,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")


def setup_tracking(config_path: str) -> wandb.sdk.Run:
    """Set up WandB tracking."""
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT_NAME, config=config_path)
    logging.info("WandB run initialized.")
    return run


def load_data(data_path: str) -> dict[str, numpy.ndarray]:
    try:
        data = numpy.load(data_path)
        logging.info(f"Data loaded from {data_path}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def build_model() -> keras.models.Sequential:
    base_model = keras.applications.MobileNetV2(
        input_shape=tuple(wandb.config["image_size"]),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = keras.models.Sequential(
        [
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dropout(wandb.config["dropout"]),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=wandb.config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logging.info("Model built successfully.")

    return model


def create_callbacks(
    model_save_path: str, patience: int
) -> list[Union[keras.callbacks.Callback, wandb.integration.keras.WandbCallback]]:
    """Create Keras callbacks for training."""
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    wandb_callback = wandb.integration.keras.WandbCallback(
        monitor="val_accuracy", save_graph=False, save_model=False
    )

    return [checkpoint_callback, early_stopping_callback, wandb_callback]


def train_model(
    model: keras.model.Sequential, model_name: str, data: dict[str, numpy.ndarray]
) -> keras.callbacks.History:
    callbacks = create_callbacks(
        model_save_path=f"models/{model_name}.keras", patience=wandb.config["patience"]
    )

    logging.info(f"Training started with the following configuration: {wandb.config}")

    history = model.fit(
        data["train_features"],
        data["train_labels"],
        batch_size=wandb.config["batch_size"],
        validation_data=(data["val_features"], data["val_labels"]),
        validation_batch_size=wandb.config["validation_batch_size"],
        epochs=wandb.config["epochs"],
        callbacks=callbacks,
    )

    logging.info("Model trained.")

    return history


def evaluate_model(
    model: keras.models.Sequential, data: dict[str, numpy.ndarray], run: wandb.sdk.Run
) -> float:
    true_labels = data["test_labels"]
    predicted_labels = (model.predict(data["test_features"]).flatten() > 0.5).astype(
        int
    )

    accuracy = sklearn.metrics.accuracy_score(true_labels, predicted_labels)
    precision = sklearn.metrics.precision_score(true_labels, predicted_labels)
    recall = sklearn.metrics.recall_score(true_labels, predicted_labels)
    f1 = sklearn.metrics.f1_score(true_labels, predicted_labels)

    run.log(
        {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
    )

    logging.info(
        f"Test accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )
    run.finish()

    return accuracy


def get_best_test_accuracy() -> float:
    api = wandb.Api()
    project_name = WANDB_PROJECT_NAME
    runs = api.runs(
        f"{wandb.Api().default_entity}/{project_name}",
        order="-summary_metrics.test_accuracy",
    )
    if runs:
        return runs[0].summary.get("test_accuracy", 0)
    return 0.0


def convert_to_tflite(model: keras.models.Model, model_name: str) -> str:
    output_path = f"models/{model_name}.tflite"
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    tensorflow_lite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tensorflow_lite_model)
    logging.info(f"Model converted and saved at {output_path}.")
    return output_path


def upload_to_s3(file_path: str, bucket_name: str, s3_path: str) -> None:
    try:
        s3 = boto3.client("s3")
        s3.upload_file(file_path, bucket_name, s3_path)
        logging.info(
            f"Model uploaded to S3: s3://{BUCKET_NAME}/{S3_MODEL_PATH}/mobilenetv2.tflite"
        )
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")
        raise


def setup_mqtt() -> mqtt.Client:
    mqtt_client = mqtt.Client()
    mqtt_client.tls_set()
    mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
    return mqtt_client


def notify_devices(model_url: str) -> None:
    try:
        mqtt_client = setup_mqtt()
        message = {"model_url": model_url, "timestamp": int(time.time())}
        mqtt_client.publish(MQTT_TOPIC, json.dumps(message), qos=1)
        mqtt_client.disconnect()
        logging.info("Model update notification sent to IoT devices.")
    except Exception as e:
        logging.error(f"Failed to send MQTT notification: {e}")
        raise


def main(args: argparse.Namespace):
    config_path: str = args.config_path
    data_path: str = args.data_path
    model_name: str = args.model_name
    os.makedirs("models", exist_ok=True)
    try:
        validate_env_vars()
        run = setup_tracking(config_path=config_path)
        data = load_data(data_path=data_path)

        model = build_model()
        train_model(model, data)

        accuracy = evaluate_model(model, data, run)
        best_previous_accuracy = get_best_test_accuracy()

        if accuracy > best_previous_accuracy:
            model_path = convert_to_tflite(model, model_name)
            upload_to_s3(model_path, BUCKET_NAME, S3_MODEL_PATH)
            notify_devices(
                f"https://{BUCKET_NAME}.s3.amazonaws.com/{S3_MODEL_PATH}/{model_name}.tflite"
            )
    except Exception as e:
        logging.error(f"An error occurred during the execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
