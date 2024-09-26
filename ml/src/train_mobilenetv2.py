import os
import numpy
import keras
import wandb
import wandb.integration
import wandb.integration.keras

from typing import List
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def setup_tracking() -> None:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="solar", config="configs/config-defaults.yaml", job_type="training")


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

    return model


def create_callbacks(
    model_save_path: str, patience: int
) -> List[keras.callbacks.Callback | wandb.integration.keras.WandbCallback]:
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

    wandb_callback = wandb.integration.keras.WandbCallback(monitor="val_accuracy", save_graph=False, save_model=False)

    return [checkpoint_callback, early_stopping_callback, wandb_callback]


if __name__ == "__main__":
    setup_tracking()
    data = numpy.load("data/data_splits.npz")

    model = build_model()

    callbacks = create_callbacks(
        model_save_path="models/mobilenetv2.keras", patience=wandb.config["patience"]
    )

    history = model.fit(
        data["train_features"],
        data["train_labels"],
        batch_size=wandb.config["batch_size"],
        validation_data=(data["val_features"], data["val_labels"]),
        validation_batch_size=wandb.config["validation_batch_size"],
        epochs=wandb.config["epochs"],
        callbacks=callbacks,
    )

    wandb.finish()
