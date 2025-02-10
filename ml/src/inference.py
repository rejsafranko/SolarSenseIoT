import argparse

import numpy
import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate a model, and deploy to IoT devices."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model."
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to image to run inference on."
    )
    return parser.parse_args()


def load_model(model_path: str) -> keras.Sequential:
    return keras.models.load_model(model_path)


def preprocess_image(img_path: str, target_size=(224, 224)) -> numpy.ndarray:
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return numpy.expand_dims(img_array, axis=0)


def predict(model: keras.models.Sequential, img_array) -> float:
    prediction = model.predict(img_array)
    return prediction[0][0]


def main(args: argparse.Namespace) -> None:
    model_name: str = args.model_name
    img_path: str = args.img_path
    model = load_model(f"models/{model_name}")
    img_array = preprocess_image(img_path)
    prediction = predict(model, img_array)
    print(
        f"Predicted class: {'Dirty' if prediction > 0.5 else 'Clean'} with probability {prediction:.2f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
