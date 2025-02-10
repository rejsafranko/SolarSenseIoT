import os
import argparse
import numpy
import keras
import sklearn.model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    return parser.parse_args()


def load_images_from_directory(
    directory: str, label: int, target_size=(224, 224)
) -> list[tuple[numpy.ndarray, int]]:
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img_path = os.path.join(directory, filename)
            img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1].
            images.append((img_array, label))
    return images


def load_data(
    data_dir: str, target_size=(224, 224), test_size=0.2, val_size=0.1
) -> dict[str, dict[str, numpy.ndarray]]:
    clean_dir = os.path.join(data_dir, "merged", "clean")
    dirty_dir = os.path.join(data_dir, "merged", "dirty")

    clean_images = load_images_from_directory(
        clean_dir, label=0, target_size=target_size
    )
    dirty_images = load_images_from_directory(
        dirty_dir, label=1, target_size=target_size
    )

    all_images = clean_images + dirty_images
    numpy.random.shuffle(all_images)

    features, labels = zip(*all_images)
    features = numpy.array(features)
    labels = numpy.array(labels)

    x_train, x_temp, y_train, y_temp = sklearn.model_selection.train_test_split(
        features, labels, test_size=test_size, random_state=42
    )

    val_size_adjusted = val_size / (1 - test_size)
    x_val, x_test, y_val, y_test = sklearn.model_selection.train_test_split(
        x_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )

    numpy.savez(
        f"{data_dir}data_splits.npz",
        train_features=x_train,
        train_labels=y_train,
        val_features=x_val,
        val_labels=y_val,
        test_features=x_test,
        test_labels=y_test,
    )


def main(args: argparse.Namespace) -> None:
    load_data(args.data_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
