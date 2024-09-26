import numpy
import keras


def load_model(model_path: str):
    return keras.models.load_model(model_path)


def preprocess_image(img_path: str, target_size=(224, 224)) -> numpy.ndarray:
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return numpy.expand_dims(img_array, axis=0)


def predict(model: keras.models.Sequential, img_array) -> float:
    prediction = model.predict(img_array)
    return prediction[0][0]


if __name__ == "__main__":
    model = load_model("models/mobilenetv2.h5")
    img_path = ""
    img_array = preprocess_image(img_path)
    prediction = predict(model, img_array)
    print(
        f"Predicted class: {'Dirty' if prediction > 0.5 else 'Clean'} with probability {prediction:.2f}"
    )
