import os
import ssl
import time
import keras
import numpy
import cv2
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = os.getenv("MQTT_PORT")
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
ROOT_CA_PATH = os.getenv("ROOT_CA_PATH")
CERT_PATH = os.getenv("CERT_PATH")
KEY_PATH = os.getenv("KEY_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")

payload = {"device_id": "IoTDevice001"}


def on_connect(rc: int, **kwargs) -> None:
    if rc == 0:
        print("Connected to MQTT broker.")
    else:
        print(f"Failed to connect with error code {rc}")


def on_publish(**kwargs) -> None:
    print("Message published.")


def publish_mqtt_message(client: mqtt.Client):
    """Publishes an MQTT message to trigger the AWS IoT notification."""
    client.publish(MQTT_TOPIC, str(payload), qos=1)


def capture_image() -> numpy.ndarray:
    """Captures an image from Raspberry Pi camera using OpenCV."""
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image.")
        return None
    camera.release()
    return frame


def load_model(model_path: str) -> keras.models.Sequential:
    """Loads the pre-trained TensorFlow model."""
    return keras.models.load_model(model_path)


def preprocess_image(image: numpy.ndarray) -> numpy.ndarray:
    """Preprocess the captured image for model inference."""
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return numpy.expand_dims(normalized_image, axis=0)


def run_inference(model: keras.models.Sequential, image: numpy.ndarray) -> 0 | 1:
    """Runs inference on the image and returns prediction (0 or 1)."""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction[0][0]


if __name__ == "__main__":
    model = load_model()
    image = capture_image()
    if image is not None:
        prediction = run_inference(model, image)
        print(f"Prediction: {'dirty' if 1 else 'clean'}")
        if prediction == 1:
            mqtt_client = mqtt.Client()
            mqtt_client.tls_set(
                ca_certs=ROOT_CA_PATH,
                certfile=CERT_PATH,
                keyfile=KEY_PATH,
                tls_version=ssl.PROTOCOL_TLSv1_2,
            )

            mqtt_client.on_connect = on_connect
            mqtt_client.on_publish = on_publish
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            mqtt_client.loop_start()
            publish_mqtt_message(mqtt_client)
            time.sleep(5)
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        else:
            print("Prediction was 0. No MQTT message sent.")
