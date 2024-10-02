import os
import ssl
import time
import json
import keras
import numpy
import cv2
import _thread
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from modules import CameraService, ImageProcessor, ModelService

load_dotenv()
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = os.getenv("MQTT_PORT")
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
ROOT_CA_PATH = os.getenv("ROOT_CA_PATH")
CERT_PATH = os.getenv("CERT_PATH")
KEY_PATH = os.getenv("KEY_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
print(MQTT_HOST)
print(MQTT_PORT)
print(MQTT_TOPIC)
print(ROOT_CA_PATH)
print(CERT_PATH)
print(KEY_PATH)
print(MODEL_PATH)

payload = {"device_id": "Raspberry Pi Sigma"}


def on_connect(client, userdata, flags, rc, properties):
    print(f"Connected to AWS IoT: {rc}")


def on_publish(client, userdata, mid, reason_codes, properties) -> None:
    print("Message published.")


def publish_mqtt_message(client: mqtt.Client):
    """Publishes an MQTT message to trigger the AWS IoT notification."""
    client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)


if __name__ == "__main__":
    camera_service = CameraService()
    model_service = ModelService(
        model_path=MODEL_PATH, image_processor=ImageProcessor()
    )
    prediction = 1
    if prediction == 1:
        mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqttc.tls_set(
            ca_certs=ROOT_CA_PATH,
            certfile=CERT_PATH,
            keyfile=KEY_PATH,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )
        mqttc.tls_insecure_set(True)
        mqttc.on_connect = on_connect
        mqttc.on_publish = on_publish
        mqttc.connect(MQTT_HOST, int(MQTT_PORT), keepalive=60)
        mqttc.loop_start()
        time.sleep(5)
        publish_mqtt_message(mqttc)
        time.sleep(20)
        mqttc.loop_stop()
        mqttc.disconnect()
    else:
        print("Prediction was 0. No MQTT message sent.")
