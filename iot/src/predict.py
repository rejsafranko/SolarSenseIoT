import os
import ssl
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = os.getenv("MQTT_PORT")
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
ROOT_CA_PATH = os.getenv("ROOT_CA_PATH")
CERT_PATH = os.getenv("CERT_PATH")
KEY_PATH = os.getenv("KEY_PATH")

payload = {"device_id": "IoTDevice001"}


def on_connect(client: mqtt.Client, userdata: dict, flags: dict, rc: int) -> None:
    if rc == 0:
        print("Connected to MQTT broker.")
        client.publish(MQTT_TOPIC, str(payload), qos=1)
    else:
        print(f"Failed to connect with error code {rc}")


def on_publish(client: mqtt.Client, userdata: dict, mid: int) -> None:
    print("Message published.")


if __name__ == "__main__":
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
    time.sleep(5)
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
