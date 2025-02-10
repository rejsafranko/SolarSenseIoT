import json
import logging
import ssl
import time

import boto3
import paho.mqtt.client as mqtt

from .ModelService import ModelService


class MQTTClient:
    def __init__(
        self,
        host: str,
        port: int,
        root_ca: str,
        cert: str,
        key: str,
        topic: str,
        model_path: str,
        model_service: ModelService,
    ):
        self.host = host
        self.port = port
        self.topic = topic
        self.model_path = model_path
        self.model_service = model_service
        self.client = mqtt.Client()
        self.client.tls_set(
            ca_certs=root_ca,
            certfile=cert,
            keyfile=key,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )
        self.client.tls_insecure_set(True)
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish

    def on_connect(self, client, userdata, flags, rc, properties) -> None:
        """Handle successful connection."""
        logging.info(f"Connected to AWS IoT with result code {rc}")

    def on_message(self, client, userdata, msg) -> None:
        """Handle incoming MQTT message."""
        logging.info(f"Received message: {msg.payload.decode()}")
        try:
            payload = json.loads(msg.payload.decode())
            if "model_url" in payload:
                model_url = payload["model_url"]
                logging.info(f"New model URL received: {model_url}")
                self.update_model_from_s3(model_url)
            else:
                logging.warning("Received message without model_url.")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def on_publish(self, client, userdata, mid, reason_codes, properties) -> None:
        """Handle message publishing event."""
        logging.info(f"Message published with mid: {mid}")

    def connect(self) -> None:
        """Connect to the MQTT broker."""
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            logging.info(f"Connecting to {self.host}:{self.port}")
            self.client.loop_start()
            time.sleep(1)  # Allow time for connection to establish
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        logging.info("Disconnecting from MQTT broker")
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, payload: dict) -> None:
        """Publish a message to the specified topic."""
        try:
            self.client.publish(self.topic, json.dumps(payload), qos=2)
            logging.info("Published message to MQTT topic.")
        except Exception as e:
            logging.error(f"Failed to publish message: {e}")
            raise

    def update_model_from_s3(self, model_url: str) -> None:
        """Downloads a new model from S3 and updates the local model."""
        try:
            model_url_parts = model_url.split("/")
            bucket_name = model_url_parts[2]
            model_key = "/".join(model_url_parts[3:])

            s3 = boto3.client("s3")
            s3.download_file(bucket_name, model_key, self.model_path)
            self.model_service.reload_model(self.model_path)
            logging.info(f"Model updated successfully from {model_url}")
        except Exception as e:
            logging.error(f"Failed to update model: {e}")
