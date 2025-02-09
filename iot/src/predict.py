import logging
import os
import time

import dotenv

from modules import CameraService, ImageProcessor, ModelService, MQTTClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

dotenv.load_dotenv()
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = os.getenv("MQTT_PORT")
MQTT_TOPIC = os.getenv("MQTT_TOPIC")
ROOT_CA_PATH = os.getenv("ROOT_CA_PATH")
CERT_PATH = os.getenv("CERT_PATH")
KEY_PATH = os.getenv("KEY_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
DEVICE_ID = os.getenv("DEVICE_ID", "Raspberry Pi Sigma")


def validate_env_vars() -> None:
    """Ensure all necessary environment variables are set."""
    required_vars = {
        "MQTT_HOST": MQTT_HOST,
        "MQTT_TOPIC": MQTT_TOPIC,
        "MQTT_PORT": MQTT_PORT,
        "ROOT_CA_PATH": ROOT_CA_PATH,
        "CERT_PATH": CERT_PATH,
        "KEY_PATH": KEY_PATH,
        "MODEL_PATH": MODEL_PATH,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")


def create_payload(prediction: int) -> dict:
    """Dynamically create MQTT payload based on model prediction."""
    return {
        "device_id": DEVICE_ID,
        "prediction": prediction,
        "timestamp": int(time.time()),
    }


def capture_image_with_retry(camera_service: CameraService, retries: int = 1):
    attempt = 0
    while attempt <= retries:
        try:
            image = camera_service.capture_image()
            if image is not None:
                logging.info("Image captured successfully.")
                return image
            else:
                raise Exception("Captured image is None.")
        except Exception as e:
            attempt += 1
            logging.error(f"Image capture failed on attempt {attempt}. Error: {e}")
            if attempt > retries:
                raise Exception("Image capture failed after retry attempts.")
            time.sleep(2)


def run_prediction_with_retry(model_service: ModelService, image, retries: int = 1):
    attempt = 0
    while attempt <= retries:
        try:
            prediction = model_service.run_inference(image)
            logging.info(f"Prediction: {prediction}")
            return prediction
        except Exception as e:
            attempt += 1
            logging.error(f"Prediction failed on attempt {attempt}. Error: {e}")
            if attempt > retries:
                raise Exception("Prediction failed after retry attempts.")
            time.sleep(2)


def main():
    try:
        validate_env_vars()

        model_service = ModelService(
            model_path=MODEL_PATH, image_processor=ImageProcessor()
        )

        mqtt_client = MQTTClient(
            MQTT_HOST,
            int(MQTT_PORT),
            ROOT_CA_PATH,
            CERT_PATH,
            KEY_PATH,
            MQTT_TOPIC,
            MODEL_PATH,
            model_service,
        )
        mqtt_client.connect()

        while True:
            camera_service = CameraService()

            try:
                image = capture_image_with_retry(camera_service, retries=1)
                logging.info("Image captured.")
            except Exception as e:
                logging.error(f"Image capture failed: {e}")
                continue

            try:
                prediction = run_prediction_with_retry(model_service, image, retries=1)
                logging.info(f"Model prediction: {prediction}")
            except Exception as e:
                logging.error(f"Prediction failed: {e}")
                continue

            if prediction == 1:
                payload = create_payload(prediction)
                mqtt_client.publish(payload)
            else:
                logging.info(f"Prediction was {prediction}. No MQTT message sent.")

            time.sleep(86400)

    except Exception as e:
        logging.error(f"Error occurred during deployment: {e}", exc_info=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
