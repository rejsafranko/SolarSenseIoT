import cv2
import numpy


class ImageProcessor:
    def __init__(self):
        pass

    def capture_image(self) -> numpy.ndarray:
        """Captures an image from Raspberry Pi camera using OpenCV."""
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image.")
            return None
        camera.release()
        return frame

    def preprocess_image(self, image: numpy.ndarray) -> numpy.ndarray:
        """Preprocess the captured image for model inference."""
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        return numpy.expand_dims(normalized_image, axis=0)
