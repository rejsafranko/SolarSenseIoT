import cv2
import numpy


class CameraService:
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
