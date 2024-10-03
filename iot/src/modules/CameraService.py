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

    def dummy_image(self) -> numpy.ndarray:
        image = cv2.imread(
            "/home/raspberrypi/Projects/solar/SolarSense/iot/20210916_094136_4_11zon.jpg",
            cv2.IMREAD_COLOR,
        )
        return image
