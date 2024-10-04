import cv2
import numpy
import picamera2


class CameraService:
    def __init__(self):
        self._configure_camera()

    def _configure_camera(self):
        self.camera = picamera2.Picamera2()
        config = self.camera.still_configuration()
        self.camera.configure(config)

    def capture_image(self) -> numpy.ndarray:
        """Captures an image from Raspberry Pi camera using OpenCV."""
        image = self.camera.capture_array()
        print(image.shape)
        exit(0)
        if not image:
            print("Failed to capture image.")
            return None
        return image

    def dummy_image(self) -> numpy.ndarray:
        image = cv2.imread(
            "/home/raspberrypi/Projects/solar/SolarSense/iot/Imgdirty_5_1.jpg",
            cv2.IMREAD_COLOR,
        )
        return image
