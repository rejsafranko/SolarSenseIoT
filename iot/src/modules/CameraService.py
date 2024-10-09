import cv2
import numpy
import picamera2


class CameraService:
    def __init__(self):
        self.camera = picamera2.Picamera2()
        print("Initialized camera.")
        self._configure_camera()

    def _configure_camera(self):
        """Configures the camera for still image capture."""
        still_config = self.camera.create_still_configuration()
        self.camera.configure(still_config)
        print("Configured camera for still image capture.")

    def capture_image(self) -> numpy.ndarray:
        """Captures an image from Raspberry Pi camera using OpenCV."""
        self.camera.start()
        image = self.camera.capture_array()
        print("Image captured.")
        self.camera.stop()
        if image is None or image.size == 0:
            print("Failed to capture image.")
            return None
        return image

    def dummy_image(self) -> numpy.ndarray:
        image = cv2.imread(
            "/home/raspberrypi/Projects/solar/SolarSense/iot/Imgdirty_5_1.jpg",
            cv2.IMREAD_COLOR,
        )
        return image
