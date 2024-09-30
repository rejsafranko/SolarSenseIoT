import keras
import numpy

from ImageProcessor import ImageProcessor


class ModelService:
    def __init__(self, image_processor: ImageProcessor):
        self.image_processor = image_processor
        self._model = self._load_model()

    def _load_model(self, model_path: str) -> keras.models.Sequential:
        """Loads the pre-trained TensorFlow model."""
        return keras.models.load_model(model_path)

    def run_inference(self, image: numpy.ndarray) -> 0 | 1:
        """Runs inference on the image and returns prediction (0 or 1)."""
        preprocessed_image = self.image_processor.preprocess_image(image)
        prediction = self._model.predict(preprocessed_image)
        return prediction[0][0]
