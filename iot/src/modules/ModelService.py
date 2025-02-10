import logging

import numpy
import tensorflow
import tensorflow.lite.python.interpreter

from .ImageProcessor import ImageProcessor


class ModelService:
    def __init__(self, model_path: str, image_processor: ImageProcessor) -> None:
        self.image_processor = image_processor
        self._model = self._load_model(model_path=model_path)

    def _load_model(
        self, model_path: str
    ) -> tensorflow.lite.python.interpreter.Interpreter:
        """Loads the pre-trained TensorFlow model."""
        interpreter = tensorflow.lite.python.interpreter.Interpreter(model_path)
        interpreter.allocate_tensors()
        return interpreter

    def reload_model(self, model_path: str) -> None:
        """Reloads the model from the given path."""
        if model_path != self.model_path:
            logging.info(f"Reloading model from {model_path}")
            self._model = self._load_model(model_path)
            self.model_path = model_path
        else:
            logging.info(f"Model path is unchanged, skipping reload.")

    def run_inference(self, image: numpy.ndarray) -> 0 | 1:
        """Runs inference on the image and returns prediction (0 or 1)."""
        preprocessed_image = self.image_processor.preprocess_image(image)
        input_details = self._model.get_input_details()
        output_details = self._model.get_output_details()
        self._model.set_tensor(input_details[0]["index"], preprocessed_image)
        self._model.invoke()
        output_data = self._model.get_tensor(output_details[0]["index"])
        prediction = 1 if output_data[0] > 0.5 else 0
        return prediction
