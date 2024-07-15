from typing import Any
import numpy as np
import logging

classes = ["bici", "moto"]


def predict(img: np.array, model: Any) -> tuple[str, float]:
    prediction = model.predict(img)
    logging.debug("predizione completata")
    index = np.argmax(prediction)
    predicted_class = classes[index]
    prediction_confidence = prediction[0][index]
    return predicted_class, prediction_confidence

