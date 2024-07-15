from typing import Any

import numpy as np
from image_utils import preproces_image

classes = ["bici", "moto"]


def predict(img: np.array, model: Any) -> tuple[str, float]:
    prediction = model.predict(img)
    index = np.argmax(prediction)
    predicted_class = classes[index]
    prediction_confidence = prediction[0][index]
    return predicted_class, prediction_confidence

