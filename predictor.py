from typing import Any

import numpy as np
from keras.models import load_model as load_tf_model
from image_utils import preprocces_image
import cv2


def load_model(path: Any) -> str:
    model = load_tf_model(path, compile=False)
    return model


def predict(img: np.array) -> (str, float):
    modello = load_model('C:\\Users\\Alternanza\\Downloads\\keras_model.h5')
    prediction = modello.predict(img)
    index = np.argmax(prediction)
    class_names = ["bici", "moto"]
    predicted_class = class_names[index]
    prediction_confidence = prediction[0][index]
    return predicted_class, prediction_confidence


def test_image(img: np.array, true_label: str) -> (str, int):
    img = cv2.imread(img)
    img_preprocessed = preprocces_image(img, 224, 224)
    predicted_class, confidence_score = predict(img_preprocessed)
    return predicted_class, confidence_score
