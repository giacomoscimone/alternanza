import logging
from typing import Any
import cv2
import numpy as np
from image_utils import preproces_image
from predictor import predict
from io_utils import load_image


def test_all_image(model: Any, image_path_with_labels: list) -> (int, int):

    num_correct_predictions = 0
    logging.info("inizio test immagini")

    for image_path, true_label in image_path_with_labels:
        img = load_image(image_path)
        img_preprocessed = preproces_image(img)
        test_result = test_image(model, img_preprocessed, true_label)

        if test_result is True:
            num_correct_predictions += 1
            logging.debug(f"test immagine riuscita file {image_path}")
    logging.info("fine test immagini")

    return num_correct_predictions, len(image_path_with_labels)


def test_image(model: Any, img: np.array, true_label: str) -> bool:
    logging.info("inizio predizione")
    prediction, confidence = predict(img, model)
    logging.info("predizione terminata")
    return prediction == true_label

