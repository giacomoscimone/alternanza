import logging
import os
from typing import Any

import cv2
import numpy as np
from keras.models import load_model as load_tf_model


def load_model(path: str) -> Any:
    model = load_tf_model(path, compile=False)
    return model


def load_image(path: str) -> np.array:
    image = cv2.imread(path)
    return image


def save_image(img: np.array, path: str) -> None:
    cv2.imwrite(path, img)


def load_all_images(folder_path: str, true_label: str) -> list:
    image_paths = []
    logging.info("inizio caricamento immagini")

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        image_paths.append((full_path, true_label))
        logging.debug(f"caricamento immagine da {full_path}")

    logging.info("termine caricamento immagini")
    return image_paths


def print_prediction(class_name, prediction):
    # TODO: valutare riscrittura/ cancellazione
    return f"L'immagine caricata e' un/una {class_name} prediction: {prediction * 100}%"
