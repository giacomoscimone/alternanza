import logging

import cv2
import numpy as np

DEFAULT_ALTEZZA = 224
DEFAULT_LARGHEZZA = 224


def preproces_image(img: np.array) -> np.array:
    image_resized = resize(img, DEFAULT_ALTEZZA, DEFAULT_LARGHEZZA)
    logging.debug("immagine ridimensionata")
    image_normalized = normalize(image_resized)
    logging.debug("immagine normalizzata")
    image_preprocessed = to_tensor(image_normalized)
    logging.debug("immagine trasformata in tensore")
    return image_preprocessed


def resize(img: np.array, altezza: int, larghezza: int) -> np.array:
    try:
        image_resized = cv2.resize(img, (larghezza, altezza))
    except Exception as e:
        raise Exception("Resize error") from e
    return image_resized


def to_tensor(img: np.array) -> np.array:
    num_images = 1
    altezza = img.shape[0]
    larghezza = img.shape[1]
    num_color_channels = img.shape[2]
    tensore = np.asarray(img, dtype=np.float32).reshape(num_images, larghezza, altezza, num_color_channels)
    return tensore


def normalize(img: np.array):
    image_normalized = (img / 127.5) - 1
    return image_normalized


def show_image(img: np.array, img_title: str) -> None:
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grayscale(img: np.array) -> np.array:
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grayscale
