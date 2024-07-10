import cv2
import numpy as np


def preprocces_image(img: np.array, altezza: int, larghezza: int) -> np.array:
    image_resized = resize(img,altezza,larghezza)
    image_normalized = (image_resized / 127.5) - 1
    image_preprocessed = to_tensor(image_normalized)
    return image_preprocessed


def resize(img: np.array, altezza: int, larghezza: int) -> np.array:
    image_resized = cv2.resize(img, (larghezza, altezza))
    return image_resized


def to_tensor(img: np.array) -> np.array:
    num_images = 1
    altezza = img.shape[0]
    larghezza = img.shape[1]
    num_color_channels = img.shape[2]
    tensore = (np.asarray(img, dtype=np.float32).reshape(num_images,larghezza,altezza,num_color_channels))
    return tensore