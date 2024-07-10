import os
import cv2
import numpy as np


def load_image(path: str) -> np.array:
    image = cv2.imread(path)
    return image


def show_image(img: np.array, img_title: str) -> None:
    cv2.imshow(img_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img: np.array, path: str) -> (str, np.array):
    return cv2.imwrite(path, img)


def load_all_images(path: str, true_label: str) -> np.array:
    image_paths = []
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        image_paths.append((full_path, true_label))
    return image_paths
