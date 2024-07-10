import os
import cv2


def load_image(path):
    image = cv2.imread(path)
    return image

def show_image(img):
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img,path):
    return cv2.imwrite(path,img)


def load_all_images(path,true_label):
    image_paths = []
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        image_paths.append((full_path, tag))
    return image_paths