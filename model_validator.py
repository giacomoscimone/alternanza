import validation_utils as vu
import sys
from io_utils import load_all_images
from io_utils import load_model
import logging

logging.basicConfig(filename='C:\\Users\\Alternanza\\Downloads\\log_model_validator.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
original = 'C:\\Users\\Alternanza\\Downloads'
folder_class1 = original + '\\moto'
class1 = "moto"
folder_class2 = original + '\\bici'
class2 = "bici"

MODEL_PATH = original + '\\keras_model.h5'


def validate(argv: list) -> None:
    path = argv[1]
    logging.info("inizio programma")
    logging.info(f"path ricevuto: {path}")

    model = load_model(MODEL_PATH)
    logging.info("modello caricato")

    image_paths_with_labels = load_all_images(folder_class1, class1) + load_all_images(folder_class2, class2)
    logging.info(f"immagini caricate dalle cartelle {folder_class1}  {folder_class2}")

    correct_predition, tot_image = vu.test_all_image(model, image_paths_with_labels)
    logging.info("fine test immagini")

    accuracy = correct_predition / tot_image
    with open(path, 'w') as file:
        file.write(f"Accuracy: {accuracy * 100}%")
    logging.info("fine programma")


if __name__ == "__main__":
    validate(sys.argv)
