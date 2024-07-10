from predictor import test_image

from io_utils import load_all_images


def test_all_image(folder_class1: str, folder_class2: str, class1: str, class2: str) -> (int, int):

    image = load_all_images(folder_class1, class1) + load_all_images(folder_class2, class2)

    num_correct_predictions = 0
    for image_path, true_label in image:
        predicted_class, confidence_score = test_image(image_path, true_label)
        if predicted_class == true_label:
            num_correct_predictions += 1
    return num_correct_predictions, len(image)
