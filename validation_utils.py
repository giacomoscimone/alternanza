
from predictor import test_image

from io_utils import load_all_images

def test_all_image():
    folder_class1 = 'C:\\Users\\Alternanza\\Downloads\\moto'
    class1 = "moto"
    folder_class2 = 'C:\\Users\\Alternanza\\Downloads\\bici'
    class2 = "bici"
    image = load_all_images(folder_class1, class1) + load_all_images(folder_class2, class2)

    num_correct_predictions = 0
    for image_path, true_label in image:
        predicted_class , confidence_score = test_image(image_path,true_label)
        if(predicted_class == true_label ):
            num_correct_predictions += 1
    return num_correct_predictions,len(image)



