import validation_utils as vu
import sys


def validate(path: str) -> None:

    folder_class1 = 'C:\\Users\\Alternanza\\Downloads\\moto'
    class1 = "moto"
    folder_class2 = 'C:\\Users\\Alternanza\\Downloads\\bici'
    class2 = "bici"

    correct_predition, tot_image = vu.test_all_image(folder_class1, folder_class2, class1, class2)
    accuracy = correct_predition / tot_image
    with open(path, 'w') as file:
        file.write(f"Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    validate(sys.argv[0])
