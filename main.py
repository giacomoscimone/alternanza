import predictor as pr
import sys


def main(argv):
    image = argv[1]
    # 'C:\\Users\\Alternanza\\Downloads\\bici3.jpg'

    predicted_class, prediction_confidence = pr.test_image(image, "bici")
    print("classe: ", predicted_class, f"confidence:{prediction_confidence * 100} %")


if __name__ == "__main__":
    main(sys.argv[0:])
