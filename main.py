from predictor import predict
import sys
from io_utils import load_image
from io_utils import load_model

MODEL_PATH = "C:\\Users\\Alternanza\\Downloads\\keras_model.h5"


def main(argv: list):
    img_path = argv[1]
    img = load_image(img_path)
    model = load_model(MODEL_PATH)
    predicted_class, prediction_confidence = predict(img, model)
    print(f"prediction class: {predicted_class} and prediction confidence: {prediction_confidence*100}%")


if __name__ == "__main__":
    main(sys.argv)
