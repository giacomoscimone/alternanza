import validation_utils as vu



def validate():

    correct_predition, tot_image = vu.test_all_image()
    accuracy =  correct_predition / tot_image
    with open('C:\\Users\\Alternanza\\Downloads\\accuracy.txt','w') as file:
        file.write(f"Accuracy: {accuracy*100}%")


if __name__ == "__main__":
    validate()