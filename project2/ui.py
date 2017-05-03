

def ui():
    print("Welcome to the Cuisine Prediction System!")
    print("Please enter 1 to proceed with Cuisine Prediction.")
    print("Enter 0 to Exit the System.")
    num = 99
    while num != 0:
        num = int(str(input('--> ')))
        if num == 1:
            print("You selected Cuisine Prediction")
        elif num == 0:
            print("Thanks for using the Cuisine Predictor")
        else:
            print("Please enter a valid input!")

