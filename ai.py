import os  # For interacting with the operating system
import platform  # For detecting the operating system
import numpy as np  # For handling numerical operations
from time import sleep  # To introduce delays in execution
from PIL import ImageGrab  # For capturing screenshots
from game_control import *  # Importing game control functions (like key presses and mouse actions)
from predict import predict  # Importing the prediction function
from tensorflow.keras.models import model_from_json  # To load a model from a JSON file
import tensorflow as tf  # Importing TensorFlow for deep learning

# Print TensorFlow version to check compatibility
print(tf.__version__)

def main():
    # Load the AI model
    model_file = open('Data/Model/model.json', 'r')  # Open the model JSON file
    model = model_file.read()  # Read model structure
    model_file.close()  # Close the file
    model = model_from_json(model)  # Convert JSON to TensorFlow model
    model.load_weights("Data/Model/weights.weights.h5")  # Load the pre-trained weights

    print('AI start now!')  # Indicate the AI is running

    while True:  # Infinite loop to continuously process frames
        # Capture a screenshot of the current screen
        screen = ImageGrab.grab()
        
        # Convert the screenshot into a NumPy array (for image processing)
        screen = np.array(screen)

        # Convert 4-channel image (PNG) to 3-channel image (JPG) if necessary
        Y = predict(model, screen)  # Use the trained model to predict an action
        
        # Convert model output to an action index (assuming 60 possible actions)
        action = np.argmax(Y)  # Get the predicted class index
        
        # Perform an action based on the predicted class
        if action == 0:  # If class 0 is "no action," continue without doing anything
            continue
        elif action == 1:  # If class 1 is a keyboard action
            key = get_key(1)  # Get the corresponding key for this action
            press(key)  # Simulate a key press
            sleep(0.1)  # Hold the key for a short time
            release(key)  # Release the key
        elif action == 2:  # If class 2 is a mouse click action
            click(100, 200)  # Click at coordinates (100, 200)
        elif action == 3:  
            pass  # Placeholder for additional actions

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
