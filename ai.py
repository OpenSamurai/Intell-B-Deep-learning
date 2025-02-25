import os
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab
from game_control import *
from predict import predict
from game_control import *
from tensorflow.keras.models import model_from_json
import tensorflow as tf
print(tf.__version__)

def main():
    # Get Model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    model.load_weights("Data/Model/weights.weights.h5")

    print('AI start now!')

    while True:
        # Get screenshot:
        screen = ImageGrab.grab()
        # Image to numpy array:
        screen = np.array(screen)
        # 4 channel(PNG) to 3 channel(JPG)
        Y = predict(model, screen)
        
        # Convert model output to action array (modify this based on your actual class mapping)
        action = np.argmax(Y)  # Get the predicted class index from 60 possible classes
        
        # Handle different actions based on the predicted class
        if action == 0:  # Assuming class 0 is "no action"
            continue
        elif action == 1:  # Example: keyboard action
            key = get_key(1)  # Modify with actual key mapping
            press(key)
            sleep(0.1)
            release(key)
        elif action == 2:  # Example: mouse action
            click(100, 200)  # Modify with actual coordinates
        elif action == 3:  # Another action
            # Add more actions as needed
            pass
        # Add more cases for other actions...

if __name__ == '__main__':
    main()
